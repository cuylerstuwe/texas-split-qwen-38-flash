"""Leased Texas router holds; MLX ownership stays entirely inside Sushi."""
from __future__ import annotations
import ctypes, math, threading, time, uuid
from pathlib import Path

FIELDS = ('seq','hold','state','ack','bytes','pauses','tokens','prompt','phase',
          'park_error','restore_wait','freed','available')

class IPC:
    def __init__(self, path):
        self.path = Path(path)
        self.lib = ctypes.CDLL(str(Path(__file__).with_name('texas_ipc.dylib')))
        self.lib.texas_ipc_open.argtypes = [ctypes.c_char_p,ctypes.c_int]
        self.lib.texas_ipc_open.restype = ctypes.c_void_p
        self.lib.texas_ipc_get.argtypes = [ctypes.c_void_p,ctypes.c_uint]
        self.lib.texas_ipc_get.restype = ctypes.c_uint64
        self.lib.texas_ipc_set.argtypes = [ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint64]
        self.lib.texas_ipc_close.argtypes = [ctypes.c_void_p]
        self.page = self.lib.texas_ipc_open(bytes(self.path),1)
        if not self.page: raise RuntimeError('Cannot create exclusive private Texas control page')
    def get(self, field): return self.lib.texas_ipc_get(self.page,FIELDS.index(field))
    def set(self, field, value): self.lib.texas_ipc_set(self.page,FIELDS.index(field),value)
    def close(self):
        self.lib.texas_ipc_close(self.page);self.page=None
        self.path.unlink(missing_ok=True)

class Controller:
    def __init__(self, ipc):
        self.ipc = ipc
        self.condition = threading.Condition(threading.RLock())
        self.owners, self.tickets = {}, {}
        self.stop = threading.Event()
        self.seq = 0
        self.thread = threading.Thread(target=self.maintain,daemon=True)
        self.thread.start()
    def held(self):
        now=time.monotonic()
        return any(v[1] and v[2]>now for v in self.owners.values()) or 'foreground' in self.tickets.values()
    def publish(self, force=False):
        hold=int(self.held())
        if force or hold!=self.ipc.get('hold'):
            self.seq+=1
            self.ipc.set('hold',hold)
            self.ipc.set('seq',self.seq)
            self.condition.notify_all()
        return self.seq
    def maintain(self):
        while not self.stop.wait(.02):
            with self.condition:self.publish()
    def wait_drained(self, seq):
        deadline=time.monotonic()+25
        with self.condition:
            while self.ipc.get('ack')<seq or self.ipc.get('state') not in (0,3):
                if self.stop.is_set() or time.monotonic()>=deadline:
                    raise TimeoutError('Native owner did not acknowledge a completed GPU boundary')
                self.condition.wait(.01)
    def control(self, data):
        if not isinstance(data,dict):raise ValueError('expected object')
        owner,epoch,wanted=data.get('owner'),data.get('epoch'),data.get('foreground')
        ttl=data.get('lease_seconds',30)
        if (not isinstance(owner,str) or not 1<=len(owner)<=128 or type(epoch) is not int
            or not 0<=epoch<=10**12 or type(wanted) is not bool or type(ttl) not in (int,float)
            or not math.isfinite(ttl) or not 1<=ttl<=120):
            raise ValueError('invalid owner/epoch/foreground/lease_seconds')
        with self.condition:
            old=self.owners.get(owner)
            if old and (epoch<old[0] or epoch==old[0] and wanted!=old[1]):
                raise ValueError('stale or conflicting epoch')
            if old is None and len(self.owners)>=1024:raise ValueError('owner table full')
            self.owners[owner]=(epoch,wanted,time.monotonic()+ttl)
            seq=self.publish(force=True)
        if wanted:self.wait_drained(seq)
        with self.condition:
            current=self.owners[owner]
            if current[0]!=epoch or current[1]!=wanted or wanted and current[2]<=time.monotonic():
                raise TimeoutError('lease expired or changed before acknowledgement')
        return dict(owner=owner,epoch=epoch,foreground=wanted)
    def enter(self, mode):
        ticket=uuid.uuid4().hex
        with self.condition:
            if mode=='background' and ('background' in self.tickets.values() or self.ipc.get('state')!=0):
                raise BlockingIOError('Texas permits one native background request')
            self.tickets[ticket]=mode
            seq=self.publish(force=mode=='foreground')
        try:
            if mode=='foreground':self.wait_drained(seq)
        except BaseException:
            self.leave(ticket);raise
        return ticket
    def leave(self, ticket):
        with self.condition:
            self.tickets.pop(ticket,None);self.publish()
    def snapshot(self):
        with self.condition:
            self.publish()
            native={f:self.ipc.get(f) for f in FIELDS}
            state=native['state']
            requests=[]
            for ticket,mode in self.tickets.items():
                d=dict(id=ticket,mode=mode,phase='foreground' if mode=='foreground' else
                       {0:'queued',1:'running',2:'parking',3:'paused',4:'restoring'}.get(state,'unknown'))
                if mode=='background':d.update(output_tokens=native['tokens'],prompt_tokens=native['prompt'],
                    paused=state==3,pauses=native['pauses'],checkpoint_bytes=native['bytes'],
                    checkpoint_freed_bytes=native['freed'],checkpoint_error=bool(native['park_error']),
                    restore_waiting_for_memory=bool(native['restore_wait']))
                requests.append(d)
            return dict(native_pause=True,backend='sushi-1.3.0-texas',foreground_active=self.held(),
                        local_foreground=sum(v=='foreground' for v in self.tickets.values()),
                        max_background_context_tokens=262144,max_background_retained_bytes=24*1024**3,
                        background_system_reserve_bytes=12*1024**3,requests=requests,native=native)
    def close(self):
        self.stop.set();self.thread.join(2)
        with self.condition:
            self.tickets.clear();self.owners.clear();self.publish(force=True)
