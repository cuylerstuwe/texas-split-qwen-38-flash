#!/usr/bin/env python3
"""The host's eGPU preflight (texas_split.egpu_check / Host.egpu_preflight), with macuda's scripts replaced by fakes.

    python3 host/tests/test_egpu_preflight.py

macuda's preflight refuses the card after every re-enumeration (a reboot, a sleep/wake or a replug) until someone
runs tools/dart-stale.sh --clear. The host clears a refusal whose reasons are only re-enumeration ones and checks
again; any other refusal must leave the card alone. Plain asserts; no pytest, no GPU, no service. The verdict lines
are ones preflight.sh prints.
"""
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))
import texas_split as ts  # noqa: E402  (main() runs only under __main__)

OK = 'VERDICT: OK to proceed (one GPU step at a time)'
REBOOT = 'VERDICT: ABORT - do not touch the GPU [nub re-enumerated since the DART snapshot; a new dext instance since the record]'
DEXT = 'VERDICT: ABORT - do not touch the GPU [a new dext instance since the record]'
FAULT = 'VERDICT: ABORT - do not touch the GPU [new DART error data]'
WEDGED = 'VERDICT: ABORT - do not touch the GPU [a new dext instance since the record; dext or server wedged]'
NOCARD = 'VERDICT: ABORT - do not touch the GPU [no NVIDIA GPU on the bus]'
DEXT_LINE = ('   dext instances: 1, NEWEST pid 612 started 2026-01-02 09:41:22 is NEWER than the record '
             '(2026-01-01 13:56:48):')
CLEARS = None
DEXT_START = time.mktime(time.strptime('2026-01-02 09:41:22', '%Y-%m-%d %H:%M:%S'))


class FakeRun:
    """Stands in for subprocess.run: preflight.sh answers from `verdicts` in turn, sysctl with `boot`."""

    def __init__(self, verdicts, boot=DEXT_START - 82, extra=DEXT_LINE):
        self.verdicts, self.boot, self.extra, self.calls = list(verdicts), boot, extra, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[:1] == ['sysctl']:
            return SimpleNamespace(stdout=f'{{ sec = {int(self.boot)}, usec = 0 }} Fri Jan  2 09:40:00 2026\n', returncode=0)
        if cmd[-1].endswith('preflight.sh'):
            assert kw.get('env', {}).get('EGPU_ROOT') == 'x', 'the stage environment reaches the preflight'
            return SimpleNamespace(stdout=f'== preflight ==\n{self.extra}\n{self.verdicts.pop(0)}\n', returncode=0)
        if cmd[-2:] == [str(ts.MACUDA / 'tools/dart-stale.sh'), '--clear']:
            return SimpleNamespace(stdout='cleared: ...\n', returncode=0)
        raise AssertionError(f'unexpected command {cmd}')

    def cleared(self):
        return sum(1 for c in self.calls if c[-1] == '--clear')

    def preflights(self):
        return sum(1 for c in self.calls if c[-1].endswith('preflight.sh'))


def run_case(verdicts, **kw):
    fake = FakeRun(verdicts, **kw)
    ok, reasons, rec = ts.egpu_check({'EGPU_ROOT': 'x'}, clears=CLEARS, run=fake)
    return fake, ok, reasons, rec


def main():
    global CLEARS
    tmp = Path(tempfile.mkdtemp())
    CLEARS = tmp / 'egpu-auto-clears.jsonl'
    n = 0

    # the parser
    assert ts.egpu_verdict(OK + '\n') == (True, [])
    assert ts.egpu_verdict(REBOOT) == (False, ['nub re-enumerated since the DART snapshot',
                                               'a new dext instance since the record'])
    assert ts.egpu_verdict('preflight.sh: line 3: syntax error\n') == (False, ['preflight printed no verdict'])
    assert ts.egpu_verdict('VERDICT: ABORT - garbled')[1][0].startswith('unparsed verdict')
    n += 4

    # a clean card: one preflight, nothing cleared
    fake, ok, reasons, rec = run_case([OK])
    assert (ok, reasons, rec) == (True, [], None) and fake.cleared() == 0 and fake.preflights() == 1
    n += 1

    # the state after a reboot: both re-enumeration reasons -> cleared, checked again, OK, logged
    fake, ok, reasons, rec = run_case([REBOOT, OK])
    assert ok and reasons == [] and fake.cleared() == 1 and fake.preflights() == 2
    assert rec['reasons'] == ['nub re-enumerated since the DART snapshot', 'a new dext instance since the record']
    assert rec['cause'] == 'reboot' and rec['dext_pid'] == 612 and rec['after'] == 'OK' and rec['clear_rc'] == 0
    logged = [json.loads(line) for line in CLEARS.read_text().splitlines()]
    assert len(logged) == 1 and logged[0]['reasons'] == rec['reasons']
    n += 1

    # only the dext record moved, hours after boot (a replug or a sleep/wake): cleared too, cause says so
    fake, ok, reasons, rec = run_case([DEXT, OK], boot=DEXT_START - 5 * 3600)
    assert ok and fake.cleared() == 1 and rec['cause'].startswith('re-enumerated after boot')
    n += 1

    # anything else is not cleared and the card is not touched again
    for verdict, why in ((FAULT, ['new DART error data']),
                         (WEDGED, ['a new dext instance since the record', 'dext or server wedged']),
                         (NOCARD, ['no NVIDIA GPU on the bus'])):
        fake, ok, reasons, rec = run_case([verdict])
        assert (ok, reasons, rec) == (False, why, None) and fake.cleared() == 0 and fake.preflights() == 1, verdict
        n += 1

    # a preflight that prints no verdict is a refusal, not a re-enumeration
    fake = FakeRun([])
    fake.verdicts = ['']
    ok, reasons, rec = ts.egpu_check({'EGPU_ROOT': 'x'}, run=fake)
    assert not ok and rec is None and fake.cleared() == 0
    n += 1

    # cleared, but the second look finds something else: refused with the new reasons, the clear still logged
    fake, ok, reasons, rec = run_case([DEXT, NOCARD])
    assert not ok and reasons == ['no NVIDIA GPU on the bus'] and rec['after'] == reasons and fake.cleared() == 1
    n += 1

    # TEXAS_SPLIT_EGPU_AUTO_CLEAR=0: the refusal stands, for a human
    ts.EGPU_AUTO_CLEAR = False
    fake, ok, reasons, rec = run_case([REBOOT])
    assert not ok and rec is None and fake.cleared() == 0
    ts.EGPU_AUTO_CLEAR = True
    n += 1

    # Host.egpu_preflight: the log line, the stage log line, and the reasons kept for the launch error
    real = ts.egpu_check
    try:
        host = SimpleNamespace(egpu_refused=['stale'], run_dir=tmp / 'run')
        slog = io.BytesIO()
        ts.egpu_check = lambda env, clears=None: (True, [], {'reasons': ['a new dext instance since the record'], 'cause': 'reboot'})
        assert ts.Host.egpu_preflight(host, {}, slog) is True and host.egpu_refused is None
        assert b'record cleared automatically, preflight OK' in slog.getvalue()
        ts.egpu_check = lambda env, clears=None: (False, ['new DART error data'], None)
        slog = io.BytesIO()
        assert ts.Host.egpu_preflight(host, {}, slog) is False and host.egpu_refused == ['new DART error data']
        assert slog.getvalue() == b''
        def broken(env, clears=None):
            raise OSError('sh: not found')
        ts.egpu_check = broken
        assert ts.Host.egpu_preflight(host, {}, io.BytesIO()) is False and host.egpu_refused is None
        n += 3
    finally:
        ts.egpu_check = real
    print(f'test_egpu_preflight: {n} checks passed')


if __name__ == '__main__':
    main()
