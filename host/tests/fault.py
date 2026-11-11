#!/usr/bin/env python3
"""texas-split fatal-stage fallback test. Streams one chat completion and, mid-flight, takes the
5090 stage away: `term` sends the stage server SIGTERM (libtinynv puts the card down and exits),
`stop` freezes it with SIGSTOP (a hang: Sushi's stage timeout fires) and resumes it afterwards with
SIGCONT so it exits through its normal path. `prefill` does `term` during a long prompt's split
prefill. `long` does `term` DELAY_S into the reply of a long conversation (a copy task, greedy): the
Mac re-run of the context then spans several prefill chunks, with decode ticks between them, and the
reply must keep copying across the fault. Reports the stream's gaps, errors, finish reason and the
host's mode afterwards.
  fault.py term|stop|prefill|long [DELAY_S]"""
import http.client, json, os, signal, subprocess, sys, threading, time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'term'
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0


def stage_pid():
    # the server process itself (exact name), not its nv_shim_step.sh wrapper whose arguments name it
    out = subprocess.run(['pgrep', '-x', 'canon-stage-server'], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def health():
    c = http.client.HTTPConnection('127.0.0.1', int(os.environ.get('TEXAS_SPLIT_PORT', '8003')), timeout=5)
    c.request('GET', '/health')
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    return d


def main():
    pid = stage_pid()
    print(json.dumps({'stage_server_pid': pid, 'health': {k: health()[k] for k in ('status', 'mode')}}), flush=True)
    if MODE == 'prefill':
        filler = '\n'.join(f'Record {i}: sensor {i % 17} read {(i * 37) % 1000} units at minute {i}.' for i in range(5000))
        # FAULT_NONCE (optional) changes the first tokens, so a rerun misses the prefix cache and prefills cold
        nonce = os.environ.get('FAULT_NONCE')
        head = f'[run {nonce}]\n' if nonce else ''
        messages = [{'role': 'user', 'content': head + filler + '\n\nWhich sensor reading is on Record 4321? Answer with the number of units only.'}]
        max_tokens, think = 200, 'none'
    elif MODE == 'long':
        records = [f'Record {i}: sensor {i % 17} read {(i * 37) % 1000} units at minute {i}.' for i in range(1500)]
        nonce = os.environ.get('FAULT_NONCE')
        head = f'[run {nonce}]\n' if nonce else ''
        messages = [{'role': 'user', 'content': head + '\n'.join(records) + '\n\nRepeat records 0 to 199 exactly as written, one per line, and nothing else.'}]
        max_tokens, think = 2500, 'none'
    else:
        messages = [{'role': 'user', 'content': 'Write a detailed, numbered list of 40 practical tips for maintaining a home vegetable garden.'}]
        max_tokens, think = 2500, 'none'
    body = {'model': 'texas', 'stream': True, 'max_tokens': max_tokens, 'reasoning_effort': think,
            'messages': messages, 'stream_options': {'include_usage': True}}
    if MODE == 'long':
        body.update(temperature=0, presence_penalty=0)   # an exactness check: greedy, no penalty on the repeated words
    fired = {}
    times, marks = [], []

    def inject():
        if MODE == 'long':
            for _ in range(12000):                        # the reply is under way first
                if times:
                    break
                time.sleep(0.05)
        time.sleep(DELAY)
        p = stage_pid()
        if p is None:
            return
        fired['at'] = time.time()
        if MODE == 'stop':
            os.kill(p, signal.SIGSTOP)
            fired['signal'] = 'SIGSTOP'
            # resume once Sushi has given up (its socket closed), so the server exits normally
            for _ in range(240):
                time.sleep(1)
                h = health()
                if h.get('mode') == 'mac-only':
                    break
            os.kill(p, signal.SIGCONT)
            fired['resumed'] = time.time()
        else:
            os.kill(p, signal.SIGTERM)
            fired['signal'] = 'SIGTERM'
    threading.Thread(target=inject, daemon=True).start()
    c = http.client.HTTPConnection('127.0.0.1', int(os.environ.get('TEXAS_SPLIT_PORT', '8003')), timeout=1800)
    t0 = time.time()
    c.request('POST', '/v1/chat/completions', body=json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    text, errors, finish, usage = '', [], None, None
    buf = b''
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            raw, buf = buf.split(b'\n\n', 1)
            for line in raw.split(b'\n'):
                if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                    continue
                ev = json.loads(line[6:])
                if 'error' in ev:
                    errors.append(ev['error'])
                if ev.get('usage'):
                    usage = ev['usage']
                for ch in ev.get('choices', []):
                    d = ch.get('delta') or {}
                    piece = (d.get('content') or '') + (d.get('reasoning_content') or '')
                    if piece:
                        times.append(time.time())
                        text += piece
                        marks.append((times[-1], len(text)))
                    finish = ch.get('finish_reason') or finish
    c.close()
    gaps = sorted(((times[i + 1] - times[i], i) for i in range(len(times) - 1)), reverse=True)[:3]
    gap_at = gaps[0][1] if gaps else None
    out = {'status': r.status, 'errors': errors, 'finish': finish, 'usage': usage, 'chars': len(text),
           'total_s': round(time.time() - t0, 2), 'injected': fired.get('signal'),
           'injected_after_s': round(fired['at'] - t0, 2) if 'at' in fired else None,
           'largest_gaps_s': [round(g, 2) for g, _ in gaps],
           'text_around_gap': text[max(0, len(''.join([]))):0] if gap_at is None else None}
    if gap_at is not None:
        before = ''.join(text)  # whole text; show a window around the largest gap by char count
        # approximate window: characters emitted up to the gap
        n = sum(1 for _ in range(gap_at))
        out['text_around_gap'] = None
    if MODE == 'long':
        # the reply's complete lines against the records, and the ones from the fault on (the line it cut included)
        got = text.split('\n')
        if finish != 'stop' or (got and not got[-1]):
            got = got[:-1]
        at = max([n for t, n in marks if 'at' in fired and t <= fired['at']] or [0])
        first_after = text[:at].count('\n')
        out['copied_lines'] = len(got)
        out['matching_lines'] = sum(1 for i, line in enumerate(got) if i < len(records) and line == records[i])
        out['lines_before_fault'] = first_after
        out['matching_after_fault'] = sum(1 for i, line in enumerate(got) if i >= first_after and i < len(records) and line == records[i])
    print(json.dumps(out), flush=True)
    print('--- text (first 600 chars) ---')
    print(text[:600] if len(text) > 3 else repr(text))
    print('--- text (last 400 chars) ---')
    print(text[-400:])
    time.sleep(2)
    h = health()
    print(json.dumps({'health_after': {k: h[k] for k in ('status', 'mode', 'mode_reason', 'served')}}), flush=True)


if __name__ == '__main__':
    main()
