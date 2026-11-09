#!/usr/bin/env python3
"""texas-split fatal-stage fallback test. Streams one chat completion and, mid-flight, takes the
5090 stage away: `term` sends the stage server SIGTERM (libtinynv puts the card down and exits),
`stop` freezes it with SIGSTOP (a hang: Sushi's stage timeout fires) and resumes it afterwards with
SIGCONT so it exits through its normal path. `prefill` does `term` during a long prompt's split
prefill. Reports the stream's gaps, errors, finish reason and the host's mode afterwards.
  fault.py term|stop|prefill [DELAY_S]"""
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
    else:
        messages = [{'role': 'user', 'content': 'Write a detailed, numbered list of 40 practical tips for maintaining a home vegetable garden.'}]
        max_tokens, think = 2500, 'none'
    body = {'model': 'texas', 'stream': True, 'max_tokens': max_tokens, 'reasoning_effort': think,
            'messages': messages, 'stream_options': {'include_usage': True}}
    fired = {}

    def inject():
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
    times, text, errors, finish, usage = [], '', [], None, None
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
