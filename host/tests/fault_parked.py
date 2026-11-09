#!/usr/bin/env python3
"""texas-split: the 5090 stage is lost while a background request is parked.

A background request is held (parked to SSD on both halves), the stage server gets SIGTERM
(libtinynv puts the card down), a foreground request runs (its stage call fails: Sushi gives the
stage up and re-runs it on the Mac), the hold is released. The background's next stage call finds
the stage gone; it was resident there, so instead of continuing from Sushi's stale caches it
re-runs from its tokens on the Mac and finishes its stream. Reports both outputs and the host's
mode.  fault_parked.py [MAX_TOKENS]"""
import concurrent.futures, json, os, signal, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import priority as P  # noqa: E402

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 700


def stage_pid():
    out = subprocess.run(['pgrep', '-x', 'canon-stage-server'], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def health():
    return P.request('GET', '/health', timeout=10)[1]


def main():
    print(json.dumps({'stage_pid': stage_pid(), 'mode': health().get('mode')}), flush=True)
    b = P.body(P.PROMPT + ' Add a short example to each item.', tokens=TOKENS)
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        fut = ex.submit(P.generate, b, 'background')
        P.wait_background(lambda r, s: s['native']['state'] == 1 and r['output_tokens'] >= 64, fut)
        P.control(True)
        paused = P.background_row(P.status())
        pid = stage_pid()
        os.kill(pid, signal.SIGTERM)
        t0 = time.monotonic()
        while stage_pid() == pid and time.monotonic() - t0 < 60:
            time.sleep(0.2)
        fg = P.generate(P.body('In two sentences, what does a linker do?', tokens=96))
        mode_after_fg = health().get('mode')
        P.control(False)
        got = fut.result(timeout=1800)
    m = got['choices'][0]['message']['content']
    out = {'paused_tokens': paused['output_tokens'], 'stage_killed_pid': pid, 'foreground': fg['choices'][0]['message']['content'],
           'mode_after_foreground': mode_after_fg, 'background_finish': got['choices'][0]['finish_reason'],
           'background_usage': got['usage'], 'background_chars': len(m), 'health_after': {k: health().get(k) for k in ('status', 'mode', 'mode_reason')}}
    print(json.dumps(out), flush=True)
    print('--- background text around the pause point ---')
    print(m[:1400])
    print('...')
    print(m[-500:])


if __name__ == '__main__':
    main()
