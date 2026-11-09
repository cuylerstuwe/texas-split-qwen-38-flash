#!/usr/bin/env python3
"""Video turns through texas-split (Sushi's video_url: an ordered array of frame data URLs plus fps).
Greedy outputs for a video question and a follow-up, saved to OUT.json so a split run and a
--mac-only run can be compared token for token.  video_decode.py OUT.json [PORT] [MODEL]"""
import base64, json, os, struct, sys, zlib, http.client

OUT = sys.argv[1] if len(sys.argv) > 1 else None
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get('TEXAS_SPLIT_PORT', '8003'))
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'texas'


def png(frame, w=192, h=192):
    rows = []
    cx = 24 + frame * 24
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            r = g = b = 245
            if cx - 16 <= x < cx + 16 and 60 <= y < 92:
                r, g, b = 220, 30, 30
            if (x - 140) ** 2 + (y - 150) ** 2 < 18 ** 2:
                r, g, b = 30, 60, 220
            row += bytes((r, g, b))
        rows.append(bytes(row))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b''.join(rows), 9)) + chunk(b'IEND', b''))


def chat(messages, max_tokens):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=900)
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0.0, top_k=1,
                reasoning_effort='none', stream=False)
    c.request('POST', '/v1/chat/completions', body=json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    raw = r.read()
    c.close()
    if r.status != 200:
        raise SystemExit(f'HTTP {r.status}: {raw[:400]!r}')
    d = json.loads(raw)
    m = d['choices'][0]['message']
    return (m.get('content') or ''), d.get('usage'), r.getheader('X-Texas-Split-Mode')


if __name__ == '__main__':
    frames = ['data:image/png;base64,' + base64.b64encode(png(i)).decode() for i in range(6)]
    u1 = [{'type': 'text', 'text': 'Watch this short video. Describe what moves, in which direction, and what stays still. Then write a two-line poem about it.'},
          {'type': 'video_url', 'video_url': {'frames': frames, 'fps': 2}}]
    t1, us1, mode1 = chat([{'role': 'user', 'content': u1}], 300)
    t2, us2, mode2 = chat([{'role': 'user', 'content': u1}, {'role': 'assistant', 'content': t1},
                           {'role': 'user', 'content': 'How many frames did you see, and roughly how far did the moving object travel?'}], 200)
    res = {'video': t1, 'video_usage': us1, 'followup': t2, 'followup_usage': us2, 'mode': [mode1, mode2]}
    if OUT:
        open(OUT, 'w').write(json.dumps(res, indent=1))
    print(json.dumps({'mode': res['mode'], 'video_tokens': (us1 or {}).get('completion_tokens'),
                      'prompt_tokens': (us1 or {}).get('prompt_tokens'),
                      'followup_tokens': (us2 or {}).get('completion_tokens'), 'video_head': t1[:300], 'followup_head': t2[:200]}))
