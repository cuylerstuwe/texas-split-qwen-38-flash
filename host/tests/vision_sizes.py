#!/usr/bin/env python3
"""Image-size sweep: greedy answers with top-5 logprobs for the same picture at several sizes (the
M-RoPE decode delta grows with the image's token count), saved for split vs Mac-only comparison
with compare_logprobs.py.  vision_sizes.py OUT.json"""
import base64, http.client, json, os, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vision_decode import png  # noqa: E402

OUT = sys.argv[1]


def run(messages, n=40):
    c = http.client.HTTPConnection('127.0.0.1', int(os.environ.get('TEXAS_SPLIT_PORT', '8003')), timeout=900)
    c.request('POST', '/v1/chat/completions', body=json.dumps(dict(model='texas', messages=messages, max_tokens=n, temperature=0.0,
              top_k=1, reasoning_effort='none', logprobs=True, top_logprobs=5, stream=False)), headers={'Content-Type': 'application/json'})
    r = c.getresponse(); d = json.loads(r.read()); c.close()
    content = (d['choices'][0].get('logprobs') or {}).get('content') or []
    return {'mode': r.getheader('X-Texas-Split-Mode'), 'prompt_tokens': d['usage']['prompt_tokens'], 'tokens': [x.get('token') for x in content],
            'top': [{t['token']: t['logprob'] for t in x.get('top_logprobs', [])} for x in content]}


import os, zlib, struct
SWEEP = os.environ.get('SWEEP', 'a')


def png2(w, h, seed):
    # a second family of pictures: horizontal bands and a diagonal bar, varied by seed
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            band = (y * 5 // h + seed) % 3
            r, g, b = [(240, 200, 60), (60, 160, 230), (250, 250, 250)][band]
            if abs((x - y * w / h) - seed * w / 8) < w / 24:
                r, g, b = 20, 20, 20
            row += bytes((r, g, b))
        rows.append(bytes(row))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b''.join(rows), 6)) + chunk(b'IEND', b''))


out = {}
if SWEEP == 'b':
    for size, seed in ((600, 1), (640, 2), (704, 3), (768, 1), (832, 2)):
        url = 'data:image/png;base64,' + base64.b64encode(png2(size, size, seed)).decode()
        out[f'bands_{size}_{seed}'] = run([{'role': 'user', 'content': [{'type': 'text', 'text': 'What do you see in this picture? Describe the colors, the bands and the bar.'},
                                                                         {'type': 'image_url', 'image_url': {'url': url}}]}])
for size in (() if SWEEP == 'b' else (128, 256, 384, 512, 640, 896)):
    url = 'data:image/png;base64,' + base64.b64encode(png(size, size)).decode()
    out[f'image_{size}'] = run([{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe this image in detail: every shape, its color and its position.'},
                                                             {'type': 'image_url', 'image_url': {'url': url}}]}])
open(OUT, 'w').write(json.dumps(out))
print(json.dumps({k: v['prompt_tokens'] for k, v in out.items()}))
