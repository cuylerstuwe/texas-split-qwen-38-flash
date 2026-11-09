#!/usr/bin/env python3
"""Greedy image and video answers (and a text control) with per-token logprobs (top 5), saved for a
split vs Mac-only comparison (tests/compare_logprobs.py)."""
import json, sys, http.client
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vision_decode import png  # noqa: E402  (png() only; importing runs nothing else heavy)
from video_decode import png as frame  # noqa: E402
import base64
OUT = sys.argv[1]


import os
PORT = int(os.environ.get('VLP_PORT', os.environ.get('TEXAS_SPLIT_PORT', '8003')))
MODEL = os.environ.get('VLP_MODEL', 'texas')


def run(messages):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=900)
    c.request('POST', '/v1/chat/completions', body=json.dumps(dict(model=MODEL, messages=messages, max_tokens=120,
              temperature=0.0, top_k=1, reasoning_effort='none', logprobs=True, top_logprobs=5, stream=False)),
              headers={'Content-Type': 'application/json'})
    r = c.getresponse(); d = json.loads(r.read()); c.close()
    content = (d['choices'][0].get('logprobs') or {}).get('content') or []
    return {'mode': r.getheader('X-Texas-Split-Mode'), 'tokens': [x.get('token') for x in content],
            'logprobs': [x.get('logprob') for x in content],
            'top': [{t['token']: t['logprob'] for t in x.get('top_logprobs', [])} for x in content]}


small = 'data:image/png;base64,' + base64.b64encode(png()).decode()
large = 'data:image/png;base64,' + base64.b64encode(png(640, 640)).decode()
ask = 'Describe this image in detail: every shape, its color and its position.'
out = {
    'image_small': run([{'role': 'user', 'content': [{'type': 'text', 'text': ask}, {'type': 'image_url', 'image_url': {'url': small}}]}]),
    'image_large': run([{'role': 'user', 'content': [{'type': 'text', 'text': ask}, {'type': 'image_url', 'image_url': {'url': large}}]}]),
    'video': run([{'role': 'user', 'content': [{'type': 'text', 'text': 'Watch this short video. Describe what moves, in which direction, and what stays still.'},
                                              {'type': 'video_url', 'video_url': {'frames': ['data:image/png;base64,' + base64.b64encode(frame(i)).decode() for i in range(6)], 'fps': 2}}]}]),
    'text': run([{'role': 'user', 'content': 'Describe, in detail, a red square, a blue circle and a green triangle on a white page.'}]),
}
open(OUT, 'w').write(json.dumps(out))
print(json.dumps({k: (v['mode'], len(v['tokens'])) for k, v in out.items()}))
