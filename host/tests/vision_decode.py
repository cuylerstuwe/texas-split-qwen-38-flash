#!/usr/bin/env python3
"""Image turns through texas-split: greedy (temperature 0) outputs for an image question and an
image follow-up, saved to OUT.json so a split run and a --mac-only run can be compared. With the
M-RoPE rope delta the split decodes image turns on the 5090; a wrong delta shows up at once as
divergent or incoherent text.  vision_decode.py OUT.json"""
import base64, json, struct, sys, zlib, http.client, os

OUT = sys.argv[1] if len(sys.argv) > 1 else None


def png(w=128, h=128):
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            r = g = b = 255
            if 16 <= x < 56 and 16 <= y < 56:
                r, g, b = 220, 30, 30
            if (x - 92) ** 2 + (y - 36) ** 2 < 20 ** 2:
                r, g, b = 30, 60, 220
            if 70 <= y < 112 and abs(x - 64) < (y - 70):
                r, g, b = 30, 170, 60
            row += bytes((r, g, b))
        rows.append(bytes(row))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b''.join(rows), 9)) + chunk(b'IEND', b''))


def chat(messages, max_tokens):
    c = http.client.HTTPConnection('127.0.0.1', int(os.environ.get('TEXAS_SPLIT_PORT', '8003')), timeout=900)
    body = dict(model='texas', messages=messages, max_tokens=max_tokens, temperature=0.0, top_k=1,
                reasoning_effort='none', stream=False)
    c.request('POST', '/v1/chat/completions', body=json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    m = d['choices'][0]['message']
    return (m.get('content') or ''), d.get('usage'), r.getheader('X-Texas-Split-Mode')


if __name__ == '__main__':
    img = 'data:image/png;base64,' + base64.b64encode(png()).decode()
    u1 = [{'type': 'text', 'text': 'Describe this image in detail: every shape, its color and its position. Then write a short poem about it.'},
          {'type': 'image_url', 'image_url': {'url': img}}]
    t1, us1, mode1 = chat([{'role': 'user', 'content': u1}], 400)
    t2, us2, mode2 = chat([{'role': 'user', 'content': u1}, {'role': 'assistant', 'content': t1},
                           {'role': 'user', 'content': 'Now list the shapes from largest to smallest and explain why.'}], 300)
    res = {'image': t1, 'image_usage': us1, 'followup': t2, 'followup_usage': us2, 'mode': [mode1, mode2]}
    open(OUT, 'w').write(json.dumps(res, indent=1))
    print(json.dumps({'mode': res['mode'], 'image_tokens': (us1 or {}).get('completion_tokens'),
                      'followup_tokens': (us2 or {}).get('completion_tokens'), 'image_head': t1[:200]}))
