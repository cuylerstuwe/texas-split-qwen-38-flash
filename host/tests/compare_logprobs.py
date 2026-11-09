#!/usr/bin/env python3
"""Compare two vision_logprobs.py outputs (split vs Mac-only): greedy agreement and the mean top-5 KL
(Mac-only as reference, renormalized over the union of both top-5 sets, floor e^-20) over the
common greedy prefix.  compare_logprobs.py SPLIT.json MAC.json"""
import json, math, sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
out = {}
for k in a:
    x, y = a[k], b[k]
    n = next((i for i in range(min(len(x['tokens']), len(y['tokens']))) if x['tokens'][i] != y['tokens'][i]),
             min(len(x['tokens']), len(y['tokens'])))
    kls = []
    for i in range(n):
        p, q = y['top'][i], x['top'][i]
        keys = set(p) | set(q)
        P = {t: math.exp(p.get(t, -20.0)) for t in keys}
        Q = {t: math.exp(q.get(t, -20.0)) for t in keys}
        zp, zq = sum(P.values()), sum(Q.values())
        kls.append(sum(P[t] / zp * math.log((P[t] / zp) / (Q[t] / zq)) for t in keys))
    out[k] = {'same_greedy_tokens': n, 'of': len(y['tokens']), 'mean_top5_kl': round(sum(kls) / max(1, len(kls)), 6),
              'modes': [x.get('mode'), y.get('mode')]}
print(json.dumps(out, indent=1))
