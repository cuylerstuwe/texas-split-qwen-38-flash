#!/usr/bin/env python3
"""texas-split native priority qualification (127.0.0.1:$TEXAS_SPLIT_PORT, default 8003).

A background request is held mid-flight (in decode, or in its prefill), parked to SSD (Sushi's
request-N.bin and the 5090 stage's stage-request-N.bin), a foreground request runs, and the
background resumes. Its final output must equal an uninterrupted reference exactly (greedy and
seeded sampling). Also: the hold persists across queued foregrounds and is released when the
last one completes or its client leaves; Texas's admission rules (one background, text-only
background). Writes tests/evidence/priority-latest.json.

  priority.py [CASE ...]     cases: decode sampled prefill queue disconnect stream rules (default all)
"""
import concurrent.futures, hashlib, http.client, json, os, pathlib, socket, sys, time, uuid

ROOT = pathlib.Path(__file__).parent
PORT = int(os.environ.get('TEXAS_SPLIT_PORT', '8003'))
KEY = pathlib.Path(os.path.expanduser(os.environ.get('TEXAS_SPLIT_CONTROL_KEY', '~/.config/texas-split/control-key'))).read_text().strip()
OWNER = 'texas-split-qualification-' + uuid.uuid4().hex[:12]
epoch = 0
results = []
PROMPT = ('Write a long, detailed, numbered explanation of how a compiler parses and optimizes a program. '
          'Continue until you reach at least forty items.')


def request(method, path, body=None, headers=None, timeout=1800):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    h = {'Content-Type': 'application/json', **(headers or {})}
    c.request(method, path, body=None if body is None else json.dumps(body), headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        data = json.loads(raw)
    except ValueError:
        data = raw.decode(errors='replace')
    return r.status, data, dict(r.getheaders())


def status():
    return request('GET', '/v1/scheduler', headers={'Authorization': 'Bearer ' + KEY})[1]


def control(held):
    global epoch
    epoch += 1
    t = time.monotonic()
    code, data, _ = request('POST', '/v1/scheduler', {'owner': OWNER, 'epoch': epoch, 'foreground': held, 'lease_seconds': 120},
                            {'Authorization': 'Bearer ' + KEY}, timeout=60)
    assert code == 200 and data.get('foreground') == held and data.get('epoch') == epoch, (code, data)
    return time.monotonic() - t


def body(prompt, temp=0, tokens=512, think=False, **extra):
    b = {'model': 'texas', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': tokens, 'temperature': temp,
         'seed': 431, 'reasoning_effort': 'high' if think else 'none', 'stream': False}
    if temp == 0:
        b['top_k'] = 1
    b.update(extra)
    return b


def generate(b, mode='foreground'):
    code, data, headers = request('POST', '/v1/chat/completions', b, {'X-LocalAI-Mode': mode})
    assert code == 200, (code, data)
    return data


def digest(d):
    m = d['choices'][0]['message']
    return hashlib.sha256(json.dumps({k: m.get(k) for k in ('content', 'reasoning_content', 'tool_calls')}, sort_keys=True).encode()).hexdigest()


def background_row(s):
    return next((r for r in s.get('requests', []) if r['mode'] == 'background'), None)


def wait_background(pred, fut, limit=600):
    deadline = time.monotonic() + limit
    while True:
        s = status()
        row = background_row(s)
        if row and pred(row, s):
            return s
        if fut.done():
            raise RuntimeError(('background ended early', fut.result()))
        if time.monotonic() > deadline:
            raise RuntimeError(('no background progress', s))
        time.sleep(0.02)


def record(name, row):
    row = {'case': name, **row}
    results.append(row)
    (ROOT / 'evidence').mkdir(exist_ok=True)
    (ROOT / 'evidence' / 'priority-latest.json').write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k not in ('reference', 'resumed', 'foreground', 'before', 'after')}), flush=True)


def exact_case(name, b, at_tokens=48, in_prefill=False, lease=True, foregrounds=1):
    """Reference (uninterrupted), then the same request in background held mid-flight."""
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        reference = generate(b) if not in_prefill else None
        fut = ex.submit(generate, b, 'background')
        if in_prefill:
            before_hold = wait_background(lambda r, s: s['native']['phase'] == 1 and s['native']['state'] == 1, fut)
            time.sleep(1.0)   # well into the split prefill (several chunks)
        else:
            before_hold = wait_background(lambda r, s: s['native']['state'] == 1 and r['output_tokens'] >= at_tokens, fut)
        switch = control(True) if lease else None
        before = status()
        fg_futs = [ex.submit(generate, body('In one short sentence, what is a compiler?', tokens=64)) for _ in range(foregrounds)]
        if not lease:
            # the foreground itself holds: wait for the park
            wait_background(lambda r, s: r['paused'], fut, 120)
            before = status()
        time.sleep(0.5)
        mid = status()
        fgs = [f.result(timeout=900) for f in fg_futs]
        after = status()
        b_row, m_row, a_row = background_row(before), background_row(mid), background_row(after)
        if lease:
            assert b_row['paused'] and a_row['paused'] and a_row['output_tokens'] == b_row['output_tokens'], (before, after)
            control(False)
        got = fut.result(timeout=1800)
        if in_prefill:
            reference = generate(b)   # warm: the prefix the parked run cached (cold == warm checked separately)
        exact = digest(reference) == digest(got)
        record(name, {'exact': exact, 'switch_s': switch, 'in_prefill': in_prefill, 'lease': lease,
                      'paused_tokens': b_row['output_tokens'], 'paused_phase': before['native']['phase'],
                      'checkpoint_bytes': b_row.get('checkpoint_bytes'), 'pauses': a_row.get('pauses') if a_row else None,
                      'frozen': (m_row or {}).get('output_tokens') == (b_row or {}).get('output_tokens'),
                      'usage': got.get('usage'), 'reference_usage': reference.get('usage'),
                      'reference': reference['choices'][0]['message'], 'resumed': got['choices'][0]['message'],
                      'foreground': [f['choices'][0]['message'].get('content') for f in fgs], 'before': before, 'after': after})
        assert exact, f'{name}: output changed on pause/resume'


def case_decode():
    b = body(PROMPT)
    generate(b)   # warm the prefix so the reference and the parked run take the same path
    exact_case('decode-greedy-lease', b, at_tokens=64)


def case_sampled():
    b = body(PROMPT + ' Use plain language.', temp=1.0, top_p=0.95, top_k=20)
    generate(b)
    exact_case('decode-sampled-foreground-hold', b, at_tokens=80, lease=False)


def case_prefill():
    doc = '\n'.join(f'Record {i}: sensor {i % 17} read {(i * 37) % 1000} units at minute {i}.' for i in range(2600))
    # control: a fresh long prompt cold, then warm, uninterrupted (the parked run below is cold, its reference warm)
    ctl = body(doc.replace('sensor', 'probe') + '\n\nSummarize the readings of probe 3 in three sentences.', tokens=160)
    cold, warm = generate(ctl), generate(ctl)
    record('prefill-cold-equals-warm', {'exact': digest(cold) == digest(warm), 'usage_cold': cold.get('usage'), 'usage_warm': warm.get('usage')})
    assert digest(cold) == digest(warm), 'cold and warm prefill paths differ: the prefill case cannot be judged exactly'
    b = body(doc + '\n\nSummarize the readings of sensor 5 in three sentences.', tokens=160)
    exact_case('prefill-greedy-lease', b, in_prefill=True)


def case_queue():
    """Two queued foregrounds: the background stays parked until the last completes."""
    b = body(PROMPT + ' Number every item.')
    generate(b)
    exact_case('decode-greedy-two-foregrounds', b, at_tokens=48, lease=False, foregrounds=2)


def case_disconnect():
    """A foreground whose client leaves (while the background is parked) releases the hold."""
    b = body(PROMPT + ' Be thorough.', tokens=384)
    generate(b)
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        reference = generate(b)
        fut = ex.submit(generate, b, 'background')
        wait_background(lambda r, s: s['native']['state'] == 1 and r['output_tokens'] >= 32, fut)
        s = socket.create_connection(('127.0.0.1', PORT))
        fg = json.dumps(body('Write a 2000-word essay about rivers.', tokens=2000, stream=True)).encode()
        s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n'
                  b'X-LocalAI-Mode: foreground\r\nContent-Length: ' + str(len(fg)).encode() + b'\r\n\r\n' + fg)
        paused = wait_background(lambda r, st: r['paused'], fut, 120)
        time.sleep(2.0)
        frozen = background_row(status())['output_tokens']
        s.close()   # the client leaves mid-stream
        t0 = time.monotonic()
        resumed_s = None
        while not fut.done() and time.monotonic() - t0 < 120:
            row = background_row(status())
            if row is None or row['output_tokens'] > frozen:
                resumed_s = round(time.monotonic() - t0, 2)
                break
            time.sleep(0.02)
        got = fut.result(timeout=900)
        resumed_s = resumed_s if resumed_s is not None else round(time.monotonic() - t0, 2)
        exact = digest(reference) == digest(got)
        record('foreground-disconnect-releases', {'exact': exact and resumed_s < 30, 'resume_after_disconnect_s': resumed_s,
                                                  'paused_tokens': background_row(paused)['output_tokens'], 'frozen_tokens': frozen})
        assert exact and resumed_s < 30


def case_stream():
    """SSE background with tools, held mid-stream: concatenated content/reasoning/tool arguments exact."""
    tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city',
                                              'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    b = body('First think about which cities in Europe are interesting in winter, list five with one reason each, '
             'then call get_weather for Oslo.', tokens=600, think=True, tools=tools, stream=True)

    def stream(mode):
        c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1800)
        c.request('POST', '/v1/chat/completions', body=json.dumps(b), headers={'Content-Type': 'application/json', 'X-LocalAI-Mode': mode})
        r = c.getresponse()
        assert r.status == 200, r.status
        content = reasoning = ''
        args = {}
        names = {}
        buf = b''
        while chunk := r.read1(65536):
            buf += chunk
            while b'\n\n' in buf:
                ev, buf = buf.split(b'\n\n', 1)
                for line in ev.split(b'\n'):
                    if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                        d = json.loads(line[6:])
                        for ch in d.get('choices', []):
                            delta = ch.get('delta') or {}
                            content += delta.get('content') or ''
                            reasoning += delta.get('reasoning_content') or ''
                            for tc in delta.get('tool_calls') or []:
                                i = tc.get('index', 0)
                                f = tc.get('function') or {}
                                names[i] = names.get(i, '') + (f.get('name') or '')
                                args[i] = args.get(i, '') + (f.get('arguments') or '')
        c.close()
        return {'content': content, 'reasoning': reasoning, 'tools': [[names.get(i), args.get(i)] for i in sorted(names)]}

    stream('foreground')   # warm
    reference = stream('foreground')
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        fut = ex.submit(stream, 'background')
        wait_background(lambda r, s: s['native']['state'] == 1 and r['output_tokens'] >= 96, fut)
        control(True)
        fg = generate(body('Name one planet.', tokens=32))
        control(False)
        got = fut.result(timeout=1800)
    exact = got == reference
    record('stream-tools-thinking', {'exact': exact, 'tool_calls': got['tools'], 'reasoning_chars': len(got['reasoning']),
                                     'content_chars': len(got['content'])})
    assert exact


def case_rules():
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        b = body(PROMPT, tokens=256)
        fut = ex.submit(generate, b, 'background')
        wait_background(lambda r, s: s['native']['state'] == 1 and r['output_tokens'] >= 8, fut)
        second = request('POST', '/v1/chat/completions', b, {'X-LocalAI-Mode': 'background'})
        fut.result(timeout=900)
    img = {'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}
    media = request('POST', '/v1/chat/completions', {'model': 'texas', 'messages': [img], 'max_tokens': 8}, {'X-LocalAI-Mode': 'background'})
    bad = request('POST', '/v1/chat/completions', b, {'X-LocalAI-Mode': 'urgent'})
    unknown = request('POST', '/v1/chat/completions', dict(b, model='gpt-4'))
    unauth = request('GET', '/v1/scheduler')
    models = request('GET', '/v1/models')
    ok = request('POST', '/v1/chat/completions', body('Say yes.', tokens=4))
    checks = {'second_background_429': second[0] == 429 and second[1]['error']['code'] == 'background_capacity_exceeded',
              'background_media_400': media[0] == 400 and media[1]['error']['code'] == 'background_media_unsupported',
              'invalid_priority_400': bad[0] == 400 and bad[1]['error']['code'] == 'invalid_priority',
              'unknown_model_404': unknown[0] == 404 and unknown[1]['error']['code'] == 'model_not_found',
              'scheduler_unauthorized_401': unauth[0] == 401,
              'models_id': models[0] == 200 and models[1]['data'][0]['id'] == 'Qwen3.8-Flash-Next-Sushi-4bpw',
              'backend_header': ok[0] == 200 and ok[2].get('X-Texas-Backend', '').startswith('texas-split')}
    record('admission-rules', {'exact': all(checks.values()), 'checks': checks})
    assert all(checks.values()), checks


def case_large():
    """A ~192K-token background parked mid-decode (the stage's KV is large here)."""
    doc = '\n'.join(f'Entry {i}: station {i % 23} logged {(i * 53) % 997} units, status {"ok" if i % 7 else "check"}.' for i in range(9000))
    b = body(doc + '\n\nList the first ten entries whose status is check, one per line, with their units.', tokens=256)
    warm = generate(b)
    exact_case('large-context-decode-greedy-lease', b, at_tokens=48)
    results[-1]['prompt_tokens'] = warm['usage']['prompt_tokens']


CASES = {'large': case_large, 'decode': case_decode, 'sampled': case_sampled, 'prefill': case_prefill, 'queue': case_queue,
         'disconnect': case_disconnect, 'stream': case_stream, 'rules': case_rules}

if __name__ == '__main__':
    names = sys.argv[1:] or [n for n in CASES if n != 'large']
    failed = []
    for n in names:
        try:
            CASES[n]()
        except Exception as e:
            failed.append(n)
            print(json.dumps({'case': n, 'error': repr(e)[:1500]}), flush=True)
    print(json.dumps({'failed': failed, 'status': status()['native']}), flush=True)
    sys.exit(1 if failed else 0)
