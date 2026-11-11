#!/usr/bin/env python3
"""texas-split end-to-end smoke test against a running host (default http://127.0.0.1:$TEXAS_SPLIT_PORT, 8003).
Exercises what a coding agent uses: model listing, streaming chat with reasoning, tool calls
(streaming and not, then a tool-result turn), an image turn, a long prompt through the split
prefill, and one-generation-at-a-time (a concurrent stream waits; a queued client that leaves is
dropped; a stream cancelled mid-flight frees the engine for the next).
  smoke.py [BASE] [--only NAME,..]"""
import base64, http.client, json, os, socket, struct, sys, threading, time, zlib
from urllib.parse import urlparse

BASE = next((a for a in sys.argv[1:] if a.startswith('http')), 'http://127.0.0.1:' + os.environ.get('TEXAS_SPLIT_PORT', '8003'))
ONLY = None
for i, a in enumerate(sys.argv):
    if a == '--only':
        ONLY = set(sys.argv[i + 1].split(','))
U = urlparse(BASE)
MODEL = 'texas'
LISTED = 'Qwen3.8-Flash-Next-Sushi-4bpw'   # /v1/models lists Sushi's id, as Texas does
results = []


def conn(timeout=1800):
    return http.client.HTTPConnection(U.hostname, U.port, timeout=timeout)


def post(path, body, timeout=1800):
    c = conn(timeout)
    c.request('POST', path, body=json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data) if data else None


def stream(path, body, on_event=None, timeout=1800):
    """POST with stream=true; returns (status, events, first_event_s, total_s)."""
    c = conn(timeout)
    t0 = time.time()
    c.request('POST', path, body=json.dumps(dict(body, stream=True)), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    if r.status != 200:
        data = r.read()
        c.close()
        return r.status, [data.decode(errors='replace')], None, time.time() - t0
    events, first = [], None
    buf = b''
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b'\n\n' in buf:
            raw, buf = buf.split(b'\n\n', 1)
            for line in raw.split(b'\n'):
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    continue
                ev = json.loads(payload)
                if first is None:
                    first = time.time() - t0
                events.append(ev)
                if on_event and on_event(ev) is False:
                    c.close()
                    return 'cancelled', events, first, time.time() - t0
    c.close()
    return 200, events, first, time.time() - t0


def assemble(events):
    reasoning, content, tools, finish, usage = '', '', {}, None, None
    for ev in events:
        if ev.get('usage'):
            usage = ev['usage']
        for ch in ev.get('choices', []):
            d = ch.get('delta') or {}
            reasoning += d.get('reasoning_content') or ''
            content += d.get('content') or ''
            for tc in d.get('tool_calls') or []:
                t = tools.setdefault(tc.get('index', 0), {'id': None, 'name': '', 'arguments': ''})
                t['id'] = tc.get('id') or t['id']
                f = tc.get('function') or {}
                t['name'] += f.get('name') or ''
                t['arguments'] += f.get('arguments') or ''
            finish = ch.get('finish_reason') or finish
    return reasoning, content, [tools[k] for k in sorted(tools)], finish, usage


def check(name, ok, detail):
    results.append((name, ok, detail))
    print(json.dumps({'test': name, 'ok': bool(ok), **detail}), flush=True)


def want(name):
    return ONLY is None or name in ONLY


TOOLS = [{'type': 'function', 'function': {
    'name': 'read_file', 'description': 'Read a text file from the workspace and return its contents.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Workspace-relative path'}},
                   'required': ['path']}}},
    {'type': 'function', 'function': {
        'name': 'run_shell', 'description': 'Run a shell command in the workspace and return stdout.',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]


def png(k=2):
    # 192x192 (36 image tokens). At 96x96 (9 tokens) the thinking-mode sampler names one colour for both shapes in
    # about 5% of replies (8/150: hot or cold prompt, after text or image requests, drafts on or off), while greedy
    # is always right and seeded replies do not depend on the request before; at 192 and 448 px it was 0/30 each.
    w = h = 96 * k
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            r = g = b = 255
            if 10 * k <= x < 46 * k and 10 * k <= y < 46 * k:
                r, g, b = 220, 30, 30
            if (x - 70 * k) ** 2 + (y - 28 * k) ** 2 < (16 * k) ** 2:
                r, g, b = 30, 60, 220
            row += bytes((r, g, b))
        rows.append(bytes(row))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b''.join(rows), 9)) + chunk(b'IEND', b''))


def main():
    if want('models'):
        c = conn(10)
        c.request('GET', '/v1/models')
        r = c.getresponse()
        data = json.loads(r.read())
        c.close()
        ids = [m['id'] for m in data.get('data', [])]
        check('models', r.status == 200 and ids == [LISTED], {'status': r.status, 'ids': ids})

    if want('chat_stream'):
        st, evs, first, total = stream('/v1/chat/completions', {
            'model': MODEL, 'max_tokens': 400, 'stream_options': {'include_usage': True},
            'messages': [{'role': 'user', 'content': 'In two sentences: why is the sky blue?'}]})
        reasoning, content, tools, finish, usage = assemble(evs)
        check('chat_stream', st == 200 and (content or reasoning) and finish in ('stop', 'length'),
              {'status': st, 'first_event_s': round(first or -1, 2), 'total_s': round(total, 2), 'reasoning_chars': len(reasoning),
               'content': content[:160], 'finish': finish, 'usage': usage})

    if want('chat_nothink'):
        st, data = post('/v1/chat/completions', {'model': MODEL, 'max_tokens': 120, 'reasoning_effort': 'none',
                                                  'messages': [{'role': 'user', 'content': 'Reply with exactly: pong'}]})
        msg = (data or {}).get('choices', [{}])[0].get('message', {})
        check('chat_nothink', st == 200 and 'pong' in (msg.get('content') or '').lower(),
              {'status': st, 'content': (msg.get('content') or '')[:80], 'reasoning': (msg.get('reasoning_content') or '')[:80]})

    if want('tools_stream'):
        messages = [{'role': 'system', 'content': 'You are a coding agent. Use tools to inspect the workspace before answering.'},
                    {'role': 'user', 'content': 'What does README.md in the workspace say? Read it with the tool.'}]
        st, evs, first, total = stream('/v1/chat/completions', {'model': MODEL, 'max_tokens': 2000, 'tools': TOOLS,
                                                                'messages': messages, 'stream_options': {'include_usage': True}})
        reasoning, content, tools, finish, usage = assemble(evs)
        args_ok = False
        try:
            # parallel calls are legitimate (e.g. run_shell then read_file): every call must parse
            args_ok = bool(tools) and all(isinstance(json.loads(t['arguments']), dict) for t in tools)
        except ValueError:
            pass
        check('tools_stream', st == 200 and finish == 'tool_calls' and any(t['name'] == 'read_file' for t in tools) and args_ok,
              {'status': st, 'finish': finish, 'tools': tools, 'content': content[:120], 'total_s': round(total, 2)})
        if want('tools_result') and tools:
            call = next((t for t in tools if t['name'] == 'read_file'), tools[0])
            followup = messages + [
                {'role': 'assistant', 'content': content or None, 'reasoning_content': reasoning or None,
                 'tool_calls': [{'id': call['id'] or 'call_0', 'type': 'function',
                                 'function': {'name': call['name'], 'arguments': call['arguments']}}]},
                {'role': 'tool', 'tool_call_id': call['id'] or 'call_0', 'name': call['name'],
                 'content': '# Demo\nThis workspace holds a tiny calculator. Run `make test` to check it.'}]
            st2, evs2, first2, total2 = stream('/v1/chat/completions', {'model': MODEL, 'max_tokens': 1500, 'tools': TOOLS,
                                                                        'messages': followup})
            r2, c2, t2, f2, _ = assemble(evs2)
            check('tools_result', st2 == 200 and f2 in ('stop', 'length', 'tool_calls') and (c2 or t2),
                  {'status': st2, 'finish': f2, 'content': c2[:200], 'tools': t2, 'total_s': round(total2, 2)})

    if want('tools_nostream'):
        st, data = post('/v1/chat/completions', {'model': MODEL, 'max_tokens': 2000, 'tools': TOOLS, 'tool_choice': 'auto',
                                                  'messages': [{'role': 'user', 'content': 'List the files in the workspace using run_shell.'}]})
        ch = (data or {}).get('choices', [{}])[0]
        tc = (ch.get('message') or {}).get('tool_calls') or []
        check('tools_nostream', st == 200 and ch.get('finish_reason') == 'tool_calls' and tc and tc[0]['function']['name'] == 'run_shell',
              {'status': st, 'finish': ch.get('finish_reason'), 'tool_calls': tc})

    if want('vision'):
        img = 'data:image/png;base64,' + base64.b64encode(png()).decode()
        st, data = post('/v1/chat/completions', {'model': MODEL, 'max_tokens': 300, 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Name the two shapes and their colors, briefly.'}, {'type': 'image_url', 'image_url': {'url': img}}]}]})
        msg = (data or {}).get('choices', [{}])[0].get('message', {})
        text = ((msg.get('content') or '') + ' ' + (msg.get('reasoning_content') or '')).lower()
        check('vision', st == 200 and 'red' in text and 'blue' in text, {'status': st, 'content': (msg.get('content') or '')[:160]})

    if want('long_prompt'):
        filler = '\n'.join(f'Line {i}: the quick brown fox jumps over the lazy dog while counting {i * 7 % 101}.' for i in range(1600))
        st, evs, first, total = stream('/v1/chat/completions', {
            'model': MODEL, 'max_tokens': 300, 'reasoning_effort': 'none', 'stream_options': {'include_usage': True},
            'messages': [{'role': 'user', 'content': filler + '\n\nWhat number is counted on Line 1234? Reply with the number only.'}]})
        reasoning, content, tools, finish, usage = assemble(evs)
        check('long_prompt', st == 200 and str(1234 * 7 % 101) in content,
              {'status': st, 'prompt_tokens': (usage or {}).get('prompt_tokens'), 'ttft_s': round(first or -1, 2),
               'total_s': round(total, 2), 'content': content[:80], 'expected': 1234 * 7 % 101})

    if want('queue'):
        order, lock = [], threading.Lock()

        def worker(tag, prompt):
            st, evs, first, total = stream('/v1/chat/completions', {'model': MODEL, 'max_tokens': 200, 'reasoning_effort': 'none',
                                                                    'messages': [{'role': 'user', 'content': prompt}]})
            with lock:
                order.append((tag, st, round(first or -1, 2), round(total, 2)))
        a = threading.Thread(target=worker, args=('A', 'Count from 1 to 40, comma separated.'))
        b = threading.Thread(target=worker, args=('B', 'Say hello in French, one word.'))
        a.start()
        time.sleep(1.0)
        b.start()
        a.join()
        b.join()
        a_rec = next(o for o in order if o[0] == 'A')
        b_rec = next(o for o in order if o[0] == 'B')
        check('queue_concurrent', a_rec[1] == 200 and b_rec[1] == 200 and b_rec[2] >= a_rec[3] - 1.5,
              {'A': a_rec, 'B': b_rec, 'note': 'B first event should come after A finished'})

        # a queued client that leaves is dropped; a stream cancelled mid-flight frees the engine
        hold = threading.Thread(target=worker, args=('C', 'Count from 1 to 60, comma separated.'))
        hold.start()
        time.sleep(1.0)
        s = socket.create_connection((U.hostname, U.port))
        body = json.dumps({'model': MODEL, 'max_tokens': 50, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
        s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: '
                  + str(len(body)).encode() + b'\r\n\r\n' + body)
        time.sleep(1.0)
        s.close()                       # leaves while queued behind C
        hold.join()
        seen = {'n': 0}

        def cancel_after(ev):
            seen['n'] += 1
            return seen['n'] < 5        # drop the connection after a few chunks
        t0 = time.time()
        st, evs, first, total = stream('/v1/chat/completions', {'model': MODEL, 'max_tokens': 3000,
                                                                'messages': [{'role': 'user', 'content': 'Write a long essay about rivers.'}]},
                                       on_event=cancel_after)
        st2, evs2, first2, total2 = stream('/v1/chat/completions', {'model': MODEL, 'max_tokens': 60, 'reasoning_effort': 'none',
                                                                    'messages': [{'role': 'user', 'content': 'Reply with: ok'}]})
        _, c2, _, f2, _ = assemble(evs2)
        check('queue_cancel', st == 'cancelled' and st2 == 200 and first2 is not None and first2 < 20,
              {'cancelled_after_events': len(evs), 'next_status': st2, 'next_first_event_s': round(first2 or -1, 2),
               'next_content': c2[:40], 'elapsed_s': round(time.time() - t0, 2)})

    c = conn(10)
    c.request('GET', '/health')
    r = c.getresponse()
    h = json.loads(r.read())
    c.close()
    print(json.dumps({'health': {k: h.get(k) for k in ('status', 'mode', 'waiting', 'foreground', 'background', 'served', 'restarts')}}))
    failed = [n for n, ok, _ in results if not ok]
    print(json.dumps({'summary': {'passed': len(results) - len(failed), 'failed': failed}}))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
