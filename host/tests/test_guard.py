#!/usr/bin/env python3
"""Checks the host's memory/thermal guard (texas_split.MemoryGuard) without a running service.

    host/tests/test_guard.py [RUN_DIR ...]

Runs synthetic cases. With RUN_DIRs (each holding a memory.jsonl the host wrote), it also replays them
through the guard and through the rule it replaced (before 2026-11-10: compression growth counted until
5 minutes of idle, even after macOS had unwired the idle engine's memory), and prints what each did.
Plain asserts; no pytest.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))
import texas_split as ts  # noqa: E402  (main() runs only under __main__)

GIB = ts.GIB


def old_rule(rows):
    """The guard before 2026-11-10: the lows follow the reading only after 300 s idle.
    Returns (time, fault) of the first trip, or None."""
    compressor_low, swap_low = rows[0]['compressor_bytes'], rows[0]['swap_used_bytes']
    last_busy = rows[0]['time']
    for r in rows:
        if r['busy']:
            last_busy = r['time']
        if r['time'] - last_busy > 300:
            compressor_low, swap_low = r['compressor_bytes'], r['swap_used_bytes']
        compressor_low = min(compressor_low, r['compressor_bytes'])
        swap_low = min(swap_low, r['swap_used_bytes'])
        fault = None
        if r['available_bytes'] < 12 * GIB:
            fault = 'reclaimable memory below 12 GiB'
        elif r['compressor_bytes'] - compressor_low > 16 * GIB:
            fault = 'compression growth above 16 GiB'
        elif r['swap_used_bytes'] - swap_low > 256 * 1024 ** 2:
            fault = 'swap growth above 256 MiB'
        elif r['thermal'] in {'trapping', 'sleeping'}:
            fault = 'critical thermal pressure'
        if fault:
            return r['time'], fault
    return None


def new_rule(rows):
    """MemoryGuard over the rows: (first trip as (time, fault) or None, notes, seconds counted as unwired)."""
    guard = ts.MemoryGuard(SimpleNamespace(**rows[0]), rows[0]['time'])
    notes, unwired_s, prev = [], 0.0, None
    for r in rows:
        fault = guard.update(SimpleNamespace(**r), r['busy'], r['thermal'], r['time'])
        if guard.note:
            notes.append((r['time'], guard.note))
        if guard.unwired and prev is not None:
            unwired_s += r['time'] - prev
        prev = r['time']
        if fault:
            return (r['time'], fault), notes, unwired_s
    return None, notes, unwired_s


def replay(run_dir):
    rows = [json.loads(line) for line in (Path(run_dir) / 'memory.jsonl').open()]
    hms = lambda t: time.strftime('%H:%M:%S', time.localtime(t))  # noqa: E731
    old = old_rule(rows)
    new, notes, unwired_s = new_rule(rows)
    print(f'{Path(run_dir).name}: {len(rows)} readings')
    print(f'  old rule: {"no trip" if old is None else f"{old[1]} at {hms(old[0])}"}')
    print(f'  guard:    {"no trip" if new is None else f"{new[1]} at {hms(new[0])}"}; '
          f'engine counted as unwired {unwired_s / 60:.1f} min')
    for t, note in notes:
        print(f'  note at {hms(t)}: {note}')


# --- synthetic cases ---------------------------------------------------------------------------------------
def row(t, busy, wired, compr, avail=60.0, swap=0.05, thermal='nominal'):
    return dict(time=t, busy=busy, wired_bytes=int(wired * GIB), compressor_bytes=int(compr * GIB),
                available_bytes=int(avail * GIB), swap_used_bytes=int(swap * GIB), thermal=thermal)


def busy_then_idle(idle_until, unwire_at=None):
    """Works 0-10 s with 75 GiB wired and a 4 GiB compressor, then idles until idle_until (one reading a
    second); from unwire_at on the engine's wiring is released (6 GiB wired)."""
    rows = [row(t, True, 75, 4) for t in range(0, 11)]
    rows += [row(t, False, 6 if unwire_at is not None and t >= unwire_at else 75, 4) for t in range(11, idle_until + 1)]
    return rows


def first_trip(rows):
    trip = new_rule(rows)[0]
    return None if trip is None else trip[1]


def synthetic():
    cases = []

    # working: compressor +17 GiB trips
    rows = [row(t, True, 75, 4) for t in range(0, 11)] + [row(11, True, 75, 21)]
    cases.append(('working, +17 GiB compressed', first_trip(rows), 'compression growth above 16 GiB'))

    # idle 30 s, still wired: +17 GiB trips (the engine holds its memory; the growth is someone else's pages)
    rows = busy_then_idle(40) + [row(41, False, 75, 21)]
    cases.append(('idle 30 s, still wired, +17 GiB', first_trip(rows), 'compression growth above 16 GiB'))

    # idle and unwired: +17 GiB is not counted (the old rule tripped here); working again, +17.5 GiB above
    # the new low trips
    rows = busy_then_idle(96, unwire_at=71) + [row(97, False, 6, 21)]
    trip, notes, _ = new_rule(rows)
    cases.append(('idle and unwired, +17 GiB', None if trip is None else trip[1], None))
    cases.append(('  ... the old rule tripped', None if old_rule(rows) is None else old_rule(rows)[1],
                  'compression growth above 16 GiB'))
    cases.append(('  ... noted once', len(notes), 1))
    rows += [row(98, True, 30, 21), row(99, True, 75, 12), row(100, True, 75, 12), row(101, True, 75, 29.5)]
    cases.append(('  ... then working, +17.5 GiB above the 12 GiB low', first_trip(rows),
                  'compression growth above 16 GiB'))

    # the reclaimable floor still applies while unwired
    rows = busy_then_idle(96, unwire_at=71) + [row(97, False, 6, 5, avail=11)]
    cases.append(('unwired, 11 GiB reclaimable', first_trip(rows), 'reclaimable memory below 12 GiB'))

    # the swap check is not reset by unwiring
    rows = busy_then_idle(96, unwire_at=71) + [row(97, False, 6, 4, swap=0.05 + 300 / 1024)]
    cases.append(('unwired, swap +300 MiB', first_trip(rows), 'swap growth above 256 MiB'))

    # critical thermals trip as before
    rows = busy_then_idle(20) + [row(21, False, 75, 4, thermal='trapping')]
    cases.append(('critical thermal pressure', first_trip(rows), 'critical thermal pressure'))

    # the 300 s rule is unchanged: after 5 min idle, growth is not counted even if the engine stayed wired
    rows = busy_then_idle(400) + [row(401, False, 75, 21)]
    cases.append(('idle 390 s, still wired, +17 GiB', first_trip(rows), None))
    cases.append(('  ... old rule agrees', None if old_rule(rows) is None else old_rule(rows)[1], None))

    # a short request ending before Sushi rewires does not lower the wired reference
    rows = busy_then_idle(96, unwire_at=71) + [row(97, True, 6, 4), row(98, False, 6, 4), row(130, False, 6, 21)]
    cases.append(('short request, still unwired, +17 GiB', first_trip(rows), None))

    failed = 0
    for name, got, want in cases:
        ok = got == want
        failed += not ok
        print(f'  {"ok  " if ok else "FAIL"} {name}: {got!r}' + ('' if ok else f' (want {want!r})'))
    assert not failed, f'{failed} synthetic case(s) failed'


def main():
    for run in sys.argv[1:]:
        replay(run)
    print('synthetic:')
    synthetic()
    print('all guard tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
