"""Live status of background runs in <repo>/run_status.log (git-ignored): one block per run, rewritten on every
update. A new run drops the runs that have ended; a run killed without notice shows as 'killed'.

    import runStatus as rs
    rs.start('all3')            # at the start of a run
    rs.update('regular-6 SF')   # current item
    rs.update('node x', i, n)   # ... with progress: item i (0-based) of n, % done, elapsed, time left
    (finished / failed is written automatically at exit)
"""
import atexit
import fcntl
import json
import os
import sys
import time

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_status.log')
_DB = PATH + '.json'
_me = {}


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _write(fn):
    with open(_DB + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            runs = json.load(open(_DB))
        except (OSError, ValueError):
            runs = {}
        fn(runs)
        for pid, r in runs.items():
            if r['state'] == 'running' and not _alive(int(pid)):
                r['state'], r['ended'] = 'killed', _now()
        json.dump(runs, open(_DB, 'w'))
        lines = [f'Background runs (updated {_now()})', '']
        for pid, r in sorted(runs.items(), key=lambda kv: kv[1]['started']):
            lines += [f"[{r['state'].upper()}] {r['name']}   pid {pid}",
                      f"  command : {r['cmd']}",
                      f"  started : {r['started']}" + (f"   ended: {r['ended']}" if r.get('ended') else ''),
                      f"  now     : {r['now']}   (at {r['updated']})"]
            if r.get('total'):
                lines.append(f"  progress: {r['done']} / {r['total']} done ({100 * r['done'] / r['total']:.0f} %)"
                             f"{r.get('eta', '')}")
            lines.append('')
        with open(PATH, 'w') as f:
            f.write('\n'.join(lines))


def start(name):
    pid = str(os.getpid())
    _me.update(pid=pid)

    def fn(runs):
        for p in [p for p, r in runs.items() if r['state'] != 'running' or not _alive(int(p))]:
            del runs[p]                                             # a new run clears the ended ones
        runs[pid] = dict(name=name, cmd=' '.join(sys.argv), started=_now(), updated=_now(), now='starting',
                         state='running')
    _write(fn)
    atexit.register(_finish)
    hook = sys.excepthook

    def failed(t, v, tb):
        _me['error'] = f'{t.__name__}: {str(v)[:120]}'
        hook(t, v, tb)
    sys.excepthook = failed


def _hm(sec):
    sec = int(sec)
    return f'{sec // 3600} h {sec % 3600 // 60:02d} min' if sec >= 3600 else f'{sec // 60} min'


def update(msg, i=None, n=None):
    """Current item; with i (items already done) and n (total), also % done and the time left, estimated from this
    run's own pace (items skipped on a resume do not count)."""
    if 'pid' not in _me:
        return
    if i is not None and 'i0' not in _me:
        _me['i0'], _me['t0'] = i, time.time()

    def fn(runs):
        r = runs.get(_me['pid'])
        if r:
            r['now'], r['updated'] = msg, _now()
            if i is not None:
                r['done'], r['total'] = i, n
                el = time.time() - _me['t0']
                k = i - _me['i0']
                r['eta'] = (f", elapsed {_hm(el)}, about {_hm(el / k * (n - i))} left" if k > 0 else
                            f", elapsed {_hm(el)}, time left after the first item")
    _write(fn)


def _finish():
    def fn(runs):
        r = runs.get(_me['pid'])
        if r:
            err = _me.get('error')
            r['state'], r['ended'] = ('failed' if err else 'finished'), _now()
            r['now'], r['updated'] = (err or 'done'), _now()
            if r.get('total') and not err:                          # skipped tail items never report progress
                r['done'] = r['total']
            if r.get('total') and 't0' in _me:
                r['eta'] = f", took {_hm(time.time() - _me['t0'])}"
    _write(fn)
