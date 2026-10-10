"""Live status of background runs in <repo>/run_status.log (git-ignored): one block per run, rewritten on every
update. A new run drops the runs that have ended; a run killed without notice shows as 'killed'.

    import runStatus as rs
    rs.start('all3')            # at the start of a run
    rs.update('regular-6 SF')   # current item
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
                      f"  now     : {r['now']}   (at {r['updated']})", '']
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


def update(msg):
    if 'pid' not in _me:
        return

    def fn(runs):
        r = runs.get(_me['pid'])
        if r:
            r['now'], r['updated'] = msg, _now()
    _write(fn)


def _finish():
    def fn(runs):
        r = runs.get(_me['pid'])
        if r:
            err = _me.get('error')
            r['state'], r['ended'] = ('failed' if err else 'finished'), _now()
            r['now'], r['updated'] = (err or 'done'), _now()
    _write(fn)
