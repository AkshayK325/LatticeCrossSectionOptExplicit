"""Run ANY command, or watch an already running process, with live progress in <repo>/run_status.log (see runStatus.py).

  run a command (its output is passed through; the log shows the last line, and progress if --total is given):
    PYTHONPATH=src:tests python tests/run_logged.py --name "B10 inputs" [--total 4 --count '^wrote'] -- python abaqus/cases.py B10a B10b
  watch a process that is already running (and optionally a log file whose matching lines count as progress):
    PYTHONPATH=src:tests python tests/run_logged.py --watch 12345 --name "impact predictions" [--file run.log --count '^\\| [FH]2' --total 27]
  a long Abaqus job on another machine cannot be seen from here; log it by hand: runStatus.start / update from a small script.
The repo rule (CLAUDE.md): every background or long job reports to run_status.log; this wrapper is the way for jobs that do not call runStatus themselves."""
import argparse
import os
import re
import subprocess
import sys
import time

import runStatus as rs


def count_lines(path, pattern):
    try:
        with open(path, errors='replace') as f:
            return sum(1 for line in f if re.search(pattern, line))
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--watch', type=int, help='pid of a running process to follow')
    ap.add_argument('--file', help='(watch) log file of that process')
    ap.add_argument('--count', help='regex: every output / file line that matches counts one item done')
    ap.add_argument('--total', type=int, help='number of items expected (for % and time left)')
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rs.start(a.name)
    done = 0
    if a.watch:
        while True:
            try:
                os.kill(a.watch, 0)
                alive = True
            except OSError:
                alive = False
            if a.file and a.count:
                done = count_lines(a.file, a.count)
            last = ''
            if a.file and os.path.exists(a.file):
                tail = [line.strip() for line in open(a.file, errors='replace').read().splitlines() if line.strip()]
                last = tail[-1][:90] if tail else ''
            rs.update(f"watching pid {a.watch} ({'running' if alive else 'ended'}) {last}", done if a.total else None, a.total)
            if not alive:
                return 0
            time.sleep(30)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    if not cmd:
        ap.error('give a command after --, or --watch PID')
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t_last = 0.0
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if a.count and re.search(a.count, line):
            done += 1
        text = line.strip()
        if text and time.time() - t_last > 5:
            rs.update(text[:100], done if a.total else None, a.total)
            t_last = time.time()
    rc = p.wait()
    if rc:
        rs._me['error'] = f'exit code {rc}'
    return rc


if __name__ == '__main__':
    sys.exit(main())
