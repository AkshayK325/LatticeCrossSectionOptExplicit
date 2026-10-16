"""Keeps run_status.log up to date for a ladder of background fits (one process per fit, result files in a directory):
    PYTHONPATH=src:tests python abaqus/ladder_status.py <results dir> <number of fits expected> "<name>"
Counts the result files every 60 s and lists the fits that are running; exits when all are there."""
import glob
import os
import subprocess
import sys
import time

import runStatus as rs                                           # tests/runStatus.py

d, total, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rs.start(name)
while True:
    n = len(glob.glob(os.path.join(d, 'R*.json')))
    ps = subprocess.run(['ps', '-eo', 'cmd'], capture_output=True, text=True).stdout.splitlines()
    running = [l.split('fit_rule2.py')[1].strip() for l in ps if 'fit_rule2.py' in l and 'python' in l and 'ladder_status' not in l]
    rs.update(f"{n} of {total} fits done; running: {', '.join(running) or 'none'}", n, total)
    if n >= total:
        break
    time.sleep(60)
