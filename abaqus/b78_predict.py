"""Blind predictions for B7a-c (column + one oblique strut, 30 / 45 / 60 deg) and B8a/b (centre node with 4 / 6 diagonals), written
BEFORE the Abaqus runs; Abaqus rows are added when the CSVs exist. Models: bare beams, JOINT_FHS, the fitted rule (results/rule_fit/
<fit>.json, default M1 = a, p, mu fitted on B3 + B5 + B6) and the same rule without angle weight (p = 0, the fit that left the BCC cell
out: results/rule_fit/M1_CELL.json), i.e. the two competing hypotheses about the angle dependence.
    PYTHONPATH=src:tests python abaqus/b78_predict.py [fit]  -> abaqus/results/B78_summary.md"""
import json
import pathlib
import resource
import sys

import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 ** 3,) * 2)
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases                                                     # noqa: E402
import compare                                                   # noqa: E402
from b3_predict import stats                                     # noqa: E402


def load(name):
    return json.loads((ROOT / 'results' / 'rule_fit' / f'{name}.json').read_text())['params']


def main(fit='M3'):
    pa, pb = load(fit), load('M1_CELL')
    models = (('bare beams', None), ('JOINT_FHS', cases.JOINT_FHS),
              (f'rule {fit} (p = {pa["p"]:.2f}, zl = {pa["zl"]:.2f}, b = {pa["b"]:.2f})', dict(mode='rule', **dict(pa, mu=1.0))),
              (f'rule without angle weight (p = 0, a = {pb["a"]:.2f})', dict(mode='rule', **dict(pb, mu=1.0))))
    L = ['# B7 / B8: column + oblique strut, centre node valence 4 / 6 (current FHS geometry, linear, prescribed motion)', '',
         'Reaction = sum RF3 at the bottom plate, end of the ramp [N]; stiffening = / bare beams (ours).', '',
         '| case | model | end of ramp [N] | end of hold [N] | stiffening vs bare |', '|---|---|---|---|---|']
    for case in ('B7a', 'B7b', 'B7c', 'B8a', 'B8b'):
        sp = cases.CASES[case]['spec']
        res = {}
        for lab, jm in models:
            o = cases.b78_ours(sp, jm)
            res[lab] = stats(o['time'], o['U3ref'], o['RF3bot'], sp)
        try:
            ab = compare.read_abaqus(sp['name'])
        except Exception:
            ab = None
        if ab is not None:
            lb = compare.labels(case)
            rf, ud = compare.pick(ab, 'RF3', ('sum', lb['bot'])), compare.pick(ab, 'U3', ('node', lb['ref']))
            res['**Abaqus solid**'] = stats(rf[0], np.interp(rf[0], *ud), rf[1], sp)
        b = res['bare beams']['ramp_end']
        for lab, s in res.items():
            L.append(f"| {case} | {lab} | {s['ramp_end']:.4g} | {s['hold_end']:.4g} | {s['ramp_end'] / b:.3f} |")
        print(chr(10).join(L[-len(res):]), flush=True)
    (ROOT / 'results' / 'B78_summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main(*sys.argv[1:])
