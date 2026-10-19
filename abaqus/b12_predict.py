"""Blind predictions for the B12 short-strut series (two valence-3 Y nodes joined by a strut of length Ls; r and Ls varied; current FHS geometry, linear,
prescribed plate motion), written BEFORE the Abaqus runs; Abaqus rows are added when the CSVs exist (hold-window mean for the solid).
Models: bare beams, zone k = 3.5, JOINT_FHS (slicing, decay lambda = r, the recorded model), JOINT_FHS with no participation decay, rule R5 (set-3 fit) and R5
refitted on all 22 specimens (set-4 fit).
    PYTHONPATH=src:tests python abaqus/b12_predict.py [cases...]   -> abaqus/results/B12_summary.md"""
import json
import pathlib
import resource
import sys

import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (20 * 1024 ** 3,) * 2)
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases                                                     # noqa: E402
import compare                                                   # noqa: E402
import fit_rule2 as fr                                           # noqa: E402
from bx_predict import zone_any                                  # noqa: E402

NAMES = ['B12a', 'B12b', 'B12c', 'B12d', 'B12e']


def main(names):
    import runStatus as rs                                       # repo rule: every long job reports to run_status.log
    rs.start('B12 predictions ' + ' '.join(names))
    p3 = json.loads((ROOT / 'results' / 'rule_fit3' / 'R5.json').read_text())['params']
    p4 = json.loads((ROOT / 'results' / 'rule_fit4' / 'R5.json').read_text())['params']
    L = ['# B12: two valence-3 Y nodes joined by a strut of length Ls (current FHS geometry, linear, prescribed plate motion)', '',
         'Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams; error vs solid when it exists.', '',
         '| case | r | Ls [mm] | Ls / r | model | hold [N] | vs bare | error vs solid |', '|---|---|---|---|---|---|---|---|']
    for ci, case in enumerate(names):
        rs.update(case, ci, len(names))
        sp = cases.CASES[case]['spec']
        fn = cases.b78_ours
        runs = [('bare beams', lambda: fn(sp, None)), ('zone k = 3.5', lambda: zone_any(case, sp)),
                ('JOINT_FHS (lambda = r)', lambda: fn(sp, cases.JOINT_FHS)),
                ('JOINT_FHS (no decay)', lambda: fn(sp, dict(cases.JOINT_FHS, decay=None))),
                ('rule R5 (set 3)', lambda: fn(sp, fr.joint_of(p3))), ('rule R5 (set 4 refit)', lambda: fn(sp, fr.joint_of(p4)))]
        res = {}
        for lab, f in runs:
            o = f()
            res[lab] = fr.hold_mean(o['time'], o['RF3bot'], sp)
        sol = None
        try:
            ab = compare.read_abaqus(sp['name'])
            lb = compare.labels(case)
            t, rf = compare.pick(ab, 'RF3', ('sum', lb['bot']))
            sol = fr.hold_mean(t, rf, sp)
            res['**Abaqus solid**'] = sol
        except Exception:
            pass
        bh = res['bare beams']
        for lab, h in res.items():
            err = f'{100 * (h / sol - 1):+.1f} %' if sol and 'solid' not in lab else ''
            L.append(f"| {case} | {sp['r']:g} | {sp['Ls']:g} | {sp['Ls'] / sp['r']:.1f} | {lab} | {h:.4g} | {h / bh:.3f} | {err} |")
        print(chr(10).join(L[-len(res):]), flush=True)
    (ROOT / 'results' / 'B12_summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main(sys.argv[1:] or NAMES)
