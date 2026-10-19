"""Blind predictions for B13 (C4 cell, long and short struts with decoupled radii) and the equal-radius members B10b (0.6, 0.6) / B10a (0.8, 0.8); Abaqus rows when the CSVs
exist (hold-window mean for the solid). Models: bare beams, zone k = 3.5, JOINT_FHS (decay lambda = r), JOINT_FHS without decay, rule R5 (set 3), rule R5 refitted on 22 specimens (set 4).
    PYTHONPATH=src:tests python abaqus/b13_predict.py [cases...]   -> abaqus/results/B13_summary.md"""
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

NAMES = ['B10b', 'B13a', 'B13b', 'B10a', 'B13c', 'B13d']


def main(names):
    import runStatus as rs                                       # repo rule: every long job reports to run_status.log
    rs.start('B13 predictions ' + ' '.join(names))
    p3 = json.loads((ROOT / 'results' / 'rule_fit3' / 'R5.json').read_text())['params']
    p4 = json.loads((ROOT / 'results' / 'rule_fit4' / 'R5.json').read_text())['params']
    L = ['# B13: C4 unit cell with decoupled radii of the long (6.35 mm) and short (2.63 mm) struts (current FHS geometry, linear, prescribed plate motion)', '',
         'Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams; error vs solid when it exists.', '',
         '| case | r long | r short | model | hold [N] | vs bare | error vs solid |', '|---|---|---|---|---|---|---|']
    for ci, case in enumerate(names):
        rs.update(case, ci, len(names))
        sp = cases.CASES[case]['spec']
        rl, rsh = sp.get('r_long', sp['r']), sp.get('r_short', sp['r'])
        runs = [('bare beams', lambda: cases.b4_ours(sp, None)), ('zone k = 3.5', lambda: zone_any(case, sp)),
                ('JOINT_FHS (lambda = r)', lambda: cases.b4_ours(sp, cases.JOINT_FHS)),
                ('JOINT_FHS (no decay)', lambda: cases.b4_ours(sp, dict(cases.JOINT_FHS, decay=None))),
                ('rule R5 (set 3)', lambda: cases.b4_ours(sp, fr.joint_of(p3))), ('rule R5 (set 4 refit)', lambda: cases.b4_ours(sp, fr.joint_of(p4))),
                ('rule R5 + short-strut law (gain 1.5, l0 4.6)', lambda: cases.b4_ours(sp, fr.joint_of(dict(p3, short_gain=1.5, short_l0=4.6))))]
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
            L.append(f"| {case} | {rl:g} | {rsh:g} | {lab} | {h:.4g} | {h / bh:.3f} | {err} |")
        print(chr(10).join(L[-len(res):]), flush=True)
    (ROOT / 'results' / 'B13_summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main(sys.argv[1:] or NAMES)
