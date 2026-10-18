"""Blind predictions for the prescribed-motion specimens set up 2026-10-17: B10a-d (C4 r 0.8 / 0.6, graded C4 both ways), B11a-d (T-joints with
UNEQUAL radii), B9d (C12 long hold). Models: bare beams, zone k = 3.5, JOINT_FHS, rule R5 (results/rule_fit3/R5.json). Abaqus rows are added when
the CSVs exist (hold-window mean for the solid).
    PYTHONPATH=src:tests python abaqus/bx_predict.py [cases...]   -> abaqus/results/B10_B11_summary.md"""
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
from b9_predict import zone_model                                # noqa: E402

NAMES = ['B10a', 'B10b', 'B10c', 'B10d', 'B11a', 'B11b', 'B11c', 'B11d', 'B9d']
RUN = lambda case: cases.b78_ours if case.startswith('B11') else cases.b4_ours


def zone_any(case, sp):
    import torch
    import jointSection
    from latticeSetup import joint_factors
    orig = jointSection.element_factors_aniso

    def zone(lat, r_beam, **kw):
        re = torch.as_tensor(np.asarray(r_beam)[np.asarray(lat['beam_of_elem'])], dtype=torch.float64)
        return joint_factors(lat, re, 3.5).numpy(), None
    jointSection.element_factors_aniso = zone
    try:
        return RUN(case)(sp, {'x': 1})
    finally:
        jointSection.element_factors_aniso = orig


def main(names):
    r5 = json.loads((ROOT / 'results' / 'rule_fit3' / 'R5.json').read_text())['params']
    import runStatus as rs                                       # repo rule: every long job reports to run_status.log
    rs.start('B10/B11/B9d predictions ' + ' '.join(names))
    L = ['# B10 / B11 / B9d blind predictions (current FHS geometry, linear, prescribed plate motion)', '',
         'Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams; error vs solid when it exists.', '',
         '| case | model | end of ramp [N] | hold [N] | vs bare | error vs solid |', '|---|---|---|---|---|---|']
    for ci, case in enumerate(names):
        rs.update(case, ci, len(names))
        sp = cases.CASES[case]['spec']
        fn = RUN(case)
        runs = [('bare beams', lambda: fn(sp, None)), ('zone k = 3.5', lambda: zone_any(case, sp)),
                ('JOINT_FHS', lambda: fn(sp, cases.JOINT_FHS)), ('rule R5', lambda: fn(sp, fr.joint_of(r5)))]
        res = {}
        for lab, f in runs:
            o = f()
            res[lab] = (float(np.interp(sp['t_ramp'], o['time'], o['RF3bot'])), fr.hold_mean(o['time'], o['RF3bot'], sp))
        sol = None
        try:
            ab = compare.read_abaqus(sp['name'])
            lb = compare.labels(case)
            t, rf = compare.pick(ab, 'RF3', ('sum', lb['bot']))
            sol = (float(np.interp(sp['t_ramp'], t, rf)), fr.hold_mean(t, rf, sp))
            res['**Abaqus solid**'] = sol
        except Exception:
            pass
        bh = res['bare beams'][1]
        for lab, (r, h) in res.items():
            err = f'{100 * (h / sol[1] - 1):+.1f} %' if sol and 'solid' not in lab else ''
            L.append(f"| {case} | {lab} | {r:.4g} | {h:.4g} | {h / bh:.3f} | {err} |")
        print(chr(10).join(L[-len(res):]), flush=True)
    (ROOT / 'results' / 'B10_B11_summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main(sys.argv[1:] or NAMES)
