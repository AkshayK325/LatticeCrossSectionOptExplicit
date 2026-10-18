"""Blind predictions for B9a-c (octet r 1.0, octet r 1.5, C12 r 0.5; current FHS geometry, linear, prescribed plate motion), written
BEFORE the Abaqus runs; Abaqus rows are added when the CSVs exist. Models: bare beams, zone k = 3.5, JOINT_FHS, and the rules R1-R3 of
results/rule_fit2 (full fits). The rule R3 over-stiffens the octet and C12 IMPACT solids by 10-28 % (notes/18 section 11); this is the
first prescribed-motion test of why.
    PYTHONPATH=src:tests python abaqus/b9_predict.py [cases...]   -> abaqus/results/B9_summary.md"""
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


def zone_model(sp):
    import torch
    import jointSection
    from latticeSetup import joint_factors
    orig = jointSection.element_factors_aniso

    def zone(lat, r_beam, **kw):
        re = torch.as_tensor(np.asarray(r_beam)[np.asarray(lat['beam_of_elem'])], dtype=torch.float64)
        return joint_factors(lat, re, 3.5).numpy(), None
    jointSection.element_factors_aniso = zone
    try:
        return cases.b4_ours(sp, {'x': 1})
    finally:
        jointSection.element_factors_aniso = orig


def main(names):
    import runStatus as rs                                       # repo rule: every long job reports to run_status.log
    rs.start('B9 predictions ' + ' '.join(names))
    pars = {m: json.loads((ROOT / 'results' / 'rule_fit2' / f'{m}.json').read_text())['params'] for m in ('R1', 'R2', 'R3')}
    L = ['# B9: octet r 1.0 / 1.5 and C12 r 0.5 unit cells, prescribed plate motion (current FHS geometry, linear)', '',
         'Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams.', '',
         '| case | model | end of ramp [N] | hold [N] | vs bare | error vs solid |', '|---|---|---|---|---|---|']
    for ci, case in enumerate(names):
        rs.update(case, ci, len(names))
        sp = cases.CASES[case]['spec']
        runs = [('bare beams', lambda: cases.b4_ours(sp, None)), ('zone k = 3.5', lambda: zone_model(sp)),
                ('JOINT_FHS', lambda: cases.b4_ours(sp, cases.JOINT_FHS))]
        runs += [(f'rule {m}', (lambda m=m: cases.b4_ours(sp, fr.joint_of(pars[m])))) for m in pars]
        res = {}
        for lab, fn in runs:
            o = fn()
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
    (ROOT / 'results' / 'B9_summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main(sys.argv[1:] or list(cases.B9_CASES))
