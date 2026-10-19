"""Effective stiffness of the SHORT (2.63 mm) C4 struts: for each C4 specimen with a solid (B10a-d, B13a-d) find the multiplier m on the factors of the short-strut elements
(on top of rule R5, set-3 fit) that makes the beam model's hold reaction equal the solid's (bisection on log m). m = 1: the rule already has the right short-strut stiffness;
the B10a solid needs m ~ 3 (notes/18 section 17). Reports m against L / r_short, r_short and r_long, i.e. the law a short-strut correction would have to follow.
    PYTHONPATH=src:tests python abaqus/short_strut_calibration.py [cases...]   -> abaqus/results/short_strut_calibration.md"""
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

NAMES = ['B10b', 'B10a', 'B10c', 'B10d', 'B13a', 'B13b', 'B13c', 'B13d']


def solid(case):
    sp = cases.CASES[case]['spec']
    ab = compare.read_abaqus(sp['name'])
    lb = compare.labels(case)
    t, rf = compare.pick(ab, 'RF3', ('sum', lb['bot']))
    return fr.hold_mean(t, rf, sp)


def main(names):
    import torch
    import jointRule
    import runStatus as rs
    rs.start('short-strut calibration ' + ' '.join(names))
    p3 = json.loads((ROOT / 'results' / 'rule_fit3' / 'R5.json').read_text())['params']
    orig = jointRule.rule_factors
    L = ['# Effective multiplier m on the short-strut elements (on top of rule R5) that reproduces the C4 solids', '',
         'm = 1: R5 already right; per case: stiffening of the solid and of R5 (vs bare beams), the m that matches, and the geometry.', '',
         '| case | r long | r short | L_short / r_short | solid hold [N] | R5 hold [N] | R5 error | m (short struts) | m (long-strut end elements) |', '|---|---|---|---|---|---|---|---|---|']
    for ci, case in enumerate(names):
        try:
            target = solid(case)
        except Exception:
            continue
        rs.update(case, ci, len(names))
        sp = cases.CASES[case]['spec']
        lat = cases.b4_lattice(sp)
        boe = np.asarray(lat['beam_of_elem'])
        Lb = np.linalg.norm(lat['X'][lat['conn'][:, 1]] - lat['X'][lat['conn'][:, 0]], axis=1)
        short = (np.abs(Lb - Lb.min()) < 1e-6)[boe]
        rads = cases._b3_radii(sp, lat) if hasattr(cases, '_b3_radii') else None
        r_s = float(np.asarray(rads)[np.abs(Lb - Lb.min()) < 1e-6].mean()) if rads is not None else sp['r']
        r_l = float(np.asarray(rads)[np.abs(Lb - Lb.min()) > 1e-6].mean()) if rads is not None else sp['r']
        first = np.zeros(len(boe), bool)
        last = np.zeros(len(boe), bool)
        for b in range(len(Lb)):
            ii = np.nonzero(boe == b)[0]
            first[ii[0]] = True
            last[ii[-1]] = True
        endlong = (first | last) & ~short

        def run(mask, m):
            def patched(lat_, r_elem, **kw):
                s = orig(lat_, r_elem, **kw).clone()
                s[torch.as_tensor(mask)] *= m
                return s
            jointRule.rule_factors = patched
            try:
                o = cases.b4_ours(sp, fr.joint_of(p3))
            finally:
                jointRule.rule_factors = orig
            return fr.hold_mean(o['time'], o['RF3bot'], sp)

        def solve(mask):
            lo, hi = 0.0, np.log(30.0)
            if run(mask, 1.0) >= target:
                return 1.0
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                if run(mask, np.exp(mid)) < target:
                    lo = mid
                else:
                    hi = mid
            return float(np.exp(0.5 * (lo + hi)))
        base = run(np.zeros(len(boe), bool), 1.0)
        L.append(f"| {case} | {r_l:g} | {r_s:g} | {Lb.min() / r_s:.1f} | {target:.3f} | {base:.3f} | {100 * (base / target - 1):+.1f} % | {solve(short):.2f} | {solve(endlong):.2f} |")
        print(L[-1], flush=True)
    (ROOT / 'results' / 'short_strut_calibration.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or NAMES)
