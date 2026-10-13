"""B4 full BCC cell (H1e's solid) under prescribed plate motion: our beams bare and with JOINT_FHS (joint factors
sliced from the OLD FHS geometry = the H1e / B4 solid, and from the current one), linear E0 vs EPU 45; Abaqus rows
and the joint stiffening once the CSVs exist.
    PYTHONPATH=src:tests python abaqus/b4_predict.py   -> abaqus/results/B4_summary.md"""
import functools
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


def ours(sp, joint, old_geometry):
    import nodeJoint as nj
    orig = nj.lattice_fhs
    if old_geometry:                                             # the FHS geometry before 2026-10-11 (= this solid)
        nj.lattice_fhs = functools.partial(orig, ball=False, contain=False)
    try:
        return cases.b4_ours(sp, joint)
    finally:
        nj.lattice_fhs = orig


def main(names):
    import runStatus as rs
    rs.start('B4 predictions ' + ' '.join(names))
    L = ['# B4: full BCC r 1.0 cell (H1e solid), prescribed plate motion', '',
         'Reaction = sum RF3 at the bottom plate; work = |int RF dU| during the ramp; stiffening = reaction at the end of '
         'the ramp / bare beams. JOINT_FHS (old geometry) uses the joint factors of the FHS geometry the solid was '
         'built from (before 2026-10-11); (current) those of the present FHS code.', '',
         '| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | stiffening vs bare |',
         '|---|---|---|---|---|---|---|']
    models = (('ours, bare beams', None, False), ('ours, JOINT_FHS (old geometry)', cases.JOINT_FHS, True),
              ('ours, JOINT_FHS (current)', cases.JOINT_FHS, False))
    for i, case in enumerate(names):
        sp = cases.CASES[case]['spec']
        res = {}
        for j, (lab, jm, old) in enumerate(models):
            rs.update(f'{case} {lab}', len(models) * i + j, len(models) * len(names))
            o = ours(sp, jm, old)
            res[lab] = stats(o['time'], o['U3ref'], o['RF3bot'], sp)
        ab = compare.read_abaqus(sp['name'])
        if ab is not None:
            lb = compare.labels(case)
            rf, ud = compare.pick(ab, 'RF3', ('sum', lb['bot'])), compare.pick(ab, 'U3', ('node', lb['ref']))
            res['**Abaqus solid**'] = stats(rf[0], np.interp(rf[0], *ud), rf[1], sp)
        b = res['ours, bare beams']['ramp_end']
        for lab, s in res.items():
            L.append(f"| {case} | {lab} | {s['peak']:.4g} | {s['ramp_end']:.4g} | {s['hold_end']:.4g} | {s['work']:.4g} | "
                     f"{s['ramp_end'] / b:.3f} |")
        print('\n'.join(L[-len(res):]), flush=True)
    (ROOT / 'results' / 'B4_summary.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or list(cases.B4_CASES))
