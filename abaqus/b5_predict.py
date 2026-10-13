"""B5 simple cubic cells (no internal joint) under prescribed plate motion: blind predictions of our beams bare, with
JOINT_FHS, and with JOINT_FHS + plate_layer (jointSection: the plate-bonded layer removed from the plate-end slices);
Abaqus rows and the joint stiffening once the CSVs exist.
    PYTHONPATH=src:tests python abaqus/b5_predict.py [B5a ...]   -> abaqus/results/B5_summary.md"""
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


def main(names):
    import runStatus as rs
    rs.start('B5 predictions ' + ' '.join(names))
    L = ['# B5: simple cubic cells (no internal joint), prescribed plate motion', '',
         'Reaction = sum RF3 at the bottom plate; work = |int RF dU| during the ramp; stiffening = reaction at the end of '
         'the ramp / bare beams. Current FHS geometry for the solid and the joint factors.', '',
         '| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | stiffening vs bare |',
         '|---|---|---|---|---|---|---|']
    models = (('ours, bare beams', None), ('ours, JOINT_FHS', cases.JOINT_FHS),
              ('ours, JOINT_FHS + plate_layer', dict(cases.JOINT_FHS, plate_layer=True)))
    for i, case in enumerate(names):
        sp = cases.CASES[case]['spec']
        res = {}
        for j, (lab, jm) in enumerate(models):
            rs.update(f'{case} {lab}', len(models) * i + j, len(models) * len(names))
            o = cases.b5_ours(sp, jm)
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
        (ROOT / 'results' / 'B5_summary.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or list(cases.B5_CASES))
