"""B3 BCC joint specimen: our beams bare and with JOINT_FHS, linear E0 vs EPU 45 (fast / slow), recorded before the
Abaqus runs; Abaqus rows and the joint stiffening (solid / bare vs JOINT_FHS / bare) once the CSVs exist.
    PYTHONPATH=src:tests python abaqus/b3_predict.py   -> abaqus/results/B3_summary.md"""
import pathlib
import resource
import sys

import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 ** 3,) * 2)
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases                                                     # noqa: E402
import compare                                                   # noqa: E402


def stats(t, U, RF, sp):
    tr, te = sp['t_ramp'], sp['t_ramp'] + sp['t_hold']
    k = t <= tr
    return dict(peak=np.abs(RF).max(), ramp_end=np.interp(tr, t, RF), hold_end=np.interp(te, t, RF),
                work=abs(np.trapezoid(RF[k], U[k])))


def main(names):
    import runStatus as rs
    rs.start('B3 predictions ' + ' '.join(names))
    L = ['# B3: BCC centre-node joint specimen (8 diagonals, FHS joint), prescribed plate motion', '',
         'Reaction = sum RF3 at the bottom plate; work = |int RF dU| during the ramp. Joint stiffening = reaction at the end '
         'of the ramp / the bare-beam value (ours: what JOINT_FHS adds; Abaqus: what the solid joint adds).', '',
         '| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | stiffening vs bare (ramp end) |',
         '|---|---|---|---|---|---|---|']
    for i, case in enumerate(names):
        sp = cases.CASES[case]['spec']
        res = {}
        for j, (lab, jm) in enumerate((('ours, bare beams', None), ('ours, JOINT_FHS', cases.JOINT_FHS))):
            rs.update(f'{case} {lab}', 2 * i + j, 2 * len(names))
            o = cases.b3_ours(sp, jm)
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
    (ROOT / 'results' / 'B3_summary.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or list(cases.B3_CASES))
