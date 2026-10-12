"""B1 single-strut EPU 45 checks: our beam predictions with both relaxation forms (nominal = current, kirchhoff =
candidate), recorded BEFORE the Abaqus runs, and the comparison once abaqus/csv/<name>_history.csv exists.
    PYTHONPATH=src:tests python abaqus/b1_predict.py [B1a ...]   -> abaqus/results/B1_summary.md"""
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
                work=np.trapezoid(RF[k], U[k]))


def main(names):
    import runStatus as rs
    rs.start('B1 predictions ' + ' '.join(names))
    head = ['# B1: single EPU 45 strut, prescribed ramp-and-hold, beam (ours) vs Abaqus solid', '',
            'Ours with the Prony branches relaxing the nominal axial force (current) or the Kirchhoff-type force '
            '(candidate). Abaqus columns appear once the run is extracted. Reaction = sum RF at the bottom; work = int RF dU '
            'during the ramp.', '',
            '| case | law | peak reaction [N] | at end of ramp [N] | at end of hold [N] | ramp work [N mm] |', '|---|---|---|---|---|---|']
    lines = list(head)
    for i, case in enumerate(names):
        sp = cases.CASES[case]['spec']
        for j, relax in enumerate(('nominal', 'kirchhoff')):
            rs.update(f'{case} ours {relax}', 2 * i + j, 2 * len(names))
            o = cases.b1_ours(sp, relax=relax)
            s = stats(o['time'], o['Udrv'], o['RFbot'], sp)
            lines.append(f"| {case} | ours, {relax} | {s['peak']:.4g} | {s['ramp_end']:.4g} | {s['hold_end']:.4g} | {s['work']:.4g} |")
        ab = compare.read_abaqus(sp['name'])
        if ab is not None:
            L = compare.labels(case)
            comp = '1' if sp['mode'] == 'bend' else '3'
            rf, ud = compare.pick(ab, 'RF' + comp, ('sum', L['bot'])), compare.pick(ab, 'U' + comp, ('node', L['drv']))
            s = stats(rf[0], np.interp(rf[0], *ud), rf[1], sp)
            lines.append(f"| {case} | **Abaqus** | {s['peak']:.4g} | {s['ramp_end']:.4g} | {s['hold_end']:.4g} | {s['work']:.4g} |")
        print('\n'.join(lines[-2:]), flush=True)
    (ROOT / 'results' / 'B1_summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or list(cases.B1_CASES))
