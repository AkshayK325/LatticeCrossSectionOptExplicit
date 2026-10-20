"""Fit (and leave-one-out validate) the two numbers of the short-strut law (short_gain, short_l0; rule R5 preset otherwise) on the eight C4 solids B10a-d and B13a-d, with
C12 (B9c, B9d) and the unequal-radius T-joints (B11a-d) as 'must not change' controls. Grid over gain and onset; the best pair, the leave-one-out errors and the controls are written to
abaqus/results/short_law_fit.md.
    RULE_FIT_SET=4 PYTHONPATH=src:tests python abaqus/short_law_fit.py"""
import concurrent.futures as cf
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
import fit_rule2 as fr                                           # noqa: E402

C4 = ['B10a', 'B10b', 'B10c', 'B10d', 'B13a', 'B13b', 'B13c', 'B13d']
CONTROL = ['B9c', 'B9d', 'B11a', 'B11b', 'B11c', 'B11d', 'B9b']
GAINS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
ONSETS = (4.0, 4.3, 4.6)


def solid(case):
    import cases
    import compare
    sp = cases.CASES[case]['spec']
    ab = compare.read_abaqus(sp['name'])
    lb = compare.labels(case)
    t, rf = compare.pick(ab, 'RF3', ('sum', lb['bot']))
    return fr.hold_mean(t, rf, sp)


def main():
    import jointRule
    import runStatus as rs
    sys.path[:0] = [str(ROOT)]
    ref = {c: solid(c) for c in C4 + CONTROL}
    base = {k: v for k, v in jointRule.PRESETS['R5'].items()}
    grid = [(g, l0) for g in GAINS for l0 in ONSETS]
    jobs = [(c, fr_joint(dict(base, short_gain=g, short_l0=l0))) for g, l0 in grid for c in C4 + CONTROL]
    rs.start('short-strut law fit on 8 C4 solids')
    out = {}
    with cf.ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context('spawn'), initializer=fr._init) as pool:
        for i, (c, v) in enumerate(pool.map(fr.run_case, jobs)):
            g, l0 = grid[i // len(C4 + CONTROL)]
            out[(g, l0, c)] = 100 * (v / ref[c] - 1)
            rs.update(f'g={g} l0={l0} {c}', i + 1, len(jobs))
    L = ['# Short-strut law (short_gain g, onset short_l0) fitted on the eight C4 solids, controls must not change', '',
         'error = ours / solid - 1 [%] (hold-window mean); rms over the 8 C4 solids and over the 7 controls (C12, unequal-radius T-joints, octet r 1.5).', '',
         '| g | l0 | rms C4 | max C4 | rms controls | max controls | ' + ' | '.join(C4) + ' |', '|---|---|---|---|---|---|' + '---|' * len(C4)]
    best = None
    for g, l0 in grid:
        e4 = np.array([out[(g, l0, c)] for c in C4])
        ec = np.array([out[(g, l0, c)] for c in CONTROL])
        r4 = np.sqrt(np.mean(e4 ** 2))
        L.append(f'| {g} | {l0} | {r4:.1f} | {np.abs(e4).max():.1f} | {np.sqrt(np.mean(ec ** 2)):.1f} | {np.abs(ec).max():.1f} | ' + ' | '.join(f'{x:+.1f}' for x in e4) + ' |')
        if best is None or r4 < best[0]:
            best = (r4, g, l0)
    L += ['', f'best on all 8: g = {best[1]}, l0 = {best[2]} (rms {best[0]:.1f} %)', '', 'Leave-one-out (fit on the other 7, error on the held-out solid):', '', '| held out | g | l0 | error [%] |', '|---|---|---|---|']
    loo = []
    for h in C4:
        r_best = None
        for g, l0 in grid:
            e = np.array([out[(g, l0, c)] for c in C4 if c != h])
            r = np.sqrt(np.mean(e ** 2))
            if r_best is None or r < r_best[0]:
                r_best = (r, g, l0)
        eh = out[(r_best[1], r_best[2], h)]
        loo.append(eh)
        L.append(f'| {h} | {r_best[1]} | {r_best[2]} | {eh:+.1f} |')
    L += ['', f'leave-one-out rms {np.sqrt(np.mean(np.square(loo))):.1f} %, max {np.abs(loo).max():.1f} %']
    (ROOT / 'results' / 'short_law_fit.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def fr_joint(par):
    return fr.joint_of({k: v for k, v in par.items() if k != 'mode'})


if __name__ == '__main__':
    main()
