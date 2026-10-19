"""The shipped joint presets (src/jointRule.py PRESETS, smooth = kink-free form) on the 22 prescribed-motion specimens: hold-window-mean error, rms / max overall and per group.
Confirms that making the rule smooth for the design gradient did not change its accuracy.
    RULE_FIT_SET=4 PYTHONPATH=src:tests python abaqus/preset_check.py   -> abaqus/results/preset_check.md"""
import concurrent.futures as cf
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
import fit_rule2 as fr                                           # noqa: E402


def main():
    import jointRule
    import runStatus as rs
    ref = fr.solid_ref()
    names = ['R5', 'R5_C4', 'R5_all22']
    jobs = [(c, dict(jointRule.PRESETS[n])) for n in names for c in fr.CASES]
    rs.start('joint presets (smooth) on 22 specimens')
    res = {}
    with cf.ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context('spawn'), initializer=fr._init) as pool:
        for i, (c, v) in enumerate(pool.map(fr.run_case, jobs)):
            res[(names[i // len(fr.CASES)], c)] = v
            rs.update(f'{names[i // len(fr.CASES)]} {c}', i + 1, len(jobs))
    L = ['# Shipped joint presets (smooth form) on the 22 prescribed-motion specimens (error = ours / solid - 1 [%], hold-window mean)', '',
         '| preset | rms | max | ' + ' | '.join(f'{g}' for g in fr.GROUPS) + ' |', '|---|---|---|' + '---|' * len(fr.GROUPS)]
    for n in names:
        e = {c: 100 * (res[(n, c)] / ref[c] - 1) for c in fr.CASES}
        v = np.array(list(e.values()))
        gs = [np.sqrt(np.mean([e[c] ** 2 for c in cs])) for cs in fr.GROUPS.values()]
        L.append(f'| {n} | {np.sqrt(np.mean(v ** 2)):.1f} | {np.abs(v).max():.1f} | ' + ' | '.join(f'{x:.1f}' for x in gs) + ' |')
    L += ['', '| preset | ' + ' | '.join(fr.CASES) + ' |', '|---|' + '---|' * len(fr.CASES)]
    for n in names:
        L.append(f'| {n} | ' + ' | '.join(f"{100 * (res[(n, c)] / ref[c] - 1):+.1f}" for c in fr.CASES) + ' |')
    (ROOT / 'results' / 'preset_check.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
