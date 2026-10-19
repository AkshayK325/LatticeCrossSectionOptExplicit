"""The geometry-based slicing model (cases.JOINT_FHS: fused-section slices of the real FHS joint, jointSection.element_factors_aniso) on
ALL prescribed-motion specimens of fit set 4, for several participation decays lambda (weight exp(-d / (lambda r)) of fused material outside
the strut's own circle; the model was recorded with lambda = r, chosen before any evaluation and never tuned). Not differentiable in r.
    RULE_FIT_SET=4 PYTHONPATH=src:tests python abaqus/slice_decay.py   -> abaqus/results/slice_decay.md"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
import fit_rule2 as fr                                           # noqa: E402

DECAYS = (1.0, 2.0, 4.0, 8.0, None)


def run(args):
    case, decay = args
    import cases
    sp = cases.CASES[case]['spec']
    fn = getattr(cases, fr.runner_name(case))
    j = dict(cases.JOINT_FHS)
    j['decay'] = decay
    o = fn(sp, j)
    return case, decay, fr.hold_mean(o['time'], o['RF3bot'], sp)


def main():
    import runStatus as rs
    ref = fr.solid_ref()
    for c in fr.CASES:                                        # the specimens of set 4 only
        assert c in ref
    cache = ROOT / 'results' / 'slice_decay_cache.json'                # resumable: every finished (case, lambda) is kept
    done = json.loads(cache.read_text()) if cache.exists() else {}
    jobs = [(c, d) for c in sorted(fr.CASES, key=lambda c: -len(c)) for d in DECAYS]
    todo = [j for j in jobs if f'{j[0]}|{j[1]}' not in done]
    rs.start(f'slicing model, decay scan on 22 specimens ({len(jobs) - len(todo)} of {len(jobs)} cached)')
    with cf.ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context('spawn'), initializer=fr._init) as pool:
        for i, (c, d, v) in enumerate(pool.map(run, todo)):
            done[f'{c}|{d}'] = v
            cache.write_text(json.dumps(done))
            rs.update(f'{c} lambda={d}', i + 1, len(todo))
    res = {(c, d): done[f'{c}|{d}'] for c, d in jobs}
    L = ['# Slicing model (JOINT_FHS shape) vs the prescribed-motion solids, by participation decay lambda (units of r; None = no decay)', '',
         'error = ours / solid - 1 [%], hold-window means; rms / max over the specimens.', '',
         '| lambda | ' + ' | '.join(fr.CASES) + ' | rms | max |', '|---|' + '---|' * (len(fr.CASES) + 2)]
    for d in DECAYS:
        e = np.array([100 * (res[(c, d)] / ref[c] - 1) for c in fr.CASES])
        L.append(f'| {d} | ' + ' | '.join(f'{x:+.1f}' for x in e) + f' | {np.sqrt(np.mean(e ** 2)):.1f} | {np.abs(e).max():.1f} |')
    (ROOT / 'results' / 'slice_decay.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
