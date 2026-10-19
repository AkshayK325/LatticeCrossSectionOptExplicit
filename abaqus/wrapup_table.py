"""Final comparison of the joint models on all 22 prescribed-motion specimens of fit set 4 (hold-window mean of the sum reaction; error = ours / solid - 1):
bare beams, zone k = 3.5, JOINT_FHS slicing (decay lambda = r and none), rule R5 (set-3 fit, 13 specimens) and R5 refitted on all 22 (set 4). Writes
abaqus/results/joint_model_summary.md with rms / max overall and per topology group.
    RULE_FIT_SET=4 PYTHONPATH=src:tests python abaqus/wrapup_table.py"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
import fit_rule2 as fr                                           # noqa: E402


def run_model(args):
    case, kind, par = args
    import torch
    torch.set_num_threads(1)
    import cases
    sp = cases.CASES[case]['spec']
    fn = getattr(cases, fr.runner_name(case))
    if kind == 'zone':
        import jointSection
        from latticeSetup import joint_factors
        orig = jointSection.element_factors_aniso

        def zone(lat, r_beam, **kw):
            re = torch.as_tensor(np.asarray(r_beam)[np.asarray(lat['beam_of_elem'])], dtype=torch.float64)
            return joint_factors(lat, re, 3.5).numpy(), None
        jointSection.element_factors_aniso = zone
        try:
            o = fn(sp, {'x': 1})
        finally:
            jointSection.element_factors_aniso = orig
    else:
        joint = {'bare': None, 'fhs': cases.JOINT_FHS, 'fhs_nodecay': dict(cases.JOINT_FHS, decay=None)}.get(kind)
        if kind == 'rule':
            joint = fr.joint_of(par)
        o = fn(sp, joint)
    return case, kind, fr.hold_mean(o['time'], o['RF3bot'], sp)


def main():
    import runStatus as rs
    ref = fr.solid_ref()
    p3 = json.loads((ROOT / 'results' / 'rule_fit3' / 'R5.json').read_text())['params']
    p4 = json.loads((ROOT / 'results' / 'rule_fit4' / 'R5.json').read_text())['params']
    models = [('bare beams', 'bare', None), ('zone k = 3.5 (differentiable)', 'zone', None), ('JOINT_FHS slicing, lambda = r (recorded; not differentiable)', 'fhs', None),
              ('JOINT_FHS slicing, no decay (not differentiable)', 'fhs_nodecay', None), ('rule R5, set-3 fit (13 specimens; differentiable)', 'rule3', p3),
              ('rule R5, refit on all 22 (differentiable)', 'rule4', p4)]
    jobs = [(c, k if not k.startswith('rule') else 'rule', par) for _, k, par in models for c in fr.CASES]
    meta = [name for name, _, _ in models for _ in fr.CASES]
    rs.start('final joint-model table on 22 specimens')
    out = {}
    with cf.ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context('spawn'), initializer=fr._init) as pool:
        for i, ((c, k, _), name, (c2, k2, v)) in enumerate(zip(jobs, meta, pool.map(run_model, jobs))):
            out.setdefault(name, {})[c] = v / ref[c] - 1
            rs.update(f'{name[:30]} {c}', i + 1, len(jobs))
    groups = fr.GROUPS
    L = ['# Joint models on the 22 prescribed-motion specimens (current FHS geometry, linear; hold-window mean; error = ours / solid - 1 [%])', '',
         '| model | rms | max | ' + ' | '.join(f'{g} rms' for g in groups) + ' |', '|---|---|---|' + '---|' * len(groups)]
    for name, row in out.items():
        e = np.array([100 * row[c] for c in fr.CASES])
        gs = [np.sqrt(np.mean([(100 * row[c]) ** 2 for c in cs])) for cs in groups.values()]
        L.append(f'| {name} | {np.sqrt(np.mean(e ** 2)):.1f} | {np.abs(e).max():.1f} | ' + ' | '.join(f'{x:.1f}' for x in gs) + ' |')
    L += ['', '| model | ' + ' | '.join(fr.CASES) + ' |', '|---|' + '---|' * len(fr.CASES)]
    for name, row in out.items():
        L.append(f'| {name[:40]} | ' + ' | '.join(f'{100 * row[c]:+.1f}' for c in fr.CASES) + ' |')
    (ROOT / 'results' / 'joint_model_summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
