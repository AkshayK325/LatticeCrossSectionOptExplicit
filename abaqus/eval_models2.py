"""Table of all joint models on the 10 linear new-geometry prescribed-motion solids of fit_rule2 (hold-window mean of the reaction),
including the fits in results/rule_fit2/*.json (full fits scored on all cases; leave-one-group-out fits scored on the held-out group).
Baselines: bare, zone k = 3.5, JOINT_FHS, JOINT_FHS + plate_layer, and the FIRST rule (M3 of results/rule_fit, p = 4.10, zl = 0.71, b = 0.53).
    PYTHONPATH=src:tests python abaqus/eval_models2.py   -> abaqus/results/rule_fit2/model_table.md"""
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
        joint = {'bare': None, 'fhs': cases.JOINT_FHS, 'fhs_layer': dict(cases.JOINT_FHS, plate_layer=True)}.get(kind)
        if kind == 'rule':
            joint = fr.joint_of(par)
        elif kind == 'rule1':
            joint = dict(mode='rule', **dict(par, mu=1.0))
        o = fn(sp, joint)
    return case, fr.hold_mean(o['time'], o['RF3bot'], sp)


def main():
    import runStatus as rs                                       # repo rule: every long job reports to run_status.log
    rs.start('joint model table ')
    ref = fr.solid_ref()
    models = [('bare', 'bare', None, None), ('zone k=3.5', 'zone', None, None), ('JOINT_FHS', 'fhs', None, None),
              ('JOINT_FHS+plate_layer', 'fhs_layer', None, None)]
    first = json.loads((ROOT / 'results' / 'rule_fit' / 'M3.json').read_text())['params']
    models.append(('first rule M3 (B3/B5/B6 fit)', 'rule1', first, None))
    out = ROOT / 'results' / {'2': 'rule_fit2', '3': 'rule_fit3', '4': 'rule_fit4'}[fr.SET]
    for f in sorted(out.glob('R*.json')):
        d = json.loads(f.read_text())
        models.append((f.stem + (' (all, in sample)' if not d['hold'] else ' (held out)'), 'rule', d['params'], d['hold']))
    jobs, meta = [], []
    for name, kind, par, hold in models:
        for c in (fr.GROUPS[hold] if hold else fr.CASES):
            jobs.append((c, kind, par)); meta.append(name)
    with cf.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=fr._init) as pool:
        futs = [pool.submit(run_model, j) for j in jobs]
        for k, fut in enumerate(cf.as_completed(futs)):
            rs.update('model x case jobs', k + 1, len(jobs))
        res = [fut.result() for fut in futs]
    tab = {}
    for name, (c, v) in zip(meta, res):
        tab.setdefault(name, {})[c] = v / ref[c] - 1
    L = ['# Joint models on the linear new-geometry prescribed-motion solids (hold-window mean of the reaction; fit set ' + fr.SET + ')', '',
         'error = ours / Abaqus solid - 1 [%]; rms over the scored cases; leave-one-group-out rows are scored only on the group left out of that fit.', '',
         '| model | ' + ' | '.join(fr.CASES) + ' | rms | max |', '|---|' + '---|' * (len(fr.CASES) + 2)]
    for name, row in tab.items():
        v = np.array(list(row.values())) * 100
        L.append(f"| {name} | " + ' | '.join(f"{100 * row[c]:+.1f}" if c in row else '' for c in fr.CASES) + f" | {np.sqrt(np.mean(v ** 2)):.1f} | {np.abs(v).max():.1f} |")
    (out / 'model_table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
