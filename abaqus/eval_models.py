"""Table of all joint models on the new-geometry prescribed-motion solids (B3, B5, B6): error of the reaction at the end of the
ramp / end of the hold, and the rms. Models: bare, zone k = 3.5 (latticeSetup.joint_factors), JOINT_FHS (fused-section slices),
JOINT_FHS + plate_layer, and every fitted rule in results/rule_fit/*.json (full fits and leave-one-group-out fits; for a
leave-one-group-out fit only the held-out group is scored).
    PYTHONPATH=src:tests python abaqus/eval_models.py   -> abaqus/results/rule_fit/model_table.md"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
import fit_rule as fr                                            # noqa: E402


def run_model(args):
    case, kind, par = args
    import torch
    torch.set_num_threads(1)
    import cases
    from b3_predict import stats
    sp = cases.CASES[case]['spec']
    fn = cases.b3_ours if case.startswith('B3') else cases.b4_ours if case.startswith('B6') else cases.b5_ours
    if kind == 'bare':
        joint = None
    elif kind == 'fhs':
        joint = cases.JOINT_FHS
    elif kind == 'fhs_layer':
        joint = dict(cases.JOINT_FHS, plate_layer=True)
    elif kind == 'zone':
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
        s = stats(o['time'], o['U3ref'], o['RF3bot'], sp)
        return case, kind, s['ramp_end'], s['hold_end']
    else:
        joint = dict(mode='rule', **dict(par, mu=par['mu'] if case in fr.EPU else 1.0))
    o = fn(sp, joint)
    s = stats(o['time'], o['U3ref'], o['RF3bot'], sp)
    return case, kind, s['ramp_end'], s['hold_end']


def main():
    import runStatus as rs                                       # repo rule: every long job reports to run_status.log
    rs.start('joint model table ')
    ref = fr.solid_ref()
    models = [('bare', 'bare', None), ('zone k=3.5', 'zone', None), ('JOINT_FHS', 'fhs', None),
              ('JOINT_FHS+plate_layer', 'fhs_layer', None)]
    fits = sorted((ROOT / 'results' / 'rule_fit').glob('M*.json'))
    for f in fits:
        d = json.loads(f.read_text())
        models.append((f.stem + (' (all)' if not d['hold'] else ' (scored on held-out ' + d['hold'] + ')'), 'rule', d))
    jobs, meta = [], []
    for name, kind, d in models:
        scored = fr.CASES if d is None or not d.get('hold') else fr.GROUPS[d['hold']]
        for c in scored:
            jobs.append((c, kind, None if d is None else d['params']))
            meta.append(name)
    with cf.ProcessPoolExecutor(max_workers=10, mp_context=mp.get_context('spawn')) as pool:
        futs = [pool.submit(run_model, j) for j in jobs]
        for k, fut in enumerate(cf.as_completed(futs)):
            rs.update('model x case jobs', k + 1, len(jobs))
        res = [fut.result() for fut in futs]
    tab = {}
    for name, (c, kind, r, h) in zip(meta, res):
        tab.setdefault(name, {})[c] = (r / ref[c][0] - 1, h / ref[c][1] - 1)
    L = ['# Joint models on the new-geometry prescribed-motion solids (B3, B5, B6)', '',
         'error = ours / Abaqus solid - 1 of the sum reaction, end of ramp / end of hold [%]; rms over the scored cases (both columns).', '',
         '| model | ' + ' | '.join(fr.CASES) + ' | rms |', '|---|' + '---|' * (len(fr.CASES) + 1)]
    for name, row in tab.items():
        cells = [f"{100 * row[c][0]:+.1f} / {100 * row[c][1]:+.1f}" if c in row else '' for c in fr.CASES]
        rms = np.sqrt(np.mean([v ** 2 for c in row for v in row[c]])) * 100
        L.append(f"| {name} | " + ' | '.join(cells) + f" | {rms:.1f} |")
    (ROOT / 'results' / 'rule_fit' / 'model_table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
