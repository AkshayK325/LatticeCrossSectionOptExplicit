"""Time-step (Courant) sensitivity of the plateau objective J for the bare, k=3.5 and R5_C4 joint models.
J(nominal adaptive schedule at the given courant) at two designs. Shows how much of the objective is time-step discretisation.
    PYTHONPATH=src:tests python -u tests/study_courant_sensitivity.py cpu"""
import os, sys
sys.argv = ['x', sys.argv[1] if len(sys.argv) > 1 else 'cpu', '3.5e-3']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_plateau_gradients.py')).read()
exec(compile(src[:src.index("rs.start(")], 'chk', 'exec'))
import runStatus as rs
rs.start('courant sensitivity of plateau J', )
rows_out = ['# Courant sensitivity of the plateau objective J (full drop, example3, nominal adaptive schedule)', '',
            '| joint | design (3 groups) | courant 0.7 | 0.5 | 0.35 | 0.25 | steps at 0.7 / 0.25 |', '|---|---|---|---|---|---|---|']
combos = [(j, gv) for j in ('none', 'k3.5', 'R5_C4') for gv in ([0.5, 0.5, 0.5], [0.15, 0.55, 0.9])]
for ci, (j, gv) in enumerate(combos):
    kw0 = {k: v for k, v in rig_kw.items() if k != 'joint'}
    if j == 'k3.5':
        kw0['joint'] = {'k': 3.5}
    elif j == 'R5_C4':
        kw0['joint'] = dict(jointRule.PRESETS['R5_C4'])
    Js, ns = [], []
    for c in (0.7, 0.5, 0.35, 0.25):
        rs.update(f'{j} {gv} courant {c}', ci, len(combos))
        x0 = rows(gv)
        with torch.no_grad():
            rig0, sched = nominal(lat, den(x0), SYM, mat, T=3.5e-3, **dict(kw0, courant=c))
        rig_kw_save = rig_kw; rig_kw = kw0
        Js.append(fdJ(x0, rig0, sched)); ns.append(len(sched))
        rig_kw = rig_kw_save
    rows_out.append(f'| {j} | {gv} | ' + ' | '.join(f'{v:.5f}' for v in Js) + f' | {ns[0]} / {ns[-1]} |')
    print(rows_out[-1], flush=True)
import pathlib
(pathlib.Path(__file__).resolve().parents[1] / 'abaqus' / 'results' / 'courant_sensitivity.md').write_text('\n'.join(rows_out) + '\n')
