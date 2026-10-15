"""Hold-out check of the joint rule on the IMPACT solids (not used for the fit): B2a-h (cylinder + sphere joints, linear), F1 (FHS,
linear), H1 (FHS, EPU 45). Metrics as the H / F series: max stroke, energy absorbed at 50 / 75 % of the crush, peak force.
Ours is run energy-matched to the solid (impactor KE at t = 0). `bare` = no joint model; `base` = the joint model recorded for the case (JOINT_SV for B2,
JOINT_FHS for F / H); `rule:<fit.json>` = the fitted closed-form rule.
NOTE: the F1e/f, H1e/f (BCC) and the other F / H solids were built with the pre-2026-10-11 FHS geometry; the BCC ones are
known to be unreliable for stiffness (B4 vs B6), so they are reported but not used to judge.
    PYTHONPATH=src:tests python abaqus/rule_holdout.py <case> base|rule:<fit.json>      (one line, appended to results/rule_fit/holdout.log)"""
import json
import pathlib
import resource
import sys

import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (20 * 1024 ** 3,) * 2)
ROOT = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT)]
import cases                                                     # noqa: E402
import compare                                                   # noqa: E402
import h_series_analysis as ha                                   # noqa: E402

case, model = sys.argv[1], sys.argv[2]
c = cases.CASES[case]
spec = c['spec']
if model == 'bare':
    joint = None
elif model == 'base':                                # the joint model recorded for the case: JOINT_SV (B2, sphere joints), JOINT_FHS (F / H)
    joint = cases.JOINT_SV if case.startswith('B2') else c.get('joint', spec.get('joint'))
else:
    par = json.loads(pathlib.Path(model[5:]).read_text())['params']
    epu = spec.get('material', 'epu45') == 'epu45' and case.startswith('H')
    joint = dict(mode='rule', a=par['a'], p=par['p'], zl=par['zl'], b=par['b'], mu=par['mu'] if epu else 1.0)
ab = compare.read_abaqus(spec['name'])
ke0 = compare.pick(ab, 'ALLKE')[1][0]
L = compare.labels(case)
ta, ua = compare.pick(ab, 'U3', ('node', L['ref']))
ka = compare.pick(ab, 'ALLKE')
ra = compare.pick(ab, 'RF3', ('sum', L['bot']))
sa, ea, tma, sma = ha.loading_curve(ta, ua, np.interp(ta, ka[0], ka[1]))
o = cases.a4_ours(dict(spec, energy_J=ke0 * 1e-3, joint=joint))
so, eo, tmo, smo = ha.loading_curve(o['time'], o['U3ref'], o['ALLKE'])
q = np.array([0.5, 0.75]) * sma
dE = ha.absorbed_at(so, eo, q) / ha.absorbed_at(sa, ea, q) - 1
line = (f"{case:5s} {model[:30]:30s} stroke {100 * (smo / sma - 1):+6.1f} %  E50 {100 * dE[0]:+6.1f} %  E75 {100 * dE[1]:+6.1f} %  "
        f"peak {100 * (np.abs(o['RF3bot']).max() / np.abs(ra[1]).max() - 1):+6.1f} %")
print(line, flush=True)
with open(ROOT / 'results' / 'rule_fit' / 'holdout.log', 'a') as f:
    f.write(line + '\n')
