"""Fit and cross-validate the closed-form joint rule (src/jointRule.py) against the Abaqus solids that use the CURRENT FHS
geometry and prescribed plate motion: B3a-c (BCC centre node), B5a-d (simple cubic), B6a-b (BCC cell).
(B4 / H / F / the older BCC solids used the pre-2026-10-11 geometry and are NOT used for fitting.)

Residuals per case: ours / solid - 1 of the sum reaction at the end of the ramp and at the end of the hold (the hold is
relaxed for EPU 45). Gradient-free outer fit (scipy least_squares, finite-difference Jacobian): the derivative of the
reaction w.r.t. the rule parameters through the explicit solver is NOT used here; only the design gradient w.r.t. the radii
is exact (tests/test_joint_stiffness.py, tests/test_joint_rule_adjoint.py).
    PYTHONPATH=src:tests python abaqus/fit_rule.py <model> [holdout-group]        -> abaqus/results/rule_fit/<model>[_<group>].json
models: M0 (a), M1 (a, p), M3 (a, p, zl, b); every model has a free material factor mu for the EPU 45 cases."""
import concurrent.futures as cf
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
GROUPS = {'SC': ['B5a', 'B5b', 'B5c', 'B5d'], 'NODE': ['B3a', 'B3b', 'B3c'], 'CELL': ['B6a', 'B6b']}
CASES = [c for g in GROUPS.values() for c in g]
EPU = {'B3b', 'B3c', 'B5b', 'B6b'}
FREE = {'M0': ['a'], 'M1': ['a', 'p'], 'M3': ['a', 'p', 'zl', 'b']}
START = dict(a=1.0, p=4.0, zl=1.0, b=0.0, mu=0.7)
LO = dict(a=0.0, p=0.0, zl=0.3, b=0.0, mu=0.1)
HI = dict(a=6.0, p=12.0, zl=3.0, b=2.0, mu=1.5)
FIXED = {'M0': dict(p=0.0, zl=1.0, b=0.0), 'M1': dict(zl=1.0, b=0.0), 'M3': {}}


def _init():
    sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
    import torch
    torch.set_num_threads(1)


def run_case(args):
    case, par = args
    import cases
    from b3_predict import stats
    sp = cases.CASES[case]['spec']
    fn = cases.b3_ours if case.startswith('B3') else cases.b4_ours if case.startswith('B6') else cases.b5_ours
    joint = dict(mode='rule', **par) if par is not None else None
    o = fn(sp, joint)
    s = stats(o['time'], o['U3ref'], o['RF3bot'], sp)
    return case, s['ramp_end'], s['hold_end']


def solid_ref():
    sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
    import cases
    import compare
    from b3_predict import stats
    out = {}
    for case in CASES:
        sp = cases.CASES[case]['spec']
        ab = compare.read_abaqus(sp['name'])
        lb = compare.labels(case)
        rf, ud = compare.pick(ab, 'RF3', ('sum', lb['bot'])), compare.pick(ab, 'U3', ('node', lb['ref']))
        s = stats(rf[0], np.interp(rf[0], *ud), rf[1], sp)
        out[case] = (s['ramp_end'], s['hold_end'])
    return out


class Fit:
    def __init__(self, model, hold=None):
        self.model, self.hold = model, hold
        self.train = [c for g, cs in GROUPS.items() if g != hold for c in cs]
        self.ref = solid_ref()
        self.free = FREE[model] + ['mu']
        self.pool = cf.ProcessPoolExecutor(max_workers=len(CASES), mp_context=mp.get_context('spawn'), initializer=_init)
        self.n = 0

    def params(self, x):
        par = dict(START); par.update(FIXED[self.model])
        par.update(dict(zip(self.free, x)))
        return par

    def predict(self, x, cases_):
        par = self.params(x)
        jobs = [(c, dict(par, mu=par['mu'] if c in EPU else 1.0)) for c in cases_]
        return {c: (r, h) for c, r, h in self.pool.map(run_case, jobs)}

    def resid(self, x):
        pr = self.predict(x, self.train)
        self.n += 1
        r = np.array([v for c in self.train for v in (pr[c][0] / self.ref[c][0] - 1, pr[c][1] / self.ref[c][1] - 1)])
        print(f'  eval {self.n}: x={np.round(x, 3).tolist()} rms={np.sqrt(np.mean(r ** 2)) * 100:.2f} %', flush=True)
        return r

    def fit(self):
        from scipy.optimize import least_squares
        x0 = np.array([START[k] for k in self.free])
        res = least_squares(self.resid, x0, bounds=([LO[k] for k in self.free], [HI[k] for k in self.free]),
                            diff_step=0.05, max_nfev=14, x_scale=np.maximum(x0, 0.5))
        return res.x


def main():
    model = sys.argv[1]
    hold = sys.argv[2] if len(sys.argv) > 2 else None
    f = Fit(model, hold)
    x = f.fit()
    pr = f.predict(x, CASES)
    err = {c: [pr[c][0] / f.ref[c][0] - 1, pr[c][1] / f.ref[c][1] - 1] for c in CASES}
    out = dict(model=model, hold=hold, params=f.params(x), err=err, train=f.train)
    d = ROOT / 'results' / 'rule_fit'
    d.mkdir(exist_ok=True)
    (d / f"{model}{'_' + hold if hold else ''}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out['params']), {c: [round(v * 100, 1) for v in e] for c, e in err.items()}, flush=True)


if __name__ == '__main__':
    main()
