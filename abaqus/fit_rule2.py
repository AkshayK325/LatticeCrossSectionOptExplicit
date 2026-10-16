"""Second fit of the closed-form joint rule (src/jointRule.py), after B7 / B8 (notes/18 sections 10-11).
Differences from fit_rule.py (kept as the record of the first fit):
  * the solid's static value is the MEAN of the reaction over the last 70 % of the hold (linear solids ring by +-3-6 %); ours likewise;
  * ONLY linear-elastic cases are fitted (the EPU 45 stiffening is smaller and rate dependent and does not transfer to impact: notes/18 9);
  * data: B3a, B5a, B5c, B5d, B6a, B7a-c, B8a, B8b (10 cases, one residual each: ours / solid - 1);
  * groups for leave-one-group-out: SC (B5a, c, d), STAR (B3a, B8a, B8b), TJOINT (B7a-c), CELL (B6a);
  * angle exponent allowed negative; zone and stiffness exponents; plate-oblique factor gamma.
models (free parameters; the others at their defaults p = pz = 0, b = 0, gamma = 1):
  R0: a, zl      R1: a, zl, gamma      R2: a, zl, gamma, b      R3: a, zl, gamma, b, p (= pz)
    PYTHONPATH=src:tests python abaqus/fit_rule2.py <model> [holdout-group]  -> abaqus/results/rule_fit2/<model>[_<group>].json"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
GROUPS = {'SC': ['B5a', 'B5c', 'B5d'], 'STAR': ['B3a', 'B8a', 'B8b'], 'TJOINT': ['B7a', 'B7b', 'B7c'], 'CELL': ['B6a']}
CASES = [c for g in GROUPS.values() for c in g]
FREE = {'R0': ['a', 'zl'], 'R1': ['a', 'zl', 'gamma'], 'R2': ['a', 'zl', 'gamma', 'b'], 'R3': ['a', 'zl', 'gamma', 'b', 'p'],
        'R4': ['a', 'zl', 'gamma', 'b', 'p', 'cap'], 'R5': ['a', 'zl', 'gamma', 'b', 'p', 'wsat']}   # R4 / R5: after B9 (notes/18 11)
START = dict(a=1.0, zl=1.0, gamma=0.5, b=0.2, p=0.0, cap=0.4, wsat=3.0)
LO = dict(a=0.0, zl=0.3, gamma=0.0, b=0.0, p=-4.0, cap=0.1, wsat=0.5)
HI = dict(a=6.0, zl=3.0, gamma=1.0, b=2.0, p=10.0, cap=0.5, wsat=20.0)
BASE = dict(a=1.0, p=0.0, zl=1.0, b=0.0, mu=1.0, pz=None, gamma=1.0, cap=0.5, wsat=None)
RUNNER = {'B3': 'b3_ours', 'B5': 'b5_ours', 'B6': 'b4_ours', 'B7': 'b78_ours', 'B8': 'b78_ours'}


def _init():
    sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
    import torch
    torch.set_num_threads(1)


def hold_mean(t, rf, sp):
    tr, te = sp['t_ramp'], sp['t_ramp'] + sp['t_hold']
    w = (np.asarray(t) > tr + 0.3 * (te - tr)) & (np.asarray(t) <= te)
    return float(np.mean(np.asarray(rf)[w]))


def run_case(args):
    case, joint = args
    import cases
    sp = cases.CASES[case]['spec']
    o = getattr(cases, RUNNER[case[:2]])(sp, joint)
    return case, hold_mean(o['time'], o['RF3bot'], sp)


def solid_ref():
    sys.path[:0] = [str(ROOT), str(ROOT.parent / 'src'), str(ROOT.parent / 'tests')]
    import cases
    import compare
    out = {}
    for case in CASES:
        sp = cases.CASES[case]['spec']
        ab = compare.read_abaqus(sp['name'])
        lb = compare.labels(case)
        t, rf = compare.pick(ab, 'RF3', ('sum', lb['bot']))
        out[case] = hold_mean(t, rf, sp)
    return out


def joint_of(par):
    j = dict(BASE); j.update(par)
    j['pz'] = j['p']                      # one angle exponent for zone and stiffness
    return dict(mode='rule', **j)


class Fit:
    def __init__(self, model, hold=None, workers=6):
        self.model, self.hold = model, hold
        self.train = [c for g, cs in GROUPS.items() if g != hold for c in cs]
        self.ref = solid_ref()
        self.free = FREE[model]
        self.pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=_init)
        self.n = 0

    def params(self, x):
        par = dict(zip(self.free, x))
        return par

    def predict(self, x, cases_):
        j = joint_of(self.params(x))
        return dict(self.pool.map(run_case, [(c, j) for c in cases_]))

    def resid(self, x):
        pr = self.predict(x, self.train)
        self.n += 1
        r = np.array([pr[c] / self.ref[c] - 1 for c in self.train])
        print(f'  eval {self.n}: x={np.round(x, 3).tolist()} rms={np.sqrt(np.mean(r ** 2)) * 100:.2f} %', flush=True)
        return r

    def fit(self):
        from scipy.optimize import least_squares
        x0 = np.array([START[k] for k in self.free])
        res = least_squares(self.resid, x0, bounds=([LO[k] for k in self.free], [HI[k] for k in self.free]),
                            diff_step=0.05, max_nfev=12, x_scale=np.maximum(np.abs(x0), 0.5))
        return res.x


def main():
    model = sys.argv[1]
    hold = sys.argv[2] if len(sys.argv) > 2 else None
    f = Fit(model, hold)
    x = f.fit()
    pr = f.predict(x, CASES)
    err = {c: pr[c] / f.ref[c] - 1 for c in CASES}
    out = dict(model=model, hold=hold, params=f.params(x), err=err, train=f.train, ref=f.ref)
    d = ROOT / 'results' / 'rule_fit2'
    d.mkdir(exist_ok=True)
    (d / f"{model}{'_' + hold if hold else ''}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out['params']), {c: round(100 * e, 1) for c, e in err.items()}, flush=True)


if __name__ == '__main__':
    main()
