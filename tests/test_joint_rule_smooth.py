"""Differentiability audit of the closed-form joint rule for gradient-based design (src/jointRule.py):
- with `smooth` > 0 every kink (clamps, min / max, the zone edge inside an element, the short-strut onset) is replaced by a smooth version, so autograd equals central
  finite differences at ANY radii (random draws on the C4 cell, per-strut radii, the R5_C4 preset);
- the factors are continuous in the radii (no jumps along a radius scan);
- the gradient w.r.t. the rule PARAMETERS (a, zl, b, p, wsat, short_gain) is exact too (for fitting through the solver);
- the smooth rule stays close to the hard one (a few %)."""
import numpy as np
import torch

import jointRule
from latticeSetup import example3

torch.set_default_dtype(torch.float64)
PRE = dict(jointRule.PRESETS['R5_C4'])
PRE.pop('mode')
PRE.pop('smooth')                                                         # passed explicitly in each test


def _lat():
    return example3(elemSize=1.0, nx=1, ny=1, name='C4')


def _fd_check(f, x, idx, h=1e-6):
    g, = torch.autograd.grad(f(x), x)
    d = torch.zeros_like(x)
    d[idx] = 1.0
    fd = (f(x.detach() + h * d) - f(x.detach() - h * d)) / (2 * h)
    return float((g * d).sum()), float(fd)


def test_gradient_equals_fd_at_random_radii_smooth():
    lat = _lat()
    ne = len(lat['connr'])
    rng = np.random.default_rng(1)
    w = torch.as_tensor(rng.random(ne))
    worst = 0.0
    for trial in range(12):
        r = torch.as_tensor(rng.uniform(0.45, 1.05, ne)).requires_grad_(True)
        f = lambda rr: (w * jointRule.rule_factors(lat, rr, smooth=0.05, **PRE)).sum()
        for idx in rng.choice(ne, 6, replace=False):
            ad, fd = _fd_check(f, r, idx)
            worst = max(worst, abs(ad - fd) / max(abs(fd), 1e-3))
    assert worst < 1e-5, worst


def test_factors_continuous_in_radius_smooth():
    lat = _lat()
    ne = len(lat['connr'])
    rs = np.linspace(0.45, 1.05, 1500)
    s = np.array([jointRule.rule_factors(lat, torch.full((ne,), float(r)), smooth=0.05, **PRE).numpy() for r in rs])
    jump = np.abs(np.diff(s, axis=0) / s[:-1]).max()
    assert jump < 0.05, jump                                            # 0.4 % radius step: no jumps


def test_parameter_gradients_exact():
    lat = _lat()
    ne = len(lat['connr'])
    r = torch.full((ne,), 0.7)
    w = torch.as_tensor(np.random.default_rng(2).random(ne))
    base = {k: v for k, v in PRE.items() if k not in ('a', 'zl', 'b', 'p', 'wsat', 'short_gain')}
    names = ['a', 'zl', 'b', 'p', 'wsat', 'short_gain']
    x0 = torch.tensor([PRE[k] for k in names], requires_grad=True)

    def f(x):
        kw = dict(zip(names, x))
        return (w * jointRule.rule_factors(lat, r, smooth=0.05, **base, **kw)).sum()
    for i in range(len(names)):
        ad, fd = _fd_check(f, x0, i, h=1e-6)
        assert abs(ad - fd) < 1e-5 * max(abs(fd), 1.0), (names[i], ad, fd)


def test_smooth_close_to_hard():
    lat = _lat()
    ne = len(lat['connr'])
    r = torch.full((ne,), 0.7)
    sh = jointRule.rule_factors(lat, r, **PRE).numpy()
    ss = jointRule.rule_factors(lat, r, smooth=0.05, **PRE).numpy()
    assert abs(np.mean(ss) / np.mean(sh) - 1) < 0.08 and np.median(np.abs(ss / sh - 1)) < 0.08
