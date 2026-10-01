"""Standard MMA and GCMMA (src/utilFuncs.py) on problems with known optima, incl. many design variables."""
import numpy as np
import pytest
from scipy.optimize import Bounds

from utilFuncs import custom_minimize

OPTS = dict(maxiter=200, miniter=3, kkttol=1e-6, move_limit=0.2, disp=False)


@pytest.mark.parametrize('method,variant', [('MMA', 'standard'), ('GCMMA', None)])
def test_constrained_quadratic(method, variant):
    """min (x1-2)^2 + (x2-2)^2  s.t.  x1 + x2 <= 2, x in [0, 3]^2 (as normalised [0,1] with x = 3 xn): opt (1,1)."""
    obj = lambda xn: (np.array([((3 * xn - 2) ** 2).sum()]), (2 * (3 * xn - 2) * 3).reshape(-1, 1))
    con = lambda xn: (np.array([[(3 * xn).sum() - 2]]), np.array([[3.0, 3.0]]))
    res = custom_minimize(obj, np.array([0.2, 0.9]), Bounds([0, 0], [1, 1]), method=method, constraintCall=con,
                          options=dict(OPTS, mma_variant=variant))
    assert np.allclose(3 * res.x, [1, 1], atol=1e-3)


@pytest.mark.parametrize('method,variant', [('MMA', 'standard'), ('GCMMA', None)])
def test_many_variables_with_volume_constraint(method, variant):
    """n = 300 'compliance-like' separable problem: min sum w_i / x_i  s.t.  mean(x) <= 0.4, x in [0.01, 1].
    KKT optimum: x_i proportional to sqrt(w_i) (scaled to the volume), the classic optimality criterion."""
    n = 300
    w = np.linspace(0.5, 2.0, n)
    obj = lambda x: (np.array([(w / x).sum() / n]), (-w / x**2 / n).reshape(-1, 1))
    con = lambda x: (np.array([[x.mean() / 0.4 - 1]]), np.full((1, n), 1 / (0.4 * n)))
    hist = []
    res = custom_minimize(obj, np.full(n, 0.4), Bounds(np.full(n, 0.01), np.ones(n)), method=method,
                          constraintCall=con, options=dict(OPTS, mma_variant=variant),
                          callback=lambda x: hist.append(obj(x)[0][0] + 1000 * max(con(x)[0][0, 0], 0.0)))
    x_opt = np.sqrt(w) * 0.4 * n / np.sqrt(w).sum()
    assert np.allclose(res.x, x_opt, rtol=1e-3)
    if method == 'GCMMA':
        # GCMMA: monotone in its merit f0 + c * max(g, 0) (c = 1000), not necessarily in f0 alone
        assert all(b <= a + 1e-9 for a, b in zip(hist, hist[1:])), np.diff(hist).max()
