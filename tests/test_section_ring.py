"""Control rings for non-circular strut sections (notes/17 P6): the subdivided tube reproduces the section."""
import numpy as np
from shapely.geometry import Polygon

import nodeJoint as nj


def _rect(a, b, rho):
    return np.asarray(Polygon([(-a, -b), (a, -b), (a, b), (-a, b)]).buffer(rho, 64).exterior.coords)[:-1]


def test_circle_matches_loop_shrink():
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    for levels, limit in ((3, False), (2, True)):
        c = nj.control_ring(np.c_[np.cos(t), np.sin(t)], 16, levels, limit)
        assert abs(np.linalg.norm(c, axis=1).mean() * nj.loop_shrink(16, levels=levels, limit=limit) - 1) < 1e-4


def test_ellipse_and_rounded_rectangle_within_1_percent():
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    for xy in (np.c_[1.4 * np.cos(t), 0.7 * np.sin(t)], _rect(0.9, 0.35, 0.25), _rect(0.8, 0.8, 0.1)):
        for levels, limit in ((3, False), (2, True)):
            c, n, h = nj.section_control_ring(xy, levels=levels, limit=limit)
            assert h <= 0.01 and n % 4 == 0
            lim = nj.ring_limit_operator(n, levels, limit) @ (c[:, 0] + 1j * c[:, 1])
            assert abs(Polygon(np.c_[lim.real, lim.imag]).area / Polygon(xy).area - 1) < 0.005


def _box_union(boxes):
    from shapely.geometry import box
    from shapely.ops import unary_union
    return np.asarray(unary_union([box(*b) for b in boxes]).exterior.coords)[:-1]


def test_section_properties_rectangle():
    A, I1, I2 = nj.section_properties(np.array([(0, 0), (2, 0), (2, 1), (0, 1)], float))
    assert np.allclose((A, I1, I2), (2.0, 2 / 3, 1 / 6))


def test_round_section_exact_corner_areas():
    """A 90 deg corner rounded to rho changes the area by (1 - pi/4) rho^2: removed at a convex corner, added at a
    re-entrant one (an I-section's web-flange root fillet)."""
    rho, c = 0.05, 1 - np.pi / 4
    rect = np.array([(0, 0), (2, 0), (2, 1), (0, 1)], float)
    L = _box_union([(0, 0, 2, 0.5), (0, 0, 0.5, 2)])
    I = _box_union([(-1, -1, 1, -0.8), (-1, 0.8, 1, 1), (-0.1, -1, 0.1, 1)])   # flanges + web: 8 convex, 4 roots
    for xy, n_convex, n_root in ((rect, 4, 0), (L, 5, 1), (I, 8, 4)):
        q = nj.round_section(xy, rho, n=4000)
        dA = nj.section_properties(q)[0] - nj.section_properties(xy)[0]
        assert abs(dA - (n_root - n_convex) * c * rho ** 2) < 0.02 * n_convex * c * rho ** 2 + 1e-6


def test_round_section_refuses_topology_change():
    I = _box_union([(-1, -1, 1, -0.8), (-1, 0.8, 1, 1), (-0.05, -1, 0.05, 1)])   # web 0.1 thick
    import pytest
    with pytest.raises(ValueError):
        nj.round_section(I, 0.06)                                                 # 2 rho > web
