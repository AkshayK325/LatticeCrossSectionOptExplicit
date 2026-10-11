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
