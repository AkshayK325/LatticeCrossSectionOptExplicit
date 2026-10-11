import numpy as np
import manifold3d as m3
import nodeJoint as nj
from shapely.geometry import Polygon


def ellipse(req, ratio, roll, m=720):
    a, b = req * np.sqrt(ratio), req / np.sqrt(ratio)                # same area as the circle of radius req
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    xy = np.c_[a * np.cos(t), b * np.sin(t)]
    c, s_ = np.cos(roll), np.sin(roll)
    return xy @ np.array([[c, s_], [-s_, c]])


def rrect(req, ratio, rho_frac, roll):
    # rounded rectangle of aspect `ratio` with corner radius rho_frac * short half-side, area pi req^2
    for _ in range(30):
        h = req / np.sqrt(ratio) if _ == 0 else h * np.sqrt(np.pi * req ** 2 / Polygon(xy).area)
        a, b, rho = h * ratio - rho_frac * h, h - rho_frac * h, rho_frac * h
        xy = np.asarray(Polygon([(-a, -b), (a, -b), (a, b), (-a, b)]).buffer(rho, 32).exterior.coords)[:-1]
    c, s_ = np.cos(roll), np.sin(roll)
    return xy @ np.array([[c, s_], [-s_, c]])


def slice_section(m, node, e, s_):
    u, v = nj._frame(e)
    Vl = (np.asarray(m.vertices) - node) @ np.vstack((u, v, e)).T
    man = m3.Manifold(m3.Mesh(np.ascontiguousarray(Vl, np.float32), np.ascontiguousarray(m.faces, np.uint32)))
    from matplotlib.path import Path
    for comp in man.slice(float(s_)).decompose():
        polys = comp.to_polygons()
        if sum(Path(np.asarray(p)).contains_point((0.0, 0.0)) for p in polys) % 2:
            return max((np.asarray(p) for p in polys), key=lambda p: Polygon(p).area)
    return None


