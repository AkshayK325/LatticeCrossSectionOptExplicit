"""Three-beam joint study (notes/12_ThreeBeamJointStudy.md): build each method's STL in STL/three_beam/ and report
the metrics. Run: PYTHONPATH=src python tests/study_three_beam_joint.py [method ...]
"""
import contextlib
import io
import os
import sys

import numpy as np
import trimesh

import nodeJoint as nj

OUT = 'STL/three_beam'
NODE, DIRS, RADII, LEN = nj.three_beam_case()
NAMES = ('B', 'M', 'S')


def circle(r, n=64):
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return np.stack([r * np.cos(t), r * np.sin(t)], 1).ravel()


def flat_ends_mask(m, tol=1e-4):
    """Vertices on the flat far-end faces (excluded from the node metrics)."""
    on = np.zeros(len(m.vertices), bool)
    for e, l in zip(DIRS, LEN):
        on |= np.abs((m.vertices - NODE) @ e - l) < tol
    return on


def section_profile(m, e, s_max=4.0, ds=0.02):
    """Equivalent radius sqrt(A/pi) of the section through the beam axis e (component containing the axis), on
    planes normal to e at distance s from the node."""
    import manifold3d as m3
    from matplotlib.path import Path
    u, v = nj._frame(e)
    Vl = (np.asarray(m.vertices) - NODE) @ np.vstack((u, v, e)).T
    man = m3.Manifold(m3.Mesh(np.ascontiguousarray(Vl, np.float32), np.ascontiguousarray(m.faces, np.uint32)))
    S = np.arange(ds, s_max, ds)
    R = np.full(len(S), np.nan)
    for i, s_ in enumerate(S):
        for comp in man.slice(float(s_)).decompose():
            if sum(Path(np.asarray(p)).contains_point((0.0, 0.0)) for p in comp.to_polygons()) % 2:
                R[i] = np.sqrt(comp.area() / np.pi)
                break
    return S, R


def metrics(m):
    """Watertight, volume, max / 99.9 % dihedral and the tightest concave radius (edge-based normal curvature
    k = -2 n.d / |d|^2 at each vertex, most negative over its edges), excluding the far-end flat faces."""
    flat = flat_ends_mask(m)
    ang = np.degrees(m.face_adjacency_angles)
    ok = ~flat[m.face_adjacency_edges].any(1)
    a = ang[ok]
    n = m.vertex_normals
    E = m.edges_unique
    d = m.vertices[E[:, 1]] - m.vertices[E[:, 0]]
    L2 = (d * d).sum(1)
    k0 = -2 * (n[E[:, 0]] * d).sum(1) / L2                     # normal curvature seen from vertex 0
    k1 = 2 * (n[E[:, 1]] * d).sum(1) / L2
    kmin = np.full(len(m.vertices), np.inf)
    np.minimum.at(kmin, E[:, 0], k0)
    np.minimum.at(kmin, E[:, 1], k1)
    keep = ~flat
    for e, l in zip(DIRS, LEN):                                 # also skip the rim of the far ends
        keep &= np.abs((m.vertices - NODE) @ e - l) > 0.05
    conc = -kmin[keep]
    rmin = 1.0 / conc.max() if (conc > 0).any() else np.inf
    return dict(watertight=bool(m.is_watertight), volume=float(m.volume), max_dihedral=float(a.max()),
                q999=float(np.quantile(a, 0.999)), concave_radius=float(rmin), faces=len(m.faces))


def export(man_or_mesh, name):
    m = nj.to_trimesh(man_or_mesh) if not isinstance(man_or_mesh, trimesh.Trimesh) else man_or_mesh
    f = f'{OUT}/{name}.stl'
    m.export(f)
    return trimesh.load(f, process=True)


def method_hull_ext200():
    """A: beamBuild convex hull, beams shortened by 20 % at the node (published optBeam style, ext -0.2)."""
    from beamBuild import BeamStructureSTL
    X = np.vstack((NODE, NODE + LEN[:, None] * DIRS))
    C = np.array([[0, 1], [0, 2], [0, 3]])
    f = f'{OUT}/A_hull_ext200.stl'
    with contextlib.redirect_stdout(io.StringIO()):
        BeamStructureSTL().build_structure(X, C, [circle(r) for r in RADII], [np.array([0, 0, 1.0])] * 3,
                                           [[200, 200, 200, 255]] * 3, stl_filename=f, extension=-0.2)
    return trimesh.load(f, process=True)


def method_cylsphere():
    """Reference: Abaqus solid model joint (cylinders + sphere of the largest radius at the node)."""
    import manifold3d as m3
    parts = [nj.cylinder(NODE, e, r, l) for e, r, l in zip(DIRS, RADII, LEN)]
    parts.append(m3.Manifold.sphere(float(RADII.max()), 128).translate(tuple(map(float, NODE))))
    return export(m3.Manifold.batch_boolean(parts, m3.OpType.Add), 'R_cylsphere')


def method_sphere_trim():
    """B: trim sphere: beams with everything inside the smallest sphere outside which no beam enters another."""
    R_s, pair = nj.trim_radius(DIRS, RADII, LEN)
    print('  trim radius R_s = %.3f mm; per pair: %s' % (R_s, {f'{NAMES[i]}-{NAMES[j]}': round(v, 3)
                                                             for (i, j), v in pair.items()}))
    return export(nj.trimmed_beams(NODE, DIRS, RADII, LEN, R_s), 'B_sphere_trim')


def _beam_radii_check(m):
    for nm, e, r in zip(NAMES, DIRS, RADII):                          # beams converge to their radius?
        w = (m.vertices - NODE) @ e
        sel = (w > 3.0) & (w < 4.0)
        rad = np.linalg.norm((m.vertices[sel] - NODE) - np.outer(w[sel], e), axis=1)
        print(f'  beam {nm}: radius at mid-length {rad.min():.4f}-{rad.max():.4f} (target {r:.2f})')


def method_hull_subdivision():
    """C1: hull of the node rings (+ each beam's core ring at the node, so no beam necks) and control tubes, Loop
    subdivision (3 levels, first control gaps r long), flat ends."""
    R_s, _ = nj.trim_radius(DIRS, RADII, LEN)
    sm, _ = nj.hull_subdivision_node(NODE, DIRS, RADII, LEN, R_s, lead=1.0, core=True)
    m = export(sm, 'C1_hull_subdivision')
    _beam_radii_check(m)
    return m


def method_fair():
    """C3: C1 + variational (cotangent bi-Laplacian) fairing of the node region, free 0.5 r beyond each ring."""
    R_s, _ = nj.trim_radius(DIRS, RADII, LEN)
    sm, _ = nj.hull_subdivision_node(NODE, DIRS, RADII, LEN, R_s, lead=1.0, core=True)
    fm, _ = nj.fair_node(nj.to_trimesh(sm), NODE, DIRS, RADII, nj.ring_offsets(DIRS, RADII, R_s, core=True),
                         free_len=0.5, core=False)
    return export(fm, 'C3_subdivision_faired')


def method_sdf():
    """C2: smoothJoint (SDF) on the same input: rounded convex hull of the caps at the same ring offsets, struts from
    their caps, rolling-ball fillets; far ends trimmed flat."""
    import manifold3d as m3
    import smoothJoint as sj
    R_s, _ = nj.trim_radius(DIRS, RADII, LEN)
    t_ring = nj.ring_offsets(DIRS, RADII, R_s, core=True)
    lookup = {round(float(r), 6): float(t) for r, t in zip(RADII, t_ring)}
    keep = sj._hull_extent
    sj._hull_extent = lambda R, r, L: lookup[round(float(r), 6)]
    try:
        X = np.vstack((NODE, NODE + (LEN[:, None] + 1.0) * DIRS))
        m = sj.build_smooth_stl(X, np.array([[0, 1], [0, 2], [0, 3]]), RADII)
    finally:
        sj._hull_extent = keep
    man = m3.Manifold(m3.Mesh(np.ascontiguousarray(m.vertices, np.float32), np.ascontiguousarray(m.faces, np.uint32)))
    for e, l in zip(DIRS, LEN):
        man = man.trim_by_plane(tuple(map(float, -e)), float(-(NODE @ e) - l))
    return export(man, 'C2_sdf_hull_rolling_ball')


METHODS = {'A_hull_ext200': method_hull_ext200, 'R_cylsphere': method_cylsphere, 'B_sphere_trim': method_sphere_trim,
           'C1_hull_subdivision': method_hull_subdivision, 'C2_sdf_hull_rolling_ball': method_sdf,
           'C3_subdivision_faired': method_fair}


def main(only):
    os.makedirs(OUT, exist_ok=True)
    print('| method | watertight | faces | volume mm3 | max dihedral | 99.9 % | tightest concave radius mm | '
          'min section / r (B, M, S) |')
    print('|---|---|---|---|---|---|---|---|')
    for name, fn in METHODS.items():
        if only and name not in only:
            continue
        m = fn()
        r = metrics(m)
        neck = ', '.join(f'{np.nanmin(section_profile(m, e)[1]) / rr:.3f}' for e, rr in zip(DIRS, RADII))
        print(f"| {name} | {r['watertight']} | {r['faces']} | {r['volume']:.3f} | {r['max_dihedral']:.0f} | "
              f"{r['q999']:.0f} | {r['concave_radius']:.3f} | {neck} |", flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
