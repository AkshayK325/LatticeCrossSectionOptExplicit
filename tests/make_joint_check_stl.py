"""Write the joint-geometry comparison STLs to STL/joint_check/ and print the comparison table.

Per cell, three joints:
  <cell>_hull_ext025.stl        convex hull of the strut end caps, extension -0.025 (the published optBeam setting)
  <cell>_abaqus_cylsphere.stl   cylinders + max-radius node spheres (the validated Abaqus solid models)
  <cell>_smooth.stl             smoothJoint: tapered flares + rolling-ball fillets (smooth_joints=True)
Run: PYTHONPATH=src:abaqus python tests/make_joint_check_stl.py [cell ...]
"""
import contextlib
import io
import os
import sys
import time

import manifold3d as m3
import numpy as np
import trimesh

import cases
import smoothJoint as sj
from beamBuild import BeamStructureSTL
from latticeSetup import example3

OUT = 'STL/joint_check'
CELLS = {  # tag: (topology, elemSize, radii spec, nx)
    'C4_r080': ('C4', 1.0, dict(r=0.8), 1),
    'C12_r050': ('C12', 0.5, dict(r=0.5), 1),
    'BCC_r100': ('B', 1.0, dict(r=1.0), 1),
    'octet_r100': ('O', 1.0, dict(r=1.0), 1),
    'C4_graded_top100_bot060': ('C4', 1.0, dict(r=0.8, rg=(1.0, 0.6, 0.8)), 1),
    'C4_random_radii_040_110': ('C4', 1.0, 'random', 1),
    'C4_graded_top100_bot060_3x3': ('C4', 1.0, dict(r=0.8, rg=(1.0, 0.6, 0.8)), 3),
}


def circle(r, n=64):
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return np.stack([r * np.cos(t), r * np.sin(t)], 1).ravel()


def cylsphere(X, C, rs, H):
    """Abaqus solid geometry: cylinders + a sphere of the largest radius at every node, trimmed to the plates."""
    rn = np.zeros(len(X))
    for (a, b), r in zip(C, rs):
        rn[a], rn[b] = max(rn[a], r), max(rn[b], r)
    parts = []
    for (a, b), r in zip(C, rs):
        d = X[b] - X[a]
        L = np.linalg.norm(d)
        u = d / L
        e = np.cross([0, 0, 1.0], u)
        s, c = np.linalg.norm(e), u[2]
        if s < 1e-12:
            Rm = np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
        else:
            k = e / s
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rm = np.eye(3) + s * K + (1 - c) * K @ K
        parts.append(m3.Manifold.cylinder(L, float(r), float(r), 64).transform(np.hstack((Rm, X[a][:, None])).tolist()))
    for p, r in zip(X, rn):
        parts.append(m3.Manifold.sphere(float(r), 64).translate(tuple(map(float, p))))
    u_ = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
    u_ = u_ ^ m3.Manifold.cube((400, 400, H)).translate((-200 + X[:, 0].mean(), -200 + X[:, 1].mean(), 0.0))
    mm = u_.to_mesh64()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def max_dihedral(m, H):
    ang = np.degrees(m.face_adjacency_angles)
    z = m.vertices[m.face_adjacency_edges][:, :, 2]
    a = ang[np.minimum(np.abs(z), np.abs(z - H)).min(1) > 1e-3]
    return a.max(), np.quantile(a, 0.999)


def main(only):
    os.makedirs(OUT, exist_ok=True)
    print('| cell | joint | faces | volume mm3 | max dihedral | 99.9 % | watertight | time s |')
    print('|---|---|---|---|---|---|---|---|')
    for tag, (cell, es, spec, nx) in CELLS.items():
        if only and tag not in only:
            continue
        lat = example3(elemSize=es, nx=nx, ny=nx, name=cell)
        X, C = lat['X'], np.asarray(lat['conn'])
        H = X[:, 2].max()
        rs = np.random.default_rng(0).uniform(0.4, 1.1, len(C)) if spec == 'random' else cases.strut_radii(spec, lat)
        builds = {}
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            BeamStructureSTL().build_structure(X, C, [circle(r) for r in rs], [np.array([0, 0, 1.0])] * len(C),
                                               [[200, 200, 200, 255]] * len(C), stl_filename=f'{OUT}/{tag}_hull_ext025.stl',
                                               extension=-0.025, z_range=(0.0, H))
        builds['hull_ext025'] = time.time() - t
        t = time.time()
        cylsphere(X, C, rs, H).export(f'{OUT}/{tag}_abaqus_cylsphere.stl')
        builds['abaqus_cylsphere'] = time.time() - t
        t = time.time()
        sj.build_smooth_stl(X, C, rs, f'{OUT}/{tag}_smooth.stl', z_range=(0.0, H))
        builds['smooth'] = time.time() - t
        for meth, dt in builds.items():
            m = trimesh.load(f'{OUT}/{tag}_{meth}.stl', process=True)
            mx, q = max_dihedral(m, H)
            print(f'| {tag} | {meth} | {len(m.faces) / 1e3:.1f}k | {m.volume:.2f} | {mx:.0f} | {q:.0f} | {m.is_watertight} | {dt:.0f} |',
                  flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
