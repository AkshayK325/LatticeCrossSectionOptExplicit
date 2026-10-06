"""C3 (hull + core rings + Loop subdivision + bi-Laplacian fairing) on three-beam nodes with different angles and
radii (notes/12_ThreeBeamJointStudy.md). STLs: STL/three_beam/sweep/. Run: PYTHONPATH=src python tests/study_three_beam_sweep.py
"""
import os
import time

import numpy as np
import trimesh

import nodeJoint as nj
import study_three_beam_joint as st

OUT = 'STL/three_beam/sweep'
L = np.full(3, 6.0)


def unit(*v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def from_angles(a12, a13, a23):
    """Three unit directions with the given pairwise angles (deg); the third leaves the e1-e2 plane (z > 0)."""
    a12, a13, a23 = np.radians([a12, a13, a23])
    e1 = np.array([1.0, 0, 0])
    e2 = np.array([np.cos(a12), np.sin(a12), 0.0])
    x = np.cos(a13)
    y = (np.cos(a23) - x * np.cos(a12)) / np.sin(a12)
    z2 = 1 - x * x - y * y
    if z2 <= 0:
        raise ValueError('angles not realisable in 3D')
    return np.vstack((e1, e2, [x, y, np.sqrt(z2)]))


ANGLES = {   # name: pairwise angles B-M, B-S, M-S (deg)
    'base_107_117_87': (107, 117, 87),
    'acute_50_120_110': (50, 120, 110),
    'wideY_118_118_118': (118, 118, 118),       # tilted: sum 354 < 360, so 3D (not the planar 120 deg Y)
    'tripod_75_75_75': (75, 75, 75),
}
RADII = {
    'r110_075_040': (1.10, 0.75, 0.40),
    'r080_080_080': (0.80, 0.80, 0.80),
    'r110_110_040': (1.10, 1.10, 0.40),
    'r040_040_110': (0.40, 0.40, 1.10),
    'r100_060_060': (1.00, 0.60, 0.60),
}


FREE, LIMIT, TAG, ANCHOR = 0.5, False, 'C3', 10.0


def c3(node, dirs, radii):
    R_s, _ = nj.trim_radius(dirs, radii, L)
    sm, _ = nj.hull_subdivision_node(node, dirs, radii, L, R_s, lead=1.0, core=True)
    t_ring = nj.ring_offsets(dirs, radii, R_s, core=True)
    if LIMIT == 'anchor':
        fm, _ = nj.fair_node_anchored(nj.to_trimesh(sm), node, dirs, radii, t_ring, free_len=FREE, anchor=ANCHOR)
    else:
        fm, _ = nj.fair_node(nj.to_trimesh(sm), node, dirs, radii, t_ring, free_len=FREE, core=False)
    return fm, R_s


def main():
    os.makedirs(OUT, exist_ok=True)
    print('| angles | radii | R_s | watertight | volume mm3 | max dihedral | 99.9 % | concave radius mm | '
          'min section / r | time s |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    node = np.zeros(3)
    for an, ang in ANGLES.items():
        dirs = from_angles(*ang)
        for rn, rad in RADII.items():
            radii = np.array(rad)
            st.DIRS, st.RADII, st.LEN, st.NODE = dirs, radii, L, node      # metric helpers use these
            t = time.time()
            try:
                fm, R_s = c3(node, dirs, radii)
                f = f'{OUT}/{TAG}_{an}_{rn}.stl'
                fm.export(f)
                m = trimesh.load(f, process=True)
                r = st.metrics(m)
                neck = ', '.join(f'{np.nanmin(st.section_profile(m, e)[1]) / rr:.3f}' for e, rr in zip(dirs, radii))
                print(f"| {an} | {rn} | {R_s:.3f} | {r['watertight']} | {r['volume']:.2f} | {r['max_dihedral']:.0f} | "
                      f"{r['q999']:.0f} | {r['concave_radius']:.3f} | {neck} | {time.time() - t:.0f} |", flush=True)
            except Exception as err:
                print(f'| {an} | {rn} | FAILED: {type(err).__name__}: {err} |', flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:                         # e.g. 1.0 anchor C3A 10
        FREE, LIMIT, TAG = float(sys.argv[1]), sys.argv[2], sys.argv[3]
        ANCHOR = float(sys.argv[4]) if len(sys.argv) > 4 else ANCHOR
    main()
