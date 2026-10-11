"""Case 3a robustness sweep (notes/17 P6): FHS single nodes with non-circular convex sections, graded equivalent
radii and random rolls. SF cannot take non-circular sections (circle-only construction) and our Savio node builds its
own square frustums, so FHS is the only method here. Rows: STL/soap_vs_fhs/case3.rows.md.
Run: PYTHONPATH=src:tests python tests/study_case3.py"""
import os
import sys
import time

import numpy as np
import trimesh
from shapely.geometry import LineString, Polygon

import nodeJoint as nj
import runStatus as rs
import study_soap_vs_fhs as st
import study_three_beam_joint as tb
from case3_shapes import ellipse, rrect, slice_section

FAMILIES = ('ellipse 2:1', 'ellipse 3:1', 'rrect 2:1 corner .25', 'near-square corner .1', 'mixed')


def section(family, req, roll, rng):
    if family == 'mixed':
        family = FAMILIES[rng.integers(4)]
    if family.startswith('ellipse'):
        return ellipse(req, float(family.split()[1].split(':')[0]), roll)
    if family.startswith('rrect'):
        return rrect(req, 2.0, 0.25, roll)
    return rrect(req, 1.0, 0.1, roll)


def run_node(name, E, secs):
    E = E / np.linalg.norm(E, axis=1)[:, None]
    req = np.array([np.sqrt(Polygon(s).area / np.pi) for s in secs])
    rout = np.array([np.linalg.norm(s, axis=1).max() for s in secs])
    L = np.full(len(E), max(8.0, 6 * rout.max()))
    node = np.zeros(3)
    t0 = time.time()
    R_s, _ = nj.trim_radius(E, rout, L, dt=0.005)
    sm, _ = nj.hull_subdivision_node(node, E, req, L, R_s, lead=1.0, core=True, sections=secs)
    t_ring = nj.ring_offsets(E, rout, R_s, core=True)
    fm, _ = nj.fair_node_anchored(nj.to_trimesh(sm), node, E, req, t_ring, sections=secs)
    tt = time.time() - t0
    f = f'{st.OUT}/case3_{name}.stl'
    fm.export(f)
    m = trimesh.load(f, process=True)
    os.remove(f)
    ang = np.degrees(m.face_adjacency_angles)
    far = (((m.vertices - node) @ E.T) > L - 0.05).any(1)
    dih = ang[~far[m.face_adjacency_edges].any(1)].max()
    inb = np.linalg.norm(m.vertices, axis=1) < (t_ring + req).max()
    conc = st.concave_radius(m, inb) / req.min()
    hs, secmin = [], []
    tb.NODE = node
    for i, (e, s, rq) in enumerate(zip(E, secs, req)):
        P = slice_section(m, node, e, 0.5 * (t_ring[i] + rq + L[i]))
        hs.append(LineString(np.vstack((P, P[:1]))).hausdorff_distance(LineString(np.vstack((s, s[:1])))) / rq)
        _, R_ = tb.section_profile(m, e, s_max=L[i] - 0.3, ds=0.05)
        secmin.append(np.nanmin(R_) / rq)
    return dict(ok=bool(m.is_watertight and m.is_volume), dih=dih, conc=conc, h=max(hs), sec=min(secmin), t=tt,
                faces=len(m.faces))


def main():
    rs.start('case 3a sweep (FHS, non-circular sections)')
    print('| node | family | valid | max dihedral deg | tightest concave / r_eq,min | max strut-middle Hausdorff / r_eq | '
          'min section / r_eq | build s | faces | good |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    todo = [(n, th, fam) for n in (3, 4, 6) for th in (50, 70, 90) for fam in FAMILIES]
    for i, (n, th, fam) in enumerate(todo):
        nm = f'c3_v{n}_t{th}_{fam.split()[0]}{"" if fam == "mixed" else fam.split()[1].replace(":", "")}'
        rs.update(nm, i, len(todo))
        try:
            E, _ = st.case2_node(n, th, 1.0)
        except ValueError as ex:
            print(f'| {nm} | {fam} | SKIP: {ex} | | | | | | | |', flush=True)
            continue
        rng = np.random.default_rng(i)
        req = np.r_[1.0, 0.5, rng.uniform(0.5, 1.0, n - 2)]          # graded, worst case on the acute pair
        secs = [section(fam, rq, rng.uniform(0, np.pi), rng) for rq in req]
        try:
            o = run_node(nm, E, secs)
            good = o['ok'] and o['sec'] >= 0.95 and o['dih'] < 90 and o['h'] <= 0.01
            print(f"| {nm} | {fam} | {o['ok']} | {o['dih']:.0f} | {o['conc']:.3f} | {100 * o['h']:.2f} % | {o['sec']:.3f} | "
                  f"{o['t']:.1f} | {o['faces']} | {'GOOD' if good else 'FAIL'} |", flush=True)
        except Exception as ex:
            print(f'| {nm} | {fam} | FAIL: {type(ex).__name__}: {str(ex)[:70]} | | | | | | | FAIL |', flush=True)


if __name__ == '__main__':
    main()
