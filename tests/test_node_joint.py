"""FHS (Faired Hull-Subdivision) node joints (notes/13_FHS_NodeAlgorithm.md): watertight, crease-free, no necking."""
import numpy as np
import trimesh

import nodeJoint as nj
from latticeSetup import example3


def _dihedral_max(m, H=None):
    ang = np.degrees(m.face_adjacency_angles)
    if H is not None:
        z = m.vertices[m.face_adjacency_edges][:, :, 2]
        ang = ang[np.minimum(np.abs(z), np.abs(z - H)).min(1) > 1e-3]
    return ang.max()


def test_three_beam_fhs_no_neck(tmp_path):
    """Big / medium / small beam in 3D: watertight, smooth, and each beam keeps its section (the core rings and the
    flank anchors stop the thin-plate fairing from necking it)."""
    node, E, r, L = nj.three_beam_case()
    R_s, _ = nj.trim_radius(E, r, L)
    sm, _ = nj.hull_subdivision_node(node, E, r, L, R_s, lead=1.0, core=True)
    fm, _ = nj.fair_node_anchored(nj.to_trimesh(sm), node, E, r, nj.ring_offsets(E, r, R_s, core=True))
    f = str(tmp_path / 'n.stl')
    fm.export(f)
    m = trimesh.load(f, process=True)
    assert m.is_watertight and m.is_volume
    ang = np.degrees(m.face_adjacency_angles)
    far = (((m.vertices - node) @ E.T) > L - 0.05).any(1)        # the flat far-end faces have 90 deg rims
    assert ang[~far[m.face_adjacency_edges].any(1)].max() < 20
    import manifold3d as m3
    for e, rr in zip(E, r):                                   # section along each beam into the node: no neck
        u, v = nj._frame(e)
        Vl = (m.vertices - node) @ np.vstack((u, v, e)).T
        man = m3.Manifold(m3.Mesh(np.ascontiguousarray(Vl, np.float32), np.ascontiguousarray(m.faces, np.uint32)))
        from matplotlib.path import Path
        for s in np.arange(0.2, 4.0, 0.05):                    # the section component that contains the axis
            own = [c for c in man.slice(float(s)).decompose()
                   if sum(Path(np.asarray(p)).contains_point((0.0, 0.0)) for p in c.to_polygons()) % 2]
            assert own and np.sqrt(own[0].area() / np.pi) > 0.98 * rr


def test_lattice_fhs_watertight_smooth(tmp_path):
    lat = example3(elemSize=1.0, nx=1, ny=1, name='C4')
    X, C = lat['X'], np.asarray(lat['conn'])
    H = X[:, 2].max()
    man, _ = nj.lattice_fhs(X, C, 0.8, z_range=(0.0, H))
    f = str(tmp_path / 'c.stl')
    nj.to_stl_mesh(man).export(f)
    m = trimesh.load(f, process=True)
    assert m.is_watertight and m.is_volume
    assert _dihedral_max(m, H) < 20
    assert abs(m.volume - 283.4) < 0.01 * 283.4
    assert len(m.faces) < 150_000                                # 2 levels + limit + coarse middles (was 558k)
