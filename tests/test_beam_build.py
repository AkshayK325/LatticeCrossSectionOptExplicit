"""beamBuild STL export: watertight closed solids for any radius, section shape, winding and lattice topology."""
import numpy as np
import pytest

from beamBuild import BeamStructureSTL
from latticeSetup import example3


def _circle(r, n=32, cw=False):
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    P = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return (P[::-1] if cw else P).ravel()


def _shapes():
    t = np.linspace(0, 2 * np.pi, 41)[:-1]
    ellipse = np.stack([0.9 * np.cos(t), 0.45 * np.sin(t)], 1).ravel()
    rect = np.array([-0.6, -0.3, 0.6, -0.3, 0.6, 0.3, -0.6, 0.3])
    rr = 0.4 + 0.35 * np.cos(5 * t)                                   # non-convex star
    star = np.stack([rr * np.cos(t), rr * np.sin(t)], 1).ravel()
    padded = np.concatenate([_circle(0.5, 24), _circle(0.5, 24)[-2:], _circle(0.5, 24)[:2]])   # duplicate points
    return [ellipse, rect, star, _circle(0.5, cw=True), padded]


def _build(tmp_path, lat, sections, z_range=None):
    """Build, then return the STL AS RE-READ (vertices merged by position, as a slicer sees it)."""
    import trimesh
    X, C = lat['X'], lat['conn']
    f = str(tmp_path / 's.stl')
    BeamStructureSTL().build_structure(X, C, sections, [np.array([0.0, 0.0, 1.0])] * len(C),
                                       [[200, 200, 200, 255]] * len(C), stl_filename=f, z_range=z_range)
    m = trimesh.load(f, process=True)
    edges_per_edge = np.bincount(trimesh.grouping.group_rows(m.edges_sorted, require_count=2).ravel(), minlength=0)
    assert m.is_winding_consistent
    return m


@pytest.mark.parametrize('cell', ['C4', 'C12', 'B', 'O'])
def test_watertight_topologies(tmp_path, cell):
    lat = example3(elemSize=1.0, nx=1, ny=1, name=cell)
    m = _build(tmp_path, lat, [_circle(0.6)] * len(lat['conn']))
    assert m.is_watertight and m.is_volume and m.volume > 0


def test_graded_radii_and_mixed_shapes(tmp_path):
    lat = example3(elemSize=1.0, nx=1, ny=1)
    nb = len(lat['conn'])
    rng = np.random.default_rng(0)
    secs = [_circle(r) for r in rng.uniform(0.4, 1.1, nb)]
    m = _build(tmp_path, lat, secs)
    assert m.is_watertight and m.is_volume
    shapes = _shapes()
    m2 = _build(tmp_path, lat, [shapes[i % len(shapes)] for i in range(nb)])
    assert m2.is_watertight and m2.is_volume


def test_trim_to_plates_and_volume_vs_exact(tmp_path):
    """Circular struts with hull joints, trimmed to the plate planes: flat faces at z = 0 and H, and the volume
    matches the exact cylinder + sphere union to within the joint-shape difference (hull inside sphere)."""
    lat = example3(elemSize=1.0, nx=1, ny=1)
    X = lat['X']
    H = X[:, 2].max()
    m = _build(tmp_path, lat, [_circle(0.8, 64)] * len(lat['conn']), z_range=(0.0, H))
    assert m.is_watertight
    assert m.bounds[0][2] == pytest.approx(0.0, abs=1e-6) and m.bounds[1][2] == pytest.approx(H, abs=1e-6)
    assert m.volume == pytest.approx(286.1, rel=0.01)                 # OCC cylinders + spheres, trimmed: 286.11


@pytest.mark.parametrize('ext,smooth', [(-0.1, False), (-0.2, False), (-0.1, 'hull'), (-0.2, 'hull')])
def test_shortened_struts_and_rounded_joints(tmp_path, ext, smooth):
    """The print settings of optBeam (extension -0.1 / -0.2: struts shortened at the joints, hull spans the gap),
    with and without the rounded hulls (smooth_joints='hull'), on C4 and BCC: still a closed 2-manifold when re-read."""
    import trimesh
    for cell in ('C4', 'B'):
        lat = example3(elemSize=1.0, nx=1, ny=1, name=cell)
        X, C = lat['X'], lat['conn']
        f = str(tmp_path / 's.stl')
        BeamStructureSTL().build_structure(X, C, [_circle(0.6)] * len(C), [np.array([0.0, 0.0, 1.0])] * len(C),
                                           [[200, 200, 200, 255]] * len(C), stl_filename=f, extension=ext,
                                           smooth_joints=smooth, z_range=(0.0, X[:, 2].max()))
        m = trimesh.load(f, process=True)
        assert m.is_watertight and m.is_volume, (cell, ext, smooth)


def _interior_dihedral(m, H):
    """Dihedral angles (deg) between adjacent faces, excluding edges on the plate planes z = 0, H."""
    ang = np.degrees(m.face_adjacency_angles)
    z = m.vertices[m.face_adjacency_edges][:, :, 2]
    return ang[np.minimum(np.abs(z), np.abs(z - H)).min(1) > 1e-3]


@pytest.mark.parametrize('cell', ['C4', 'B'])
def test_filleted_joints_smooth_and_watertight(tmp_path, cell):
    """smooth_joints=True (smoothJoint: round-union fillets, fine mesh at the joints, stitched tubes): closed when
    re-read, no crease anywhere (the cylinder + sphere / hull joints have 90-150 deg edges), and the volume is the
    plain strut union plus < 0.5 % (rolling-ball fillets fill only the concave crevices)."""
    import trimesh
    lat = example3(elemSize=1.0, nx=1, ny=1, name=cell)
    X, C = lat['X'], lat['conn']
    H = X[:, 2].max()
    f = str(tmp_path / 's.stl')
    BeamStructureSTL().build_structure(X, C, [_circle(0.8)] * len(C), [np.array([0.0, 0.0, 1.0])] * len(C),
                                       [[200, 200, 200, 255]] * len(C), stl_filename=f, smooth_joints=True,
                                       z_range=(0.0, H))
    m = trimesh.load(f, process=True)
    assert m.is_watertight and m.is_volume
    assert _interior_dihedral(m, H).max() < 35.0
    assert m.bounds[0][2] == pytest.approx(0.0, abs=1e-6) and m.bounds[1][2] == pytest.approx(H, abs=1e-6)
    if cell == 'C4':
        assert m.volume == pytest.approx(287.4, rel=0.005)  # 32-gon -> same-area circle r 0.797; plain union 286.2


def test_filleted_joints_mixed_radii_and_polygons(tmp_path):
    """Random radii 0.4-1.1 (thin struts into thick ones: the fillet is sized to the thinner strut, no step) and
    polygon sections of every kind (ellipse, rectangle, non-convex star, clockwise, duplicate points)."""
    import trimesh
    lat = example3(elemSize=1.0, nx=1, ny=1)
    X, C = lat['X'], lat['conn']
    H = X[:, 2].max()
    nb = len(C)
    v = [np.array([0.0, 0.0, 1.0])] * nb
    col = [[200, 200, 200, 255]] * nb
    f = str(tmp_path / 's.stl')
    radii = np.random.default_rng(0).uniform(0.4, 1.1, nb)
    BeamStructureSTL().build_structure(X, C, [_circle(r) for r in radii], v, col, stl_filename=f, smooth_joints=True,
                                       z_range=(0.0, H))
    m = trimesh.load(f, process=True)
    assert m.is_watertight and m.is_volume
    assert _interior_dihedral(m, H).max() < 35.0
    shapes = _shapes()
    BeamStructureSTL().build_structure(X, C, [shapes[i % len(shapes)] for i in range(nb)], v, col, stl_filename=f,
                                       smooth_joints=True, z_range=(0.0, H))
    m2 = trimesh.load(f, process=True)
    assert m2.is_watertight and m2.is_volume


def test_smooth_adaptive_matches_uniform():
    """The adaptive build (joint patches + stitched tubes) is the same surface as one uniform marching-cubes grid."""
    from smoothJoint import build_smooth_stl
    lat = example3(elemSize=1.0, nx=1, ny=1)
    X, C = lat['X'], lat['conn']
    a = build_smooth_stl(X, C, 0.8, adaptive=True)
    u = build_smooth_stl(X, C, 0.8, adaptive=False)
    assert a.is_watertight and u.is_watertight
    assert a.volume == pytest.approx(u.volume, rel=2e-3)


def test_rolling_ball_adds_material_only_where_concave():
    """Two struts merging at one node: collinear (no crevice) gets no added volume, and the fill grows with the
    concavity of the joint (150 < 90 < 45 deg between the struts). The old round union added ~2 % in every case."""
    from smoothJoint import build_smooth_stl
    def two(between):
        t = np.radians(between)
        X = np.array([[-6.0, 0, 0], [0, 0, 0], [-6 * np.cos(t), 6 * np.sin(t), 0.0]])
        return X, np.array([[0, 1], [1, 2]])
    added = {}
    for between in (180, 150, 90, 45):
        X, C = two(between)
        plain = build_smooth_stl(X, C, 0.8, fillet=1e-4, adaptive=False).volume
        added[between] = build_smooth_stl(X, C, 0.8, fillet=0.5, adaptive=False).volume / plain - 1
    assert abs(added[180]) < 2e-4
    assert added[150] < added[90] < added[45] < 0.015


def test_flare_tapers_big_to_small():
    """A 1.0 mm strut continuing into a 0.5 mm one: without a flare the thin strut leaves the 1.0 mm end sphere with a
    step (section radius slope ~1.7 at the root); with the flare the radius falls along a tangent 20 deg cone into a
    rolling-ball fillet, so the slope stays near tan(20 deg) = 0.36 everywhere. Same both ways round."""
    import manifold3d as m3
    from smoothJoint import build_smooth_stl
    for order in ((1.0, 0.5), (0.5, 1.0)):
        X = np.array([[-6.0, 0, 0], [0, 0, 0], [6.0, 0, 0]])
        m = build_smooth_stl(X, np.array([[0, 1], [1, 2]]), np.array(order), adaptive=False)
        assert m.is_watertight
        V = np.asarray(m.vertices)[:, [1, 2, 0]]                     # strut axis -> z
        man = m3.Manifold(m3.Mesh(np.ascontiguousarray(V, np.float32), np.ascontiguousarray(m.faces, np.uint32)))
        z = np.linspace(-3.0, 3.0, 301)
        r = np.array([np.sqrt(man.slice(float(t)).area() / np.pi) for t in z])
        assert np.abs(np.gradient(r, z)).max() < 0.5
        assert r.max() < 1.01 and r.min() > 0.49
