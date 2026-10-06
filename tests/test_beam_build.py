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
