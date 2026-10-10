"""Soap-film node (Luo & Zou 2025, src/soapFilm.py), gate S1 of notes/16: closed node solid, the patch ends exactly
on the cut circles, and it meets the retained cylinders tangentially (G1: axial tilt of the boundary facets
<= 3 deg at the paper's 3 levels) for valence 4 to 20. Valence 3 (lune Voronoi cells, not in the paper's examples)
is a recorded finding: 18 deg at 3 levels, decaying ~x0.65 per level (G1 only in the limit)."""
import numpy as np
import pytest

import soapFilm as sfm
from study_soap_vs_fhs import regular, irregular, g1_jump, tetra


@pytest.mark.parametrize('name', ['three', 'tetra', 'regular6', 'regular12', 'regular20', 'irregular12'])
def test_soap_film_node(name, tmp_path):
    import trimesh
    E = {'three': np.array([[1.0, 0, 0], [-0.30, 0.80, 0.52], [-0.45, -0.55, 0.70]]), 'tetra': tetra(),
         'regular6': regular(6), 'regular12': regular(12), 'regular20': regular(20),
         'irregular12': irregular(12)}[name]
    E = E / np.linalg.norm(E, axis=1)[:, None]
    r = np.ones(len(E))
    sf = sfm.soap_film_node(np.zeros(3), E, r)
    b = sf['circ'] >= 0
    e = sf['E'][sf['circ'][b]]
    ax = (sf['V'][b] * e).sum(1)
    rad = np.linalg.norm(sf['V'][b] - ax[:, None] * e, axis=1)
    assert np.abs(rad - 1).max() < 1e-9 and np.abs(ax - sf['d'][sf['circ'][b]]).max() < 1e-9
    if len(E) > 3:
        assert g1_jump(sf, axial=True) < 3.0
    else:
        tilt = [g1_jump(sfm.soap_film_node(np.zeros(3), E, r, levels=k), axial=True) for k in (2, 3, 4)]
        assert tilt[0] > tilt[1] > tilt[2] and tilt[2] < 0.75 * tilt[1]
    f = str(tmp_path / 'n.stl')
    sfm.node_solid(sf, np.full(len(E), sf['d'].max() + 3)).export(f)
    m = trimesh.load(f, process=True)
    assert m.is_watertight and m.is_volume
