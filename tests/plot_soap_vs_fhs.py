"""Figure for notes/16: SF (soap film, Luo & Zou 2025) vs FHS nodes, mean curvature H r on the node region.
Run after tests/study_soap_vs_fhs.py s3 / s4 / s5 (reads their STLs). Output: ResultsFig/explicit/70_soap_film_vs_fhs.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh

import study_soap_vs_fhs as st

CASES = [('regular-6', 'regular-6', 1.0), ('regular-20', 'regular-20', 1.0),
         ('tetra_r080', 'tetra_r080 (FE node)', 0.8),
         ('base_107_117_87_r110_075_040', '3-beam graded 1.1 / 0.75 / 0.4', 0.4)]


def main():
    fig = plt.figure(figsize=(13, 7))
    for k, (stem, title, rmin) in enumerate(CASES):
        for row, meth in enumerate(('SF', 'FHS')):
            f = f'{st.OUT}/{stem}_{meth}{"_fe" if stem.startswith("tetra") else ""}.stl'
            m = trimesh.load(f, process=True)
            H = st.mean_curvature(m)
            R = {'regular-6': 2.2, 'regular-20': 4.0, 'tetra_r080': 1.9}.get(stem, 2.6)
            keep = np.linalg.norm(m.vertices[m.faces].mean(1), axis=1) < R
            sub = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
            ax = fig.add_subplot(2, 4, 1 + k + 4 * row)
            st.render_z(ax, sub, H, rmin, f'{meth}: {title}', lim=1.05 * R)
    sm = plt.cm.ScalarMappable(cmap='jet_r', norm=plt.Normalize(-1.5, 1.5))
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.02)
    cb.set_label('mean curvature H r_min (red = concave)')
    fig.savefig('ResultsFig/explicit/70_soap_film_vs_fhs.png', dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    main()
