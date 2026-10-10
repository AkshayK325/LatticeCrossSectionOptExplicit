"""Figure for notes/16: SF (soap film, Luo & Zou 2025; paper-calibrated fixed band) vs FHS nodes. Rows: grey shading
(as the paper's Fig. 12), and mean curvature H r_min (red = concave; clipped at +-1.5, so any fillet tighter than
~0.67 r shows red). Run after tests/study_soap_vs_fhs.py s3 / s4 / s5 / sfcal (reads their STLs).
Output: ResultsFig/explicit/70_soap_film_vs_fhs.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh

import study_soap_vs_fhs as st

CASES = [('regular-6', 'regular-6', 1.0, 2.2), ('regular-20', 'regular-20', 1.0, 4.4),
         ('irregular-12', 'irregular-12', 1.0, 3.6),
         ('tetra_r080', 'tetra r 0.8 (FE node)', 0.8, 1.9),
         ('base_107_117_87_r110_075_040', '3-beam 1.1 / 0.75 / 0.4', 0.4, 2.6)]


def stl(stem, meth):
    if stem.startswith('tetra'):
        return f'{st.OUT}/{stem}_{"SF_cal" if meth == "SF" else meth}_fe.stl'
    return f'{st.OUT}/{stem}_cal_SF.stl' if meth == 'SF' else f'{st.OUT}/{stem}_{meth}.stl'


def main():
    fig = plt.figure(figsize=(16, 13))
    n = len(CASES)
    for k, (stem, title, rmin, R) in enumerate(CASES):
        for j, meth in enumerate(('SF', 'FHS')):
            m = trimesh.load(stl(stem, meth), process=True)
            H = st.mean_curvature(m)
            keep = np.linalg.norm(m.vertices[m.faces].mean(1), axis=1) < R
            sub = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
            Hs = H
            for row, grey in ((j, True), (2 + j, False)):
                ax = fig.add_subplot(4, n, 1 + k + n * row)
                st.render_z(ax, sub, Hs, rmin, f'{meth}{" (H r)" if not grey else ""}: {title}', lim=1.05 * R,
                            n_px=380, n_pts=1_800_000, grey=grey)
    sm = plt.cm.ScalarMappable(cmap='jet_r', norm=plt.Normalize(-1.5, 1.5))
    cb = fig.colorbar(sm, ax=fig.axes[2 * n:], shrink=0.6, pad=0.02)
    cb.set_label('mean curvature H r_min (red = concave)')
    fig.savefig('ResultsFig/explicit/70_soap_film_vs_fhs.png', dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    main()
