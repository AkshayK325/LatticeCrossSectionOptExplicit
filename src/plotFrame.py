"""Deformation GIFs for explicitFE.Frame histories (matplotlib + Pillow, no extra deps)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_frame(frame, U, path, times=None, max_frames=80, elev=20, azim=-60, title='', extra=None, fps=15):
    """U: (n_rec, N, 3) displacement history. Writes a GIF of the deformed centrelines (true scale).
    extra(ax, k) may draw more per frame (e.g. a contact plane)."""
    X = frame.X.detach().cpu().numpy()
    conn = frame.conn.cpu().numpy()
    U = np.asarray(U.detach().cpu() if hasattr(U, 'detach') else U)
    idx = np.unique(np.linspace(0, len(U) - 1, min(max_frames, len(U))).astype(int))
    P = X[None] + U[idx]
    lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05 + 1e-9
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    def draw(k):
        ax.clear()
        x = P[k]
        seg = x[conn]
        for s in seg:
            ax.plot(s[:, 0], s[:, 1], s[:, 2], 'b-', lw=1.5)
        ax.plot(X[conn][:, :, 0].T, X[conn][:, :, 1].T, X[conn][:, :, 2].T, color='0.8', lw=0.6)
        if extra is not None:
            extra(ax, idx[k])
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')
        tt = '' if times is None else f'  t = {times[idx[k]]*1e3:.2f} ms'
        ax.set_title(title + tt)

    FuncAnimation(fig, draw, frames=len(idx)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
