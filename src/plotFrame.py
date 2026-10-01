"""Deformation GIFs for explicitFE.Frame histories (matplotlib + Pillow, no extra deps)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def draw_tubes(ax, P, conn, radii, colors=None, n_theta=16):
    """Draw every element as a cylinder of its true radius (explicit counterpart of frameFE3
    plotStructure(thicknessPlot=True)), plus a sphere at each element end so the capsule surface is visible."""
    radii = np.asarray(radii)
    th = np.linspace(0, 2 * np.pi, n_theta)
    for e, (a, b) in enumerate(conn):
        p0, p1 = P[a], P[b]
        v = p1 - p0
        L = np.linalg.norm(v)
        v = v / L
        nv = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
        n1 = np.cross(v, nv)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t, T = np.meshgrid(np.linspace(0, L, 2), th)
        r = radii[e]
        X, Y, Z = [p0[i] + v[i] * t + r * np.cos(T) * n1[i] + r * np.sin(T) * n2[i] for i in range(3)]
        c = 'tab:blue' if colors is None else colors[e]
        ax.plot_surface(X, Y, Z, color=c, shade=True, linewidth=0, alpha=0.9)
        for pe in (p0, p1):
            u_, v_ = np.meshgrid(np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 6))
            ax.plot_surface(pe[0] + r * np.cos(u_) * np.sin(v_), pe[1] + r * np.sin(u_) * np.sin(v_),
                            pe[2] + r * np.cos(v_), color=c, shade=True, linewidth=0, alpha=0.9)


def animate_frame(frame, U, path, times=None, max_frames=80, elev=20, azim=-60, title='', extra=None, fps=15,
                  radii=None, colors=None):
    """U: (n_rec, N, 3) displacement history. Writes a GIF of the deformed structure (true scale).
    radii (ne,): draw tubes of the true beam radius (thickness plot) instead of centrelines.
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
        if radii is not None:
            draw_tubes(ax, x, conn, radii, colors)
        else:
            for s in x[conn]:
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


def numbered(folder, name):
    """folder/'<n>_<name>' with n one above the highest existing numeric prefix: larger number = newer."""
    import pathlib
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split('_', 1)[0]) for p in folder.iterdir() if p.name.split('_', 1)[0].isdigit()]
    return folder / f'{max(nums, default=0) + 1}_{name}'
