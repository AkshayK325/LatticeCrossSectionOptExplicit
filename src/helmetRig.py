"""Helmet crown patch: spherical shell + curved beam-lattice liner + free rigid head, hit by a rigid hemispherical anvil.

Layout (origin = head-sphere centre, +z through the crown, impact from above):
    anvil (fixed rigid sphere R_anv)  |  SHELL (true shell, t, reference surface = lattice side, offset -t/2 toward the
    anvil)  |  beam-lattice liner (flat cell lattice mapped onto the spheres)  |  head (rigid mass, tied to the liner's inner nodes)
The whole assembly (head, liner, shell) starts at v0 toward the anvil, as in the standards (helmeted head dropped onto a fixed anvil).

Lattice map (notes/19 section 8): a flat lattice with x,y in [0,W], z in [0,H] goes to
    p = (R_head + z) n(theta, alpha),  theta = rho / R_head,  rho, alpha = polar coordinates of (x - W/2, y - W/2),
so cell corner nodes lie on concentric spheres, the tangential cell size grows by (R_head + z)/R_head with height, and
beams stay straight between mapped nodes (re-refined with the element counts of the flat lattice, so the per-element
design variables and the joint factors of the flat lattice apply one-to-one).

Shell nodes are the liner's outer-end nodes (node sharing: beam end moments and forces go straight into the shell)
plus the cell-centre points, triangulated by Delaunay in the flat (x, y) plane.
"""
import numpy as np
import torch

from explicitFE import Frame, lumped_mass
from shellFE import Shell


def map_sphere(P, R_head, W):
    """Flat points (n,3) -> sphere-mapped points (head centre at the origin, crown on +z)."""
    x, y, z = P[:, 0] - W / 2, P[:, 1] - W / 2, P[:, 2]
    rho = np.hypot(x, y)
    th = rho / R_head
    al = np.arctan2(y, x)
    r = R_head + z
    return np.stack((r * np.sin(th) * np.cos(al), r * np.sin(th) * np.sin(al), r * np.cos(th)), 1)


def curved_lattice(lat, R_head=80.0):
    """Map the flat lattice `lat` (latticeSetup.example3) onto the head sphere. Returns a lat-like dict (X, conn, Xr, connr,
    beam_of_elem, ... as the flat one) plus the sphere map settings."""
    X, conn = lat['X'], lat['conn']
    W = float(X[:, 0].max())
    Xc = map_sphere(X, R_head, W)
    L = np.linalg.norm(lat['X'][conn[:, 1]] - lat['X'][conn[:, 0]], axis=1)
    n_per = np.rint(L / lat['elemSize']).astype(int)
    pts, ce = [Xc], []
    nn = len(Xc)
    for b in range(len(conn)):
        a, c = Xc[conn[b, 0]], Xc[conn[b, 1]]
        k = n_per[b]
        if k > 1:
            t = np.linspace(0, 1, k + 1)[1:-1]
            pts.append(np.outer(1 - t, a) + np.outer(t, c))
            ids = np.concatenate(([conn[b, 0]], np.arange(nn, nn + k - 1), [conn[b, 1]]))
            nn += k - 1
        else:
            ids = np.array([conn[b, 0], conn[b, 1]])
        ce += [[ids[i], ids[i + 1]] for i in range(len(ids) - 1)]
    out = dict(lat)
    out.update(Xc=Xc, Xr=np.vstack(pts), connr=np.array(ce), R_head=R_head, W=W, flat=lat)
    # the refined flat lattice has exactly the same element count and beam_of_elem
    assert len(out['connr']) == len(lat['connr']) and (n_per == np.bincount(lat['beam_of_elem'])).all()
    return out


def shell_mesh(clat, t, E, nu, rho, extra_centres=True):
    """Shell on the liner's outer face. Returns (Shell, X_all (Nl + n_extra, 3), nodes_in_frame)."""
    from scipy.spatial import Delaunay
    flat = clat['flat']
    Xf, Xrf = flat['Xr'], clat['Xr']
    W, R = clat['W'], clat['R_head']
    zt = Xf[:, 2].max()
    top = np.nonzero(np.isclose(Xf[:, 2], zt))[0]                  # outer-face lattice nodes (flat z = H)
    pts_flat = Xf[top, :2]
    extra = np.zeros((0, 3))
    if extra_centres:
        cell = W / flat['params']['nx']
        g = (np.arange(flat['params']['nx']) + 0.5) * cell
        cx, cy = np.meshgrid(g, g, indexing='ij')
        c = np.stack((cx.ravel(), cy.ravel(), np.full(cx.size, zt)), 1)
        extra = map_sphere(c, R, W)
        pts_flat = np.vstack((pts_flat, c[:, :2]))
    X_sh = np.vstack((Xrf[top], extra))
    tri = Delaunay(pts_flat).simplices
    e1, e2 = pts_flat[tri[:, 1]] - pts_flat[tri[:, 0]], pts_flat[tri[:, 2]] - pts_flat[tri[:, 0]]
    a2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    tri = tri[a2 > 0.05 * np.median(a2)]                            # drop zero-area hull triangles (collinear boundary nodes)
    nodes = np.concatenate((top, len(Xrf) + np.arange(len(extra))))
    D = -X_sh / np.linalg.norm(X_sh, axis=1, keepdims=True)         # director: toward the head (the liner side)
    n_f = np.cross(X_sh[tri[:, 1]] - X_sh[tri[:, 0]], X_sh[tri[:, 2]] - X_sh[tri[:, 0]])
    flip = (n_f * D[tri].mean(1)).sum(1) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    sh = Shell(X_sh, tri, D, t, E, nu, rho, offset=-t / 2, nodes=nodes)
    return sh, X_sh[len(top):], nodes


class HybridFrame(Frame):
    """Beam Frame plus a shell on some of its nodes (extra nodes allowed). `internal_force` = beams + shell."""

    def __init__(self, X, conn, shell, **kw):
        super().__init__(X, conn, **kw)
        self.shell = shell

    def internal_force(self, u, R, law):
        fu, fth, d, f_loc = super().internal_force(u, R, law)
        fs, ft, _ = self.shell.internal_force(u, R)
        return fu + fs, fth + ft, d, f_loc


def helmet_rig(lat, r_elem, mat, joint=None, R_head=80.0, R_anv=48.0, gap0=0.05, t_shell=2.0, v0_ms=4.8,
               head_mass=None, areal_mass_kg_cm2=5.0 / 40.0, shell_mat=None, mu=0.5, courant=0.7, k_anv_rel=1.0,
               eps_anv=0.02, line_contact=True, freeze=None):
    """Build the helmet patch rig with the interface of latticeSetup.drop_rig (fr, beam_law, m, j, fix_u, fix_r, contact,
    tie, bot, imp, dt_fn, v_init, self_contact, ...). r_elem: per-element radii [mm]; head mass defaults to the areal mass
    (5 kg over an assumed 40 cm2 effective area, NOTES: assumption) times the patch area. Units mm, t, s, N."""
    from latticeSetup import circle_sections
    from explicitFE import ViscoHyperLaw, JointScaledLaw, adaptive_dt
    from contact import hertz_potential, hertz_tangent, potential_force
    shell_mat = shell_mat or dict(E=2300.0, nu=0.38, rho=1.2e-9)
    clat = curved_lattice(lat, R_head)
    sh, X_extra, nodes = shell_mesh(clat, t_shell, shell_mat['E'], shell_mat['nu'], shell_mat['rho'])
    Xl, connr = clat['Xr'], clat['connr']
    Nl = len(Xl)
    X = np.vstack((Xl, X_extra))
    fr = HybridFrame(X, connr, sh)
    A, I = circle_sections(r_elem, nu=mat['nu'])
    beam_law = ViscoHyperLaw(fr, A, I, mat, fibre=True, relax=mat.get('relax', 'nominal'))
    if joint is not None:
        import jointRule
        sj = jointRule.rule_factors(lat, r_elem, **{k_: v for k_, v in joint.items() if k_ in jointRule.DEFAULT})
        beam_law = JointScaledLaw(beam_law, sj)
    m, j = lumped_mass(fr, A, I, mat['rho'])
    m = m.clone()
    j = j.clone()
    idx = sh.nodes
    m = m.index_add(0, idx, sh.m)
    j = j.index_add(0, idx, sh.j)
    # ---- head: inner-end liner nodes (flat z = 0) move as one rigid translating body carrying the head mass
    flat = clat['flat']
    inner = np.nonzero(np.isclose(flat['Xr'][:, 2], flat['Xr'][:, 2].min()))[0]
    inner_t = torch.as_tensor(inner)
    W = clat['W']
    area = W * W                                                    # patch plan area [mm2] (on the head sphere ~ W^2)
    M = head_mass if head_mass is not None else areal_mass_kg_cm2 * 1e-3 * (area * 1e-2)   # [t]
    if freeze is not None:
        M = freeze['M']
    m[inner_t] += M / len(inner)
    fix_u = torch.zeros(fr.N, 3, dtype=torch.bool)
    fix_r = torch.zeros(fr.N, 3, dtype=torch.bool)
    fix_r[inner_t] = True
    v0 = v0_ms * 1e3
    v_init = torch.zeros(fr.N, 3, dtype=torch.float64)
    v_init[:, 2] = v0                                                # everything moves toward the anvil (+z, above the crown)
    # ---- anvil: rigid fixed sphere above the crown; contact on the shell's outer face points x - t D
    z_crown = float(np.linalg.norm(X[nodes].reshape(-1, 3), axis=1).max())    # shell reference-surface apex radius
    z_out = z_crown + t_shell                                       # outer face at the crown
    C_a = torch.tensor([0.0, 0.0, z_out + gap0 + R_anv], dtype=fr.X.dtype)
    Dsh = sh.D
    k_node = k_anv_rel * shell_mat['E'] / t_shell * sh.A_node       # penalty stiffness per node [N/mm]
    last = {}

    def phi_anv(x):
        xs = x[idx] - t_shell * Dsh                                  # outer-face points
        g = (xs - C_a).norm(dim=1) - R_anv
        d = eps_anv * torch.nn.functional.softplus(-g / eps_anv)
        return 0.5 * (k_node * d * d).sum(), (g,)

    phi_lat, sc = None, None
    if line_contact:
        from contact import LineContact
        sc = LineContact(fr, r_elem, lat['beam_of_elem'], mat['E0'], mat['nu'], beam_conn=lat['conn'],
                         eps=None if freeze is None else freeze['self_contact'].eps)
    dt_now = {'h': 1e-6}

    def phi_total(x):
        Phi, ex = phi_anv(x)
        if sc is not None:
            Ps, g_s = sc.energy(x)
            Phi = Phi + Ps
        return Phi, ex

    def contact(u, v):
        f, Phi, ex = potential_force(phi_total, fr.X + u)
        if sc is not None and mu != 0:
            from contact import _line_friction
            ff, Pf = _line_friction(sc, fr.X + u, v, mu, m, dt_now['h'])
            f = f + ff
        g = ex[0]
        last['g_min'] = g.min().detach()
        # anvil force on the shell (z, pushing up = +): -dPhi_anvil/dx summed
        fa, _, _ = potential_force(lambda x: phi_anv(x), (fr.X + u).detach())
        last['F_anv'] = -fa[idx, 2].sum().detach()      # compressive load on the anvil > 0
        return f

    def contact_dt(u, v):
        x = fr.X + u
        w = 0.0
        with torch.no_grad():
            xs = x[idx] - t_shell * Dsh
            g = (xs - C_a).norm(dim=1) - R_anv
            near = g < 6 * eps_anv
            if near.any():
                kt = k_node[near] * torch.sigmoid(-g[near] / eps_anv)
                w = float(torch.sqrt(kt / m[idx][near]).max())
        cap = np.inf
        if sc is not None:
            w_s, cap = sc.dt_terms(x, v, m, 2 * sc.eps)
            w = float(np.sqrt(w**2 + w_s**2))
        return w, cap

    dt_base = adaptive_dt(fr, beam_law, m, j, courant=courant, contact_fn=contact_dt)

    def dt_fn(n, u, R, v=None):
        dt_now['h'] = dt_base(n, u, R, v)
        return dt_now['h']

    return dict(fr=fr, A=A, I=I, beam_law=beam_law, m=m, j=j, M=M, imp=int(inner[0]), top=inner_t, bot=idx,
                fix_u=fix_u, fix_r=fix_r, contact=contact, tie=inner_t, dt_fn=dt_fn, v_init=v_init, v0=v0, gap0=gap0,
                self_contact=sc, mu=mu, last=last, dt_now=dt_now, r_elem=r_elem, shell=sh, shell_nodes=idx,
                inner=inner_t, C_anvil=C_a, clat=clat, lat=lat)


def run_helmet(rig, T=2e-3, record_every=5):
    """Plain forward run (no graph). Returns histories: t, head deceleration [g], anvil force [N], energies, shell deflection."""
    from explicitSolve import simulate
    fr, inner, idx = rig['fr'], rig['inner'], rig['shell_nodes']
    m, M = rig['m'], rig['M']
    m_head = m[inner].sum().item()

    def cb(n, t, state, fint):
        u, v, R, w = state
        fu, fth, d, f_loc, fc = fint
        a_head = ((fc[inner, 2] - fu[inner, 2]).sum() / m_head).item()
        ke = 0.5 * (m * (v**2).sum(1)).sum().item()
        return {'a_head_g': -a_head / 9.81e3, 'F_anv': float(rig['last'].get('F_anv', 0.0)), 'KE': ke,
                'v_head': v[inner[0], 2].item(), 'u_head': u[inner[0], 2].item(), 'u_crown': u[idx[0], 2].item(),
                'g_min': float(rig['last'].get('g_min', np.nan))}

    with torch.no_grad():
        h = simulate(fr, rig['beam_law'], m, rig['j'], rig['dt_fn'], None, t_end=T, v0=rig['v_init'],
                     fix_u=rig['fix_u'], fix_r=rig['fix_r'], contact=rig['contact'], tie=rig['tie'], callback=cb,
                     record_every=record_every)
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in h.items()}
