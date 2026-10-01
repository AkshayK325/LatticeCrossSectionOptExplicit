"""Stateless corotational 3D Timoshenko frame for explicit dynamics (torch, differentiable).

State per node: position x = X + u (N,3) and rotation R (N,3,3) from the initial configuration.
Per element the local deformations d = (dL, rx, ry1, rz1, ry2, rz2) are computed from (x, R) only
(no stored frames), the local resultants f_loc = law(d) come from the section/material law, and the
global internal force is the exact virtual-work transpose  f_int = (dd/dq)^T f_loc  (autograd VJP).
That makes f_int consistent with the strain energy for elastic laws (energy-conserving explicit
integration) and lets any material law (hyperelastic, Prony) plug in by changing `law` only.

Element frame: chord e1, plus the Battini/Pacoste average of the two nodal y-axes (no state needed).
Local rotations: log map of (element frame)^T (nodal triad); equals the frameFE3 asin measures to first order
but stays smooth to ~pi (asin stiffens like 1/cos and blows up near 90 deg under crush).
Section arrays use the frameFE3 layout: A (ne,3) = [axial, shear_y, shear_z], I (ne,3) = [J, Iy, Iz].
"""
import numpy as np
import torch


def skew(w):
    O = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    O[..., 0, 1], O[..., 0, 2], O[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    return O - O.transpose(-1, -2)


def rodrigues(w):
    """Rotation matrices exp(skew(w)) for w (n,3); smooth and differentiable at w = 0."""
    r = torch.sqrt((w * w).sum(-1) + 1e-30)
    a = torch.sinc(r / torch.pi)                        # sin r / r
    b = 0.5 * torch.sinc(r / (2 * torch.pi)) ** 2       # (1 - cos r) / r^2
    S = skew(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(S)
    return I + a[..., None, None] * S + b[..., None, None] * (S @ S)


def log_so3(Q):
    """Rotation vector of Q (n,3,3), smooth for angles in [0, pi) (atan2 form, no asin/acos singularity).
    For small rotations its components equal the legacy asin(0.5*(...)) measures to first order."""
    a = 0.5 * torch.stack((Q[:, 2, 1] - Q[:, 1, 2], Q[:, 0, 2] - Q[:, 2, 0], Q[:, 1, 0] - Q[:, 0, 1]), dim=1)
    s = torch.sqrt((a * a).sum(1) + 1e-30)               # sin(phi)
    c = 0.5 * (Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2] - 1)  # cos(phi)
    return a * (torch.atan2(s, c) / s)[:, None]


class Frame:
    """Fixed geometry/topology. X (N,3), conn (ne,2) numpy arrays (already refined)."""

    def __init__(self, X, conn, dtype=torch.float64, device='cpu'):
        self.X = torch.as_tensor(X, dtype=dtype, device=device)
        self.conn = torch.as_tensor(conn, dtype=torch.long, device=device)
        self.n1, self.n2 = self.conn[:, 0], self.conn[:, 1]
        self.N, self.ne = self.X.shape[0], self.conn.shape[0]
        d = self.X[self.n2] - self.X[self.n1]
        self.L0 = d.norm(dim=1)
        x = d / self.L0[:, None]
        # reference vector: the global axis least aligned with the beam (same rule as frameFE3)
        cand = torch.eye(3, dtype=dtype, device=device)
        v = cand[(x @ cand.T).abs().argmin(dim=1)]
        y = torch.linalg.cross(v, x)
        y = y / y.norm(dim=1, keepdim=True)
        z = torch.linalg.cross(x, y)
        self.T0 = torch.stack((x, y, z), dim=1)          # rows = initial local axes

    def local_deformations(self, x, R):
        """d (ne,6) = [dL, rx, ry1, rz1, ry2, rz2] for positions x (N,3) and nodal rotations R (N,3,3)."""
        t1 = R[self.n1] @ self.T0.transpose(1, 2)        # columns = current nodal triad at end 1
        t2 = R[self.n2] @ self.T0.transpose(1, 2)
        d = x[self.n2] - x[self.n1]
        L = d.norm(dim=1)
        e1 = d / L[:, None]
        q = 0.5 * (t1[:, :, 1] + t2[:, :, 1])
        e3 = torch.linalg.cross(e1, q)
        e3 = e3 / e3.norm(dim=1, keepdim=True)
        e2 = torch.linalg.cross(e3, e1)
        Et = torch.stack((e1, e2, e3), dim=1)            # rows = current element axes
        th1, th2 = log_so3(Et @ t1), log_so3(Et @ t2)    # nodal rotation relative to the element frame
        return torch.stack((L - self.L0, th2[:, 0] - th1[:, 0], th1[:, 1], th1[:, 2], th2[:, 1], th2[:, 2]), dim=1)

    def elastic_law(self, d, A, I, E, G):
        """Linear Timoshenko resultants (N, T, My1, Mz1, My2, Mz2) — same constants as frameFE3."""
        L = self.L0
        N = E * A[:, 0] * d[:, 0] / L
        T = G * I[:, 0] * d[:, 1] / L
        out = [N, T]
        Mb = []
        for Ib, As, r1, r2 in ((I[:, 1], A[:, 2], d[:, 2], d[:, 4]), (I[:, 2], A[:, 1], d[:, 3], d[:, 5])):
            om = E * Ib / (G * As * L * L)
            mu, lam, gam = 1 + 12 * om, 1 + 3 * om, 1 - 6 * om
            c = 2 * E * Ib / (mu * L)
            Mb.append((c * (2 * lam * r1 + gam * r2), c * (gam * r1 + 2 * lam * r2)))
        (My1, My2), (Mz1, Mz2) = Mb
        return torch.stack(out + [My1, Mz1, My2, Mz2], dim=1)

    def internal_force(self, u, R, law):
        """Global internal force and moment, each (N,3): exact (dd/du, dd/dtheta)^T law(d).

        `law(d) -> f_loc (ne,6)`. Rotational variation is a spatial spin dtheta: R -> (I + skew(dtheta)) R.
        Differentiable w.r.t. u, R and any tensors inside `law` (create_graph when grad is needed).
        """
        du = torch.zeros_like(u, requires_grad=True)
        dth = torch.zeros(self.N, 3, dtype=u.dtype, device=u.device, requires_grad=True)
        with torch.enable_grad():
            d = self.local_deformations(self.X + u + du, R + skew(dth) @ R)
            f_loc = law(d)
            need_graph = torch.is_grad_enabled() and (u.requires_grad or R.requires_grad or f_loc.requires_grad)
            fu, fth = torch.autograd.grad(d, (du, dth), grad_outputs=f_loc, create_graph=need_graph)
        return fu, fth, d, f_loc


def lumped_mass(frame, A, I, rho, rot_scale=True):
    """Nodal translational mass m (N,) and scalar rotary inertia j (N,).

    Rotary inertia per element end: rho*L/2 * max(Iy+Iz, A*L^2/12) when rot_scale (standard explicit
    rotary-inertia scaling so rotation DOFs do not set the time step), else rho*L/2*(Iy+Iz).
    # ponytail: scalar (isotropic) rotary inertia per node, no gyroscopic term; full tensor if spin matters
    """
    L = frame.L0
    me = rho * A[:, 0] * L / 2
    ie = I[:, 1] + I[:, 2]
    if rot_scale:
        ie = torch.maximum(ie, A[:, 0] * L * L / 12)
    je = rho * L / 2 * ie
    z = torch.zeros(frame.N, dtype=A.dtype, device=A.device)
    m = z.index_add(0, frame.n1, me).index_add(0, frame.n2, me)
    j = z.index_add(0, frame.n1, je).index_add(0, frame.n2, je)
    return m, j


def stable_dt(frame, law, m, j, courant=0.8, iters=100, seed=0, u=None, R=None, v0=None, return_vec=False,
              phi_extra=None):
    """Central-difference limit 2/omega_max, omega_max^2 = max eig(M^-1/2 K M^-1/2) by power iteration
    on the tangent at state (u, R) (default: reference state; K v via a JVP of internal_force).
    Free-free (no BCs): an upper bound on omega for any support, so the dt is safe.
    phi_extra(x) -> (Phi, extra): additional potential (contact) whose Hessian is added to the tangent."""
    g = torch.Generator().manual_seed(seed)
    dtype = frame.X.dtype
    s = torch.cat((m[:, None].expand(-1, 3), j[:, None].expand(-1, 3)), dim=0).reshape(-1) ** -0.5
    R0 = torch.eye(3, dtype=dtype).expand(frame.N, 3, 3).contiguous() if R is None else R.detach()
    u0 = torch.zeros(frame.N, 3, dtype=dtype) if u is None else u.detach()

    def Kv(v):
        v = (v * s).reshape(2, frame.N, 3)
        q = torch.zeros(2 * frame.N * 3, dtype=dtype, requires_grad=True)
        def f(q):
            qq = q.reshape(2, frame.N, 3)
            fu, fth, _, _ = frame.internal_force(u0 + qq[0], rodrigues(qq[1]) @ R0, law)
            if phi_extra is not None:                  # e.g. contact potential: its stiffness counts too
                from contact import potential_force
                fu = fu - potential_force(phi_extra, frame.X + u0 + qq[0])[0]
            return torch.cat((fu, fth)).reshape(-1)
        _, jv = torch.autograd.functional.jvp(f, q, v.reshape(-1))
        return jv * s

    v = torch.randn(2 * frame.N * 3, dtype=dtype, generator=g) if v0 is None else v0
    lam = 0.0
    for _ in range(iters):
        w = Kv(v)
        lam = (v @ w).item() / (v @ v).item()
        v = w / w.norm()
    if return_vec:
        return courant * 2 / np.sqrt(lam), np.sqrt(lam), v
    return courant * 2 / np.sqrt(lam), np.sqrt(lam)


def adaptive_dt(frame, law, m, j, w_extra=0.0, courant=0.7, every=20, iters=8, warm_iters=60, contact_fn=None,
                phi_extra=None):
    """dt(step, u, R, v) callable for simulate(): re-estimates omega_max at the current state every `every`
    steps by a warm-started power iteration. Power iteration approaches omega_max from below, hence the
    default courant 0.7.
    w_extra: constant frequency of extra stiffness on the same nodes (e.g. flat contact), added in quadrature.
    contact_fn(u, v) -> (w_c, dt_cap): state-dependent contact frequency (added in quadrature, evaluated every
    step since contact stiffness changes fast) and a hard step cap (e.g. barrier CCD)."""
    st = {'v': None, 'w': None, 'dt0': None}

    def dt_fn(n, u, R, v=None):
        if n % every == 0:
            _, st['w'], st['v'] = stable_dt(frame, law, m, j, iters=warm_iters if st['v'] is None else iters,
                                            u=u, R=R, v0=st['v'], return_vec=True, phi_extra=phi_extra)
        w_c, cap = (0.0, np.inf) if contact_fn is None else contact_fn(u, v)
        dt = min(courant * 2 / np.sqrt(st['w']**2 + w_extra**2 + w_c**2), cap)
        st['dt0'] = st['dt0'] or dt
        if not np.isfinite(dt) or dt < 1e-3 * st['dt0']:
            raise RuntimeError(f'dt collapsed to {dt:.2e} at step {n}: solution unstable')
        return dt
    return dt_fn


class ViscoHyperLaw:
    """Section law for visco-hyperelastic elastomers (EPU 45): resultants f = f_inf(d) + sum_i q_i.

    f_inf: long-term part. Axial N = A * P_yeoh(lambda), lambda = 1 + dL/L0 (finite strain, resists element
    collapse); torsion/bending/shear linear with E_inf, G_inf (Timoshenko, as elastic_law).
    q_i: quasi-linear (Fung) Prony branches driven by increments of f_inf:
        q_i <- a_i q_i + beta_i b_i (f_inf(d) - f_inf(d_prev)),   a = exp(-h/tau), b = (1 - a)/(h/tau),
        beta_i = g_i / (1 - sum g)   (so instantaneous response = E0 / E_inf * f_inf).
    Exact for a constant rate within the step. simulate() sets `step_dt` before and calls `commit(d)` after
    each step; calls in between (stable_dt, autograd) never change the history. With step_dt -> 0 the law
    returns the instantaneous (E0) response, which is what the stable time step must see.
    # ponytail: linear (E_inf) bending/torsion; fibre integration if bending strains get large
    """

    def __init__(self, frame, A, I, mat, fibre=True, n_r=4, n_theta=16):
        self.fr, self.A, self.I, self.mat = frame, A, I, mat
        self.C = mat['yeoh']
        self.fibre = fibre
        if fibre:                                  # polar quadrature of a solid circle of radius r (unit r)
            xg, wg = np.polynomial.legendre.leggauss(n_r)
            s_, ws = 0.5 * (xg + 1), 0.5 * wg      # int_0^1 f(s) s ds ~ sum ws*s*f(s)
            th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
            S, TH = np.meshgrid(s_, th, indexing='ij')
            W = (ws * s_)[:, None] * np.full(n_theta, 2 * np.pi / n_theta)[None, :]
            self.fy = torch.tensor((S * np.cos(TH)).ravel(), dtype=A.dtype)     # fibre y / r
            self.fz = torch.tensor((S * np.sin(TH)).ravel(), dtype=A.dtype)     # fibre z / r
            self.fw = torch.tensor(W.ravel(), dtype=A.dtype)                    # area weight / r^2
            self.r_sec = torch.sqrt(A[:, 0] / torch.pi)
        self.tau = torch.tensor(mat['prony_tau'], dtype=A.dtype)[:, None, None]
        g = torch.tensor(mat['prony_g'], dtype=A.dtype)[:, None, None]
        self.beta = g / (1 - g.sum())
        self.step_dt = 0.0
        self.reset()

    def reset(self):
        ne = self.fr.ne
        self.q = torch.zeros(self.tau.shape[0], ne, 6, dtype=self.A.dtype)
        self.f_prev = torch.zeros(ne, 6, dtype=self.A.dtype)

    def _fibre_energy(self, d, quadratic):
        """Section-integrated Yeoh energy per element (fibre model), or its small-strain quadratic form.
        Curvatures/twist from the corotational end rotations: kappa = (r2 - r1)/L0, twist = rx/L0."""
        L0, r = self.fr.L0, self.r_sec
        e0 = d[:, 0] / L0
        ky, kz, tw = (d[:, 4] - d[:, 2]) / L0, (d[:, 5] - d[:, 3]) / L0, d[:, 1] / L0
        y, z = r[:, None] * self.fy[None, :], r[:, None] * self.fz[None, :]
        eps = e0[:, None] - y * kz[:, None] + z * ky[:, None]                   # fibre axial strain
        g2 = (tw[:, None] ** 2) * (y * y + z * z)                               # torsional shear^2
        C10, C20, C30 = self.C
        if quadratic:
            W = C10 * (3 * eps * eps + g2)                                      # small-strain limit of Yeoh
        else:
            lam = 1 + eps
            x = lam * lam + 2 / lam - 3 + g2                                    # I1 - 3 (uniaxial + simple shear)
            W = C10 * x + C20 * x * x + C30 * x * x * x
        return (L0 * r * r * (W * self.fw[None, :]).sum(1)).sum()

    def f_inf(self, d):
        """Long-term resultants. Fibre mode: exact linear Timoshenko law + large-strain correction
        d/dd (U_fibre - U_quadratic): equal to the validated linear element at small strain, Yeoh in axial,
        bending and torsion at large strain, and conservative. Non-fibre mode: Yeoh axial only (previous)."""
        from material import yeoh_uniaxial
        f = self.fr.elastic_law(d, self.A, self.I, self.mat['E'], self.mat['G'])
        if not self.fibre:
            lam = 1 + d[:, 0] / self.fr.L0
            return torch.cat((self.A[:, :1] * yeoh_uniaxial(lam, self.C)[:, None], f[:, 1:]), dim=1)
        with torch.enable_grad():
            dd = d if d.requires_grad else d.detach().requires_grad_(True)
            corr, = torch.autograd.grad(self._fibre_energy(dd, False) - self._fibre_energy(dd, True), dd,
                                        create_graph=True)
        return f + corr

    def _branches(self, f):
        x = torch.as_tensor(max(self.step_dt, 1e-300), dtype=f.dtype) / self.tau
        a = torch.exp(-x)
        b = -torch.expm1(-x) / x                     # (1 - a)/x, -> 1 as h -> 0
        return a * self.q + self.beta * b * (f - self.f_prev)

    def __call__(self, d):
        f = self.f_inf(d)
        return f + self._branches(f).sum(0)

    def commit(self, d):
        f = self.f_inf(d)
        self.q = self._branches(f)
        self.f_prev = f

    def stored_long_term(self, d):
        """Long-term elastic energy (diagnostics). Fibre mode: linear Timoshenko energy + (U_fibre - U_quad)."""
        from material import yeoh_energy
        f = self.fr.elastic_law(d, self.A, self.I, self.mat['E'], self.mat['G'])
        if self.fibre:
            with torch.no_grad():
                return 0.5 * (d * f).sum() + self._fibre_energy(d, False) - self._fibre_energy(d, True)
        lam = 1 + d[:, 0] / self.fr.L0
        return (self.A[:, 0] * self.fr.L0 * yeoh_energy(lam, self.C)).sum() + 0.5 * (d[:, 1:] * f[:, 1:]).sum()
