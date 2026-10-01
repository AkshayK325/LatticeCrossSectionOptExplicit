"""Stateless corotational 3D Timoshenko frame for explicit dynamics (torch, differentiable).

State per node: position x = X + u (N,3) and rotation R (N,3,3) from the initial configuration.
Per element the local deformations d = (dL, rx, ry1, rz1, ry2, rz2) are computed from (x, R) only
(no stored frames), the local resultants f_loc = law(d) come from the section/material law, and the
global internal force is the exact virtual-work transpose  f_int = (dd/dq)^T f_loc  (autograd VJP).
That makes f_int consistent with the strain energy for elastic laws (energy-conserving explicit
integration) and lets any material law (hyperelastic, Prony) plug in by changing `law` only.

Element frame: chord e1, plus the Battini/Pacoste average of the two nodal y-axes (no state needed).
Local rotations: same asin formulas as frameFE3.intForceVctNonLin.
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
        dot = lambda a, b: (a * b).sum(1)
        s = lambda v: torch.asin(torch.clamp(0.5 * v, -1 + 1e-12, 1 - 1e-12))
        nx1, ny1, nz1 = t1.unbind(-1)
        nx2, ny2, nz2 = t2.unbind(-1)
        rx = s(dot(e3, ny2) - dot(e2, nz2)) - s(dot(e3, ny1) - dot(e2, nz1))
        ry1, rz1 = s(dot(e1, nz1) - dot(e3, nx1)), s(dot(e2, nx1) - dot(e1, ny1))
        ry2, rz2 = s(dot(e1, nz2) - dot(e3, nx2)), s(dot(e2, nx2) - dot(e1, ny2))
        return torch.stack((L - self.L0, rx, ry1, rz1, ry2, rz2), dim=1)

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


def stable_dt(frame, law, m, j, courant=0.8, iters=100, seed=0):
    """Central-difference limit 2/omega_max, omega_max^2 = max eig(M^-1/2 K M^-1/2) by power iteration
    on the tangent at the reference state (K v via a JVP of internal_force). Free-free (no BCs): an upper
    bound on omega for any support, so the dt is safe."""
    g = torch.Generator().manual_seed(seed)
    dtype = frame.X.dtype
    s = torch.cat((m[:, None].expand(-1, 3), j[:, None].expand(-1, 3)), dim=0).reshape(-1) ** -0.5
    R0 = torch.eye(3, dtype=dtype).expand(frame.N, 3, 3).contiguous()

    def Kv(v):
        v = (v * s).reshape(2, frame.N, 3)
        q = torch.zeros(2 * frame.N * 3, dtype=dtype, requires_grad=True)
        def f(q):
            qq = q.reshape(2, frame.N, 3)
            fu, fth, _, _ = frame.internal_force(qq[0], rodrigues(qq[1]) @ R0, law)
            return torch.cat((fu, fth)).reshape(-1)
        _, jv = torch.autograd.functional.jvp(f, q, v.reshape(-1))
        return jv * s

    v = torch.randn(2 * frame.N * 3, dtype=dtype, generator=g)
    lam = 0.0
    for _ in range(iters):
        w = Kv(v)
        lam = (v @ w).item() / (v @ v).item()
        v = w / w.norm()
    return courant * 2 / np.sqrt(lam), np.sqrt(lam)
