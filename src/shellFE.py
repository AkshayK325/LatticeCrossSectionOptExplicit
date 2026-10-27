"""Differentiable 6-DOF director shell (3-node triangles) for the explicit solver, with a thickness offset.

Same state as the beams (explicitFE.Frame): node positions x = X + u and nodal rotations R (N,3,3) from the initial
configuration. A shell node carries a director d_i = R_i D_i (D_i = unit reference normal) and sits on the REFERENCE
surface. The material mid-surface is offset from it along the director, xm_i = x_i + o d_i (o = signed offset [mm];
Abaqus OFFSET 0.5 with the thickness toward the anvil is o = -t/2 when D points to the head). All strain measures are
objective (rigid motions give zero energy) and the internal force is the exact gradient of the energy, as in Frame.

Kinematics per element (constant fields, 2D shape-function gradients in the reference tangent basis):
  membrane   E_ab = 1/2 (xm_,a . xm_,b - delta_ab)            (Green-Lagrange of the mid-surface)
  bending    kappa_ab = -1/2 (xm_,a . d_,b + xm_,b . d_,a) - same for the reference configuration (curved shells)
  shear      edge-based assumed strain g_e = [(xm_j - xm_i).(d_i + d_j)/2 - ref] / L_e, reconstructed by least squares
             to one constant vector (locking-free, exact for linear fields)
  drilling   small penalty on the rotation of each nodal triad about the normal relative to the element frame
Material: linear elastic plane stress (E, nu), shear factor 5/6. Lumped mass rho t A/3 per node, rotary inertia
rho A t (t^2 + h^2)/12 / 3 with h the mean element size (standard rotary-inertia scaling).
"""
import numpy as np
import torch

from explicitFE import log_so3


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def vertex_normals(X, tri, toward=None):
    """Area-weighted unit vertex normals of a triangle mesh; oriented so that they point toward `toward` (a point)
    when given (mean over nodes), else along the first face normal."""
    X = np.asarray(X, float)
    fn = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])      # area-weighted face normals
    if toward is not None:
        c = X[tri].mean(1)
        s = np.sign(((np.asarray(toward) - c) * fn).sum(1))
        s[s == 0] = 1
        fn = fn * s[:, None]
    else:
        fn = fn * np.sign((fn * fn[0]).sum(1) + 1e-30)[:, None]
    D = np.zeros_like(X)
    for k in range(3):
        np.add.at(D, tri[:, k], fn)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


class Shell:
    """Fixed reference geometry. X (Ns,3) reference-surface nodes, tri (nt,3), D (Ns,3) unit reference normals.
    nodes (Ns,) optional map to global node numbers of the host frame (default 0..Ns-1)."""

    def __init__(self, X, tri, D, t, E, nu, rho, offset=0.0, nodes=None, k_drill=1.0, dtype=None, device=None):
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.get_default_device() if device is None else device
        T = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.X, self.D = T(X), _unit(T(D))
        self.tri = torch.as_tensor(np.asarray(tri), dtype=torch.long, device=device)
        self.nodes = torch.arange(len(X), device=device) if nodes is None else torch.as_tensor(nodes, device=device)
        self.t, self.o = float(t), float(offset)
        self.E, self.nu, self.rho = float(E), float(nu), float(rho)
        self.Ns, self.nt = len(X), len(tri)
        Xm = self.X + self.o * self.D
        P = Xm[self.tri]                                                     # (nt,3,3) reference mid-surface corners
        a1 = _unit(P[:, 1] - P[:, 0])
        n = _unit(torch.linalg.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]))
        a2 = torch.linalg.cross(n, a1)
        self.basis = torch.stack((a1, a2), 1)                                # (nt,2,3) reference tangent basis
        c = torch.einsum('eia,eka->eik', P - P[:, :1], self.basis)           # (nt,3,2) local 2D corner coordinates
        self.area = 0.5 * ((c[:, 1, 0] * c[:, 2, 1]) - (c[:, 1, 1] * c[:, 2, 0]))
        assert (self.area > 0).all(), 'shell triangles must be counter-clockwise about the director side'
        # shape-function gradients (nt,3,2): rows of inverse of [[1, c_i]] matrix
        M = torch.cat((torch.ones(self.nt, 3, 1, dtype=dtype, device=device), c), 2)
        self.gradN = torch.linalg.inv(M)[:, 1:, :].transpose(1, 2)           # (nt,3,2)
        # assumed-shear: edges (0,1),(1,2),(2,0); tangents in local 2D; LS reconstruction matrix (nt,2,3)
        ed = torch.tensor([[0, 1], [1, 2], [2, 0]], device=device)
        self.edges = ed
        dc = c[:, ed[:, 1]] - c[:, ed[:, 0]]                                 # (nt,3,2)
        self.Le = dc.norm(dim=2)                                             # (nt,3)
        Te = dc / self.Le[..., None]
        self.Pls = torch.linalg.solve(Te.transpose(1, 2) @ Te, Te.transpose(1, 2))   # (nt,2,3)
        # reference curvature and shear offsets (curved reference shell has none of them as strains)
        self.kappa0 = self._kappa(Xm, self.D)
        self.gam0 = self._edge_shear(Xm, self.D)
        # constitutive constants
        t_, E_, nu_ = self.t, self.E, self.nu
        self.Cm = E_ * t_ / (1 - nu_**2)
        self.Cb = E_ * t_**3 / (12 * (1 - nu_**2))
        self.Gs = (5.0 / 6.0) * E_ / (2 * (1 + nu_)) * t_
        self.k_drill = k_drill * E_ / (2 * (1 + nu_)) * t_
        # node areas, lumped mass, rotary inertia
        An = torch.zeros(self.Ns, dtype=dtype, device=device).index_add(
            0, self.tri.reshape(-1), (self.area / 3).repeat_interleave(3))
        self.A_node = An
        h = torch.sqrt(2 * self.area).mean().item()
        self.m = self.rho * t_ * An
        self.j = self.rho * t_ * An * (t_**2 + h**2) / 12
        # reference in-plane axis per element for the drilling measure (global vector at reference)
        self.a1 = self.basis[:, 0]
        self.psi0 = self._drill(Xm, self.D, torch.eye(3, dtype=dtype, device=device).expand(self.Ns, 3, 3))

    # ---- strain measures ------------------------------------------------------------------------------------
    def _grads(self, v):
        """v (Ns,3) -> (nt,3,2): per-element derivative of a nodal vector field wrt the local reference coordinates."""
        return torch.einsum('eik,eia->eka', v[self.tri], self.gradN)

    def _kappa(self, xm, d):
        xa, da = self._grads(xm), self._grads(d)                              # (nt,3,2)
        K = -0.5 * (torch.einsum('eka,ekb->eab', xa, da) + torch.einsum('ekb,eka->eab', xa, da))
        return K

    def _edge_shear(self, xm, d):
        i, j = self.tri[:, self.edges[:, 0]], self.tri[:, self.edges[:, 1]]    # (nt,3)
        dx = xm[j] - xm[i]
        return (dx * 0.5 * (d[i] + d[j])).sum(-1) / self.Le

    def _drill(self, xm, d, R):
        """Rotation (nt,3) of each nodal triad about the normal relative to the element's polar rotation: angle between
        the nodal in-plane axis R a1 and q1, the first column of the rotation part Q = x_,a (x_,a^T x_,a)^(-1/2) of
        the mid-surface tangent map (closed-form 2x2 square root; zero for any pure stretch)."""
        xa = self._grads(xm)                                                  # (nt,3,2)
        C = torch.einsum('eka,ekb->eab', xa, xa)
        sd = torch.sqrt(C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] * C[:, 1, 0])
        tr = C[:, 0, 0] + C[:, 1, 1]
        den = torch.sqrt(tr + 2 * sd)
        Ush = (C + sd[:, None, None] * torch.eye(2, dtype=C.dtype, device=C.device)) / den[:, None, None]
        Uinv = torch.linalg.inv(Ush)
        Q = xa @ Uinv                                                         # (nt,3,2)
        q1 = Q[:, :, 0]
        nq = _unit(torch.linalg.cross(Q[:, :, 0], Q[:, :, 1]))
        q2 = torch.linalg.cross(nq, q1)
        ta = torch.einsum('enij,ej->eni', R[self.tri], self.a1)               # (nt,3 nodes,3)
        return torch.atan2((ta * q2[:, None]).sum(-1), (ta * q1[:, None]).sum(-1))

    def element_energy(self, x, R):
        """Energy per element (nt,). x (Ns,3) current reference-surface positions, R (Ns,3,3) nodal rotations."""
        d = torch.einsum('nij,nj->ni', R, self.D)
        xm = x + self.o * d
        xa = self._grads(xm)
        Eg = 0.5 * (torch.einsum('eka,ekb->eab', xa, xa) - torch.eye(2, dtype=x.dtype, device=x.device))
        nu = self.nu
        e11, e22, e12 = Eg[:, 0, 0], Eg[:, 1, 1], Eg[:, 0, 1]
        Wm = 0.5 * self.Cm * (e11**2 + e22**2 + 2 * nu * e11 * e22 + 2 * (1 - nu) * e12**2)
        kap = self._kappa(xm, d) - self.kappa0
        k11, k22, k12 = kap[:, 0, 0], kap[:, 1, 1], kap[:, 0, 1]
        Wb = 0.5 * self.Cb * (k11**2 + k22**2 + 2 * nu * k11 * k22 + 2 * (1 - nu) * k12**2)
        ge = self._edge_shear(xm, d) - self.gam0                             # (nt,3)
        gv = torch.einsum('eab,eb->ea', self.Pls, ge)
        Ws = 0.5 * self.Gs * (gv**2).sum(1)
        psi = self._drill(xm, d, R) - self.psi0
        Wd = 0.5 * self.k_drill * (self.area / 3) * (psi**2).sum(1)
        return self.area * (Wm + Wb + Ws) + Wd

    def internal_force(self, u, R, create_graph=None):
        """Nodal force and moment of the shell: exact (dW/du, dW/dtheta) as 6 DOF on the host frame's nodes.
        u (N,3), R (N,3,3) are host-frame arrays; returns (fu (N,3), fth (N,3), W)."""
        outer = torch.is_grad_enabled() if create_graph is None else create_graph
        idx = self.nodes
        du = torch.zeros(self.Ns, 3, dtype=u.dtype, device=u.device, requires_grad=True)
        dth = torch.zeros(self.Ns, 3, dtype=u.dtype, device=u.device, requires_grad=True)
        with torch.enable_grad():
            Rs = R[idx]
            W = self.element_energy(self.X + u[idx] + du, Rs + _skew(dth) @ Rs).sum()
            need = outer and (u.requires_grad or R.requires_grad)
            gu, gth = torch.autograd.grad(W, (du, dth), create_graph=need)
        fu = torch.zeros_like(u).index_add(0, idx, gu)
        fth = torch.zeros_like(u).index_add(0, idx, gth)
        return fu, fth, (W if outer else W.detach())


def _skew(w):
    from explicitFE import skew
    return skew(w)
