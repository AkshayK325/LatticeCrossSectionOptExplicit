"""Smooth penalty contact (differentiable). Units mm / N.

Flat rigid surface z = z_s with outward normal +z. A node of beam radius r touches when
g = z - z_s - r < 0. Force (along +z) = kc * eps * softplus(-g / eps): smooth, -> kc * (-g) deep in contact.
"""
import numpy as np
import torch
import torch.nn.functional as F


def node_radius(frame, A):
    """Nodal contact radius: the largest equivalent-circle radius sqrt(A/pi) of the attached elements.
    # ponytail: circle-equivalent radius; VAE sections need their half-extent along the normal (plan Day 3)"""
    re = torch.sqrt(A[:, 0] / torch.pi)
    r = torch.zeros(frame.N, dtype=A.dtype, device=A.device)
    r = r.scatter_reduce(0, frame.n1, re, reduce='amax').scatter_reduce(0, frame.n2, re, reduce='amax')
    return r


def flat_surface(x, r, kc, eps, z_s=0.0):
    """Contact force (N,3) on nodes at positions x (N,3) with radii r (N,)."""
    g = x[:, 2] - z_s - r
    fz = kc * eps * F.softplus(-g / eps)
    return torch.stack((torch.zeros_like(fz), torch.zeros_like(fz), fz), dim=1)


def flat_surface_potential(x, r, kc, eps, z_s=0.0):
    """Exact potential of flat_surface (diagnostics only, not differentiable):
    Phi(g) = kc eps^2 * int_{g/eps}^inf softplus(-s) ds = -kc eps^2 Li2(-exp(-g/eps)), Li2(-y) = spence(1+y)."""
    from scipy.special import spence
    g = (x[:, 2] - z_s - r).detach().cpu().numpy()
    return float(-(kc * eps**2 * spence(1 + np.exp(-g / eps))).sum())


# ------------------------------------------------------------------------------------------------
# Scalar gap laws for body-to-body contact (impactor <-> plate). law(g, gdot) -> force >= 0 (repulsive).
# g = gap [mm] (closed when g < 0 for penalty laws, g -> 0+ for the barrier), gdot = dg/dt (< 0 closing).
# All use the smooth penetration delta = eps*softplus(-g/eps) so they are C-infinity in g.

def smooth_penetration(g, eps):
    return eps * F.softplus(-g / eps)


def law_linear(k, eps):
    """Linear penalty k*delta: stiffness jumps 0 -> k at onset (force slope discontinuity up to eps)."""
    return lambda g, gd: k * smooth_penetration(g, eps)


def law_hertz(k, eps):
    """Hertz-type k*delta^1.5: stiffness 1.5 k delta^0.5 starts from zero, so force ramps in smoothly."""
    return lambda g, gd: k * smooth_penetration(g, eps) ** 1.5


def law_hunt_crossley(k, eps, e, v_ref, s=0.05):
    """Hunt-Crossley: k delta^1.5 (1 + alpha * closing rate), alpha = 3(1-e^2)/(4 v_ref) (Lankarani-Nikravesh).
    Damping scales with delta^1.5, so force is continuous at onset AND separation (a Kelvin-Voigt dashpot
    jumps at onset). Smooth positive part (scale s) removes the small tensile 'sticky' tail at separation."""
    alpha = 3 * (1 - e**2) / (4 * v_ref)
    return lambda g, gd: k * smooth_penetration(g, eps) ** 1.5 * s * F.softplus((1 - alpha * gd) / s)


def law_crit_damped(k, eps, mu, zeta=1.0, s_frac=0.01):
    """Hertz + damping tuned to the contact mode itself: c = 2 zeta sqrt(k_t mu), k_t = 1.5 k delta^0.5,
    mu = reduced mass of the two contacting bodies. zeta = 1 critically damps the impactor-plate relative
    mode, which kills the ringing of a light plate between a heavy impactor and a soft lattice.
    Damping ~ delta^0.25 -> 0 at onset (continuous). Smooth positive part 0.5(f + sqrt(f^2+s^2)) - s/2 with
    s = s_frac * elastic force: exactly 0 when separated, no sticky tension on opening, continuous."""
    def law(g, gd):
        d = smooth_penetration(g, eps)
        f_el = k * d**1.5
        c = 2 * zeta * (1.5 * k * mu) ** 0.5 * d.clamp_min(1e-300) ** 0.25   # sqrt(k_t mu), k_t = 1.5 k d^0.5
        f = f_el - c * gd                                # gd < 0 closing -> adds compression
        sc = s_frac * f_el + 1e-12
        return torch.clamp(0.5 * (f + torch.sqrt(f * f + sc * sc)) - 0.5 * sc, min=0.0)
    return law


def law_barrier(kappa, dhat):
    """IPC log barrier: F = -kappa b'(g), b = -(g-dhat)^2 ln(g/dhat) for 0 < g < dhat, 0 beyond.
    F and dF/dg are zero at g = dhat (C1 onset); F -> inf as g -> 0, so no penetration if dt resolves it."""
    def law(g, gd):
        gc = torch.clamp(g, 1e-9 * dhat, dhat)
        f = kappa * (2 * (gc - dhat) * torch.log(gc / dhat) + (gc - dhat) ** 2 / gc)
        return torch.where(g < dhat, f, torch.zeros_like(f))
    law.ccd = True                      # gap must never reach 0: time step is capped by the closing speed
    return law


def calibrate(kind, F_ref, delta_ref, eps=None, e=0.8, v_ref=6000.0, dhat=None, mu=None, zeta=1.0):
    """Build a law that carries F_ref at penetration delta_ref (barrier: at gap 0.1*dhat).
    law.scale = contact length scale the time step must resolve (see contact_dt_cap)."""
    eps = 0.1 * delta_ref if eps is None else eps
    if kind == 'linear':
        law = law_linear(F_ref / delta_ref, eps)
    elif kind == 'hertz':
        law = law_hertz(F_ref / delta_ref**1.5, eps)
    elif kind == 'hunt_crossley':
        law = law_hunt_crossley(F_ref / delta_ref**1.5, eps, e, v_ref)
    elif kind == 'crit_damped':
        law = law_crit_damped(F_ref / delta_ref**1.5, eps, mu, zeta)
    elif kind == 'barrier':
        dhat = 4 * delta_ref if dhat is None else dhat
        unit = law_barrier(1.0, dhat)(torch.tensor(0.1 * dhat), torch.tensor(0.0))
        law = law_barrier(F_ref / unit.item(), dhat)
        law.scale = dhat
        return law
    else:
        raise ValueError(kind)
    law.scale = delta_ref
    return law


def contact_dt_cap(law, g, gd, frac=0.1):
    """Kinematic cap: near/in contact the relative motion per step must stay below frac * law.scale,
    otherwise the force jumps from ~0 to a large value inside one step (stiffness-based dt cannot see
    it coming because the stiffness is ~0 just before contact). Barrier: also never close > 30% of the gap."""
    ell = law.scale
    cap = np.inf
    if g < 3 * ell + abs(gd) * 1e-4:                  # within reach of contact in the next ~0.1 ms
        cap = frac * ell / max(abs(gd), 1e-12)
    if getattr(law, 'ccd', False) and gd < 0:
        cap = min(cap, 0.3 * g / -gd)
    return cap


def law_tangent(law, g, gd):
    """(dF/dg, dF/dgdot) at a state, for the stable time step."""
    g = torch.as_tensor(g, dtype=torch.float64).detach().requires_grad_(True)
    gd = torch.as_tensor(gd, dtype=torch.float64).detach().requires_grad_(True)
    with torch.enable_grad():
        kg, kv = torch.autograd.grad(law(g, gd), (g, gd), allow_unused=True)
    return (0.0 if kg is None else -kg.item()), (0.0 if kv is None else -kv.item())


def law_potential(law, g, g_hi, n=400):
    """Elastic potential int_g^g_hi F(s, 0) ds (diagnostics)."""
    s = torch.linspace(float(g), float(g_hi), n, dtype=torch.float64)
    return torch.trapezoid(law(s, torch.zeros_like(s)), s).item() if g < g_hi else 0.0


# ------------------------------------------------------------------------------------------------
# Beam-to-beam (capsule) and beam-to-plate contact, potential based (conservative, smooth).
# Each element is a capsule: centreline segment + radius r_e. Gap g = |P(s*) - Q(t*)| - r_i - r_j.
# Potential phi(g) = (2/5) k delta^2.5, delta = eps*softplus(-g/eps)  ->  force = k delta^1.5 * sigmoid(-g/eps)
# along the normal (Hertz-like, C-infinity). Forces = -grad(sum phi) by autograd: energy-consistent.

def hertz_k_crossed_cylinders(r_i, r_j, E, nu):
    """Hertz contact k [N/mm^1.5] for two crossed cylinders: F = (4/3) E* sqrt(R) delta^1.5,
    R = sqrt(r_i r_j) (equivalent sphere for perpendicular cylinders), E* = E / (2 (1 - nu^2)) (same material).
    # ponytail: point-contact law for every pair; parallel beams really make line contact (stiffer)"""
    return (4.0 / 3.0) * E / (2 * (1 - nu**2)) * torch.sqrt(r_i * r_j)


def hertz_potential(g, k, eps):
    d = eps * F.softplus(-g / eps)
    return 0.4 * k * d**2.5


def segment_closest_params(p0, p1, q0, q1):
    """Clamped closest-point parameters (s, t) in [0,1] between segments p0p1 and q0q1 (batched, (n,3)).
    Ericson, Real-Time Collision Detection 5.1.9. Returned detached: by the envelope theorem the force only
    needs the closest points, and detaching avoids NaN/ill-conditioned derivatives for parallel segments."""
    with torch.no_grad():
        d1, d2, r = p1 - p0, q1 - q0, p0 - q0
        a, e, f = (d1 * d1).sum(1), (d2 * d2).sum(1), (d2 * r).sum(1)
        c, b = (d1 * r).sum(1), (d1 * d2).sum(1)
        den = a * e - b * b
        s = torch.where(den > 1e-12 * a * e, ((b * f - c * e) / den.clamp_min(1e-300)).clamp(0, 1), torch.zeros_like(a))
        t = (b * s + f) / e
        s = torch.where(t < 0, (-c / a).clamp(0, 1), torch.where(t > 1, ((b - c) / a).clamp(0, 1), s))
        t = t.clamp(0, 1)
    return s, t


class SelfContact:
    """Capsule contact between non-neighbouring elements of one frame.

    pairs: all element pairs farther than `exclude_hops` apart on the element graph (joints and elements next
    to them are always 'touching' and are excluded), minus pairs already overlapping in the reference state.
    Candidate list (pairs within `margin` of contact) is refreshed by the Verlet-list rule (any node moved more
    than margin/2 since the last refresh), so no pair can close unseen; forces are exact for every candidate
    (the choice of candidates is the only non-differentiable part).
    """

    def __init__(self, frame, r_e, E, nu, eps=None, exclude_hops=2, margin=1.0, every=10, beam_of_elem=None,
                 T_frac=0.05):
        """beam_of_elem: original beam id per element. Element pairs between the same two beams are merged into
        ONE contact by a smooth minimum of their gaps (log-sum-exp, temperature T = T_frac * eps). Without it a
        contact at a node shared by k elements is counted k times (4x for two beams crossing at their nodes),
        and the force would jump when the contact point slides from one element to the next."""
        self.fr, self.r = frame, r_e
        self.beam = torch.arange(frame.ne) if beam_of_elem is None else torch.as_tensor(beam_of_elem)
        self.k_pair = None
        self.E, self.nu = E, nu
        self.eps = 0.05 * r_e.min().item() if eps is None else eps
        self.margin, self.every, self.calls = margin, every, 0
        self.T = T_frac * self.eps
        ne = frame.ne
        # element adjacency via shared nodes; hop distance by repeated boolean products
        inc = torch.zeros(ne, frame.N, dtype=torch.bool)
        inc[torch.arange(ne), frame.n1] = True
        inc[torch.arange(ne), frame.n2] = True
        adj = (inc.double() @ inc.double().T) > 0
        near = adj.clone()
        for _ in range(exclude_hops - 1):
            near = near | ((near.double() @ adj.double()) > 0)
        i, j = torch.triu_indices(ne, ne, 1)
        keep = ~near[i, j]
        i, j = i[keep], j[keep]
        g0 = self._gap(frame.X, i, j)
        ok = g0 > 0
        self.n_initial_overlap = int((~ok).sum())
        self.pairs = torch.stack((i[ok], j[ok]), 1)
        self.k_all = hertz_k_crossed_cylinders(self.r[self.pairs[:, 0]], self.r[self.pairs[:, 1]], E, nu)
        self.active = None

    def _gap(self, x, i, j):
        n1, n2 = self.fr.n1, self.fr.n2
        p0, p1, q0, q1 = x[n1[i]], x[n2[i]], x[n1[j]], x[n2[j]]
        s, t = segment_closest_params(p0, p1, q0, q1)
        P = p0 + s[:, None] * (p1 - p0)
        Q = q0 + t[:, None] * (q1 - q0)
        dist = torch.sqrt(((P - Q) ** 2).sum(1) + 1e-30)
        return dist - self.r[i] - self.r[j]

    def refresh(self, x):
        with torch.no_grad():
            g = self._gap(x, self.pairs[:, 0], self.pairs[:, 1])
            self.x_ref = x.detach().clone()
        self.active = torch.nonzero(g < self.margin).flatten()

    def _stale(self, x):
        """Verlet-list rule: a pair outside `margin` at the last refresh cannot have closed unless some node
        moved more than margin/2 since then."""
        if self.active is None:
            return True
        with torch.no_grad():
            return (x.detach() - self.x_ref).norm(dim=1).max().item() > 0.5 * self.margin

    def dt_terms(self, x, v, m, scale, frac=0.1):
        """(omega, dt_cap) for the active pairs: omega from the Hertz tangent on the lightest node pair,
        cap so that near-contact pairs close at most frac*scale per step (contact resolved over steps)."""
        if self.active is None or len(self.active) == 0:
            return 0.0, float('inf')
        with torch.no_grad():
            pr = self.pairs[self.active]
            i, j = pr[:, 0], pr[:, 1]
            g = self._gap(x, i, j)
            kt = hertz_tangent(g, self.k_all[self.active], self.eps)
            n1, n2 = self.fr.n1, self.fr.n2
            mi = torch.minimum(m[n1[i]], m[n2[i]])
            mj = torch.minimum(m[n1[j]], m[n2[j]])
            w = torch.sqrt(kt * (1 / mi + 1 / mj)).max().item()
            near = g < 3 * scale
            if not near.any():
                return w, float('inf')
            vi = 0.5 * (v[n1[i]] + v[n2[i]])
            vj = 0.5 * (v[n1[j]] + v[n2[j]])
            vrel = (vi - vj).norm(dim=1)[near].max().item()
        return w, frac * scale / max(vrel, 1e-12)

    def energy(self, x):
        if self._stale(x):
            self.refresh(x)
        self.calls += 1
        if len(self.active) == 0:
            return x.sum() * 0.0, torch.zeros(0, dtype=x.dtype)
        pr = self.pairs[self.active]
        g = self._gap(x, pr[:, 0], pr[:, 1])
        # one contact per (beam, beam): smooth min of the element-pair gaps
        bi, bj = self.beam[pr[:, 0]], self.beam[pr[:, 1]]
        key = torch.minimum(bi, bj) * (int(self.beam.max()) + 1) + torch.maximum(bi, bj)
        grp, inv = torch.unique(key, return_inverse=True)
        z = -g / self.T
        zmax = torch.full((len(grp),), -torch.inf, dtype=g.dtype).scatter_reduce(0, inv, z.detach(), 'amax')
        ssum = torch.zeros(len(grp), dtype=g.dtype).index_add(0, inv, torch.exp(z - zmax[inv]))
        g_grp = -self.T * (zmax + torch.log(ssum))
        wts = torch.exp(z - zmax[inv]) / ssum[inv]           # softmax weights (closest pair dominates)
        k_grp = torch.zeros(len(grp), dtype=g.dtype).index_add(0, inv, wts * self.k_all[self.active])
        return hertz_potential(g_grp, k_grp, self.eps).sum(), g_grp


def plate_potential(z_gap, k, eps):
    """Node-to-plate potential with the same law; z_gap = distance from node surface to the plate face."""
    return hertz_potential(z_gap, k, eps).sum()


def potential_force(phi_fn, x):
    """Nodal forces -dPhi/dx for a scalar potential phi_fn(x) -> (Phi, extras). Keeps the outer autograd graph
    (create_graph) when x needs grad, like Frame.internal_force."""
    dx = torch.zeros_like(x, requires_grad=True)
    with torch.enable_grad():
        Phi, extra = phi_fn(x + dx)
        need = torch.is_grad_enabled() and x.requires_grad
        g, = torch.autograd.grad(Phi, dx, create_graph=need, allow_unused=True)
    f = torch.zeros_like(x) if g is None else -g
    return f, Phi, extra


def hertz_tangent(g, k, eps):
    """dF/d(penetration) of the Hertz-potential law (for the stable time step)."""
    d = eps * F.softplus(-g / eps)
    return 1.5 * k * d.clamp_min(1e-300) ** 0.5
