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
