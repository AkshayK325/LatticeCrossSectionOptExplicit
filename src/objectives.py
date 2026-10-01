"""Objectives for the drop/crush rig, in the additive form the checkpointed adjoint needs:
per-step terms ell_n (vector) -> S = sum_n ell_n -> J = outer(S, theta).

Response curve: F = impactor-plate force (resisting, N), delta = top-plate stroke (mm, positive in compression).
Per step (trapezoid in stroke):  dd = delta_out - delta_in,  Fm = (F_in + F_out)/2,  dm = (delta_in + delta_out)/2.
    S[0] = sum (Fm - F*(dm))^2 * relu(dd)    target-curve misfit along the loading branch
    S[1] = sum  Fm * relu(dd)                 loading area (energy absorbed up to max stroke)
    S[2] = sum  Fm * dd                       net loop area (energy dissipated = loading - unloading)
Objectives:
    fd_mse   : S[0] / int F*^2 d(delta)        normalised mean-squared F-delta misfit (inverse design)
    sea      : -S[1] / m   or  -S[2] / m       specific energy absorption (J/g), maximised -> minimise -SEA
Note (fixed-energy drop): the loading area is ~ the impact energy whatever the design (the impactor stops),
so 'loading' SEA reduces to 1/mass there; use mode='dissipated' in a drop, or 'loading' in a constant-velocity
crush to a set stroke (the standard SEA test; latticeSetup.drop_rig(..., crush=True)).
"""
import torch


def plateau_target(F_plateau, d_rise, d_end, F_end=None, n=200):
    """'Ideal absorber' trapezoid: linear rise to F_plateau over d_rise, flat to d_end (optional end force)."""
    d = torch.linspace(0.0, d_end, n)
    F = torch.where(d < d_rise, F_plateau * d / d_rise, torch.full_like(d, F_plateau))
    if F_end is not None:
        F = F + (F_end - F_plateau) * torch.clamp((d - d_rise) / (d_end - d_rise), 0, 1)
    return d, F


def interp(x, xp, fp):
    """Piecewise-linear interpolation, differentiable in x (and fp); constant extrapolation."""
    x_ = x.detach().clamp(xp[0], xp[-1])
    i = torch.clamp(torch.searchsorted(xp, x_.reshape(1)), 1, len(xp) - 1)[0]
    t = (x.clamp(xp[0], xp[-1]) - xp[i - 1]) / (xp[i] - xp[i - 1])
    return fp[i - 1] + t * (fp[i] - fp[i - 1])


def make_terms(rig, target=None):
    """Per-step terms ell = [misfit, loading area, net area] for DropStepper (J_fn)."""
    t0 = int(rig['top'][0])
    d_t, F_t = (None, None) if target is None else (torch.as_tensor(target[0]), torch.as_tensor(target[1]))

    def terms(zin, zout, h):
        di, do = -zin['u'][t0, 2], -zout['u'][t0, 2]
        dd = do - di
        Fm = 0.5 * (zin['Fi'] + zout['Fi'])
        load = torch.relu(dd)
        mis = (Fm - interp(0.5 * (di + do), d_t, F_t)) ** 2 * load if d_t is not None else Fm * 0.0
        return torch.stack((mis, Fm * load, Fm * dd))
    return terms


def lattice_mass(rig, rho):
    """Lattice mass [t] = rho * sum A L0 (differentiable in the section areas)."""
    return rho * (rig['A'][:, 0] * rig['fr'].L0).sum()


def fd_mse_outer(target):
    d_t, F_t = torch.as_tensor(target[0]), torch.as_tensor(target[1])
    norm = torch.trapezoid(F_t**2, d_t)
    return lambda S, theta: S[0] / norm


def sea_outer(rig, rho, mode='dissipated'):
    """-SEA in J/g (area in N*mm = mJ; mass in t -> g)."""
    k = 1 if mode == 'loading' else 2
    return lambda S, theta: -(S[k] * 1e-3) / (lattice_mass(rig, rho) * 1e6)
