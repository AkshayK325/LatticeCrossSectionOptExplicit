"""Design gradients through the drop test (plan 3, Day 5).

design rows x (symType) -> per-element radii -> drop_rig(freeze=nominal) -> explicit simulation with the
nominal time-step schedule replayed -> scalar J. The objective is deliberately not chosen yet (user,
2026-10-01): `placeholder_J` exists only to verify gradients.
"""
import torch

from explicitSolve import simulate
from latticeSetup import drop_rig, element_values


def radii(x, symType, lat):
    return element_values(x, symType, lat)[:, 0]


def nominal(lat, x, symType, mat, T, **rig_kw):
    """Nominal rig + its run (adaptive dt). Returns (rig, schedule) for frozen, replayed evaluations."""
    from latticeSetup import run_drop
    with torch.no_grad():
        rig = drop_rig(lat, radii(x, symType, lat), mat=mat, **rig_kw)
        h = run_drop(rig, T=T)
    return rig, [float(d) for d in h['dt']]


def placeholder_J(F_bottom, dt):
    """Gradient-verification placeholder ONLY (objective deferred): time integral of the squared bottom force."""
    return F_bottom**2 * dt


def evaluate(lat, x, symType, mat, rig0, schedule, n_steps=None, J_fn=placeholder_J, **rig_kw):
    """J(x) with all rig constants frozen at rig0 and dt replayed; differentiable w.r.t. x (autograd)."""
    kw = {k: v for k, v in rig_kw.items() if k != 'law'}
    rig = drop_rig(lat, radii(x, symType, lat), rig0['law'], mat=mat, freeze=rig0, **kw)
    seq = schedule if n_steps is None else schedule[:n_steps]

    def dt_fn(n, u, R, v=None):
        rig['dt_now']['h'] = seq[n]                       # friction regularisation uses the step size
        return seq[n]

    acc = {'J': torch.zeros((), dtype=rig['m'].dtype)}

    def cb(n, t, state, fint):
        if n < len(seq):
            acc['J'] = acc['J'] + J_fn(fint[0][rig['bot'], 2].sum(), seq[n])
        return {}

    simulate(rig['fr'], rig['beam_law'], rig['m'], rig['j'], dt_fn, len(seq), v0=rig['v_init'],
             fix_u=rig['fix_u'], fix_r=rig['fix_r'], contact=rig['contact'], tie=rig['tie'], callback=cb)
    return acc['J']
