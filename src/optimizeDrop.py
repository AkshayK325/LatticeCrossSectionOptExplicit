"""Drop/crush objectives as differentiable PyTorch functions of the design (exact adjoint gradients).

Design convention (as OptimizeFrame / optBeam.ipynb): the optimizer works on normalised rows x in [0, 1];
radii = min_x + (max_x - min_x) * x is applied INSIDE the objective (spec['x_range'] = (min_x, max_x)), so
dJ/dx already contains the denormalisation chain rule. Without 'x_range' x is the radius itself.

value_and_grad(x, spec): nominal run at x (adaptive dt -> schedule, frozen rig constants), then the
checkpointed discrete adjoint for dJ/dx. DropObjective wraps it as a torch.autograd.Function so any upstream
PyTorch map (e.g. the VAE section model) can feed it: J = DropObjective.apply(x, spec).
"""
import torch

from adjoint import DropStepper, checkpointed_adjoint
from dropGrad import nominal, radii
from latticeSetup import drop_rig


def denormalize(x, spec):
    lo_hi = spec.get('x_range')
    return x if lo_hi is None else lo_hi[0] + (lo_hi[1] - lo_hi[0]) * x


def value_and_grad(x, spec):
    """spec: dict(lat, symType, mat, rig_kw, T, terms(rig) -> per-step terms, outer(rig) -> J(S, theta), K,
    x_range=(min_x, max_x) optional). x: normalised design rows if x_range is given."""
    lat, sym, mat = spec['lat'], spec['symType'], spec['mat']
    kw = dict(spec['rig_kw'])
    with torch.no_grad():
        rig0, sched = nominal(lat, denormalize(x.detach(), spec), sym, mat, T=spec['T'], **kw)
    xx = x.detach().clone().requires_grad_(True)
    kw.pop('law', None)
    with torch.enable_grad():                 # autograd.Function.forward runs with grad disabled
        rig = drop_rig(lat, radii(denormalize(xx, spec), sym, lat), rig0['law'], mat=mat, freeze=rig0, **kw)
    with torch.enable_grad():
        stepper = DropStepper(rig, sched, spec['terms'](rig))
    J, (g,) = checkpointed_adjoint(stepper, [xx], K=spec.get('K'), outer=spec['outer'](rig))
    return J, g


class DropObjective(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spec):
        J, g = value_and_grad(x, spec)
        ctx.save_for_backward(g)
        return torch.tensor(J, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, gout):
        g, = ctx.saved_tensors
        return gout * g, None


def response_curve(x, spec):
    """(stroke, impactor force) loading branch of the drop at design x (for targets and plots)."""
    from latticeSetup import run_drop
    kw = dict(spec['rig_kw'])
    with torch.no_grad():
        rig = drop_rig(spec['lat'], radii(denormalize(x, spec), spec['symType'], spec['lat']), mat=spec['mat'], **kw)
        h = run_drop(rig, T=spec['T'])
    d, F = -torch.as_tensor(h['z_plate']), torch.as_tensor(h['F_imp'])
    return d, F, h


def loading_branch(d, F):
    """Strictly increasing-stroke part (for interpolation targets)."""
    keep, dmax = [], -float('inf')
    for i in range(len(d)):
        if d[i] > dmax + 1e-9:
            keep.append(i)
            dmax = d[i].item()
    keep = torch.tensor(keep)
    return d[keep], F[keep]
