"""Drop/crush objectives as differentiable PyTorch functions of the design (exact adjoint gradients).

value_and_grad(x, spec): nominal run at x (adaptive dt -> schedule, frozen rig constants), then the
checkpointed discrete adjoint for dJ/dx. DropObjective wraps it as a torch.autograd.Function so any upstream
PyTorch map (e.g. the VAE section model) can feed it: J = DropObjective.apply(x, spec).
"""
import torch

from adjoint import DropStepper, checkpointed_adjoint
from dropGrad import nominal, radii
from latticeSetup import drop_rig


def value_and_grad(x, spec):
    """spec: dict(lat, symType, mat, rig_kw, T, terms(rig) -> per-step terms, outer(rig) -> J(S, theta), K)."""
    lat, sym, mat = spec['lat'], spec['symType'], spec['mat']
    kw = dict(spec['rig_kw'])
    with torch.no_grad():
        rig0, sched = nominal(lat, x.detach(), sym, mat, T=spec['T'], **kw)
    xx = x.detach().clone().requires_grad_(True)
    kw.pop('law', None)
    with torch.enable_grad():                 # autograd.Function.forward runs with grad disabled
        rig = drop_rig(lat, radii(xx, sym, lat), rig0['law'], mat=mat, freeze=rig0, **kw)
    J, (g,) = checkpointed_adjoint(DropStepper(rig, sched, spec['terms'](rig)), [xx], K=spec.get('K'),
                                   outer=spec['outer'](rig))
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
        rig = drop_rig(spec['lat'], radii(x, spec['symType'], spec['lat']), mat=spec['mat'], **kw)
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
