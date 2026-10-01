"""DropObjective (autograd.Function) gives the adjoint gradient inside ordinary PyTorch code."""
import torch

from latticeSetup import example3, n_design_rows
from material import project_material
from objectives import plateau_target, make_terms, fd_mse_outer
from optimizeDrop import DropObjective, value_and_grad

torch.set_default_dtype(torch.float64)


def test_autograd_function_matches_value_and_grad():
    lat = example3(elemSize=3.0)
    target = plateau_target(60.0, 0.5, 8.0)
    spec = dict(lat=lat, symType='Midplane', mat=project_material(), T=4e-4,
                rig_kw=dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving',
                            gap0=0.1, energy_J=0.75),
                terms=lambda rig: make_terms(rig, target), outer=lambda rig: fd_mse_outer(target))
    x = torch.full((n_design_rows('Midplane', lat), 1), 0.8, requires_grad=True)
    J = DropObjective.apply(x * 1.0, spec)                           # upstream op: autograd chains through it
    J.backward()
    J2, g2 = value_and_grad(x.detach(), spec)
    assert abs(J.item() - J2) <= 1e-12 * abs(J2)
    assert torch.allclose(x.grad, g2, rtol=1e-10, atol=1e-14), (x.grad - g2).abs().max()


def test_gradient_independent_of_ambient_grad_mode():
    """Regression: building the stepper under no_grad used to detach the masses (10% gradient error)."""
    lat = example3(elemSize=3.0)
    target = plateau_target(60.0, 0.5, 8.0)
    spec = dict(lat=lat, symType='Midplane', mat=project_material(), T=4e-4,
                rig_kw=dict(law={'kind': 'crit_damped', 'F_ref': 2000.0, 'delta_ref': 0.2}, start='co_moving',
                            gap0=0.1, energy_J=0.75),
                terms=lambda rig: make_terms(rig, target), outer=lambda rig: fd_mse_outer(target))
    x = torch.full((n_design_rows('Midplane', lat), 1), 0.8)
    _, g1 = value_and_grad(x, spec)
    with torch.no_grad():
        _, g2 = value_and_grad(x, spec)
    assert torch.allclose(g1, g2, rtol=1e-12, atol=1e-16)
