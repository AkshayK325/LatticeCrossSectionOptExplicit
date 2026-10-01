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
