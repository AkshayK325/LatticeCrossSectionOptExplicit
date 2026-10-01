"""VAE cross-section design map, moved out of optBeam.ipynb (cells 9-10).

Design row x = [z0, z1, k, theta] in [0, 1]: latent point (mapped into a rectangle of the
VAE latent space), scale factor k, rotation theta*pi/2.
Use functools.partial(propPredict, net=splineNet) where a one-argument callable is needed.
"""
import torch


def map_to_rectangle(z, add_thickness=False):
    """Map z in [0,1]^2 (N, 2) into the latent-space rectangle used for optimization."""
    if add_thickness:
        h, k, width, height = 0.0, 1.25, 5.0, 2.5
    else:
        h, k, width, height = 1.0, 0.0, 2.0, 5.0
    zc = z - 0.5
    return torch.stack((h + zc[:, 0] * width, k + zc[:, 1] * height), dim=1)


def _latent(z_in, net, add_thickness, plot_invalid_shapes):
    z_in = z_in.to(dtype=net.target_dtype)
    if not plot_invalid_shapes:
        return map_to_rectangle(z_in, add_thickness)
    # negative inputs mirror into the invalid region (plotting only)
    neg = torch.any(z_in < 0, dim=1, keepdim=True)
    z = map_to_rectangle(z_in.abs(), add_thickness)
    return torch.where(neg, -z, z)


def propPredict(x_in, net, add_thickness=False, plot_invalid_shapes=False):
    """(N, 4) design -> (N, 5) [Sf, A, J, Iy, Iz] after scaling by k and rotating by theta."""
    k = x_in[:, 2] + 0.001
    theta = x_in[:, 3] * torch.pi / 2
    props = net.getProperties(net.vaeNet.predict(_latent(x_in[:, 0:2], net, add_thickness, plot_invalid_shapes)))
    Sf, A, J, Iy, Iz = props[:, -5:].unbind(dim=1)
    A, J, Iy, Iz = A * k**2, J * k**4, Iy * k**4, Iz * k**4
    c2, s2 = torch.cos(theta)**2, torch.sin(theta)**2
    out = torch.stack([Sf, A, J, Iy * c2 + Iz * s2, Iy * s2 + Iz * c2], dim=1)
    return out.to(dtype=x_in.dtype)


def shapePredict(x_in, net, add_thickness=False, plot_invalid_shapes=False):
    """(N, 4) design -> (B-spline control points scaled by k (N, n_cp), theta (N,))."""
    k = x_in[:, 2]
    theta = x_in[:, 3] * torch.pi / 2
    props = net.getProperties(net.vaeNet.predict(_latent(x_in[:, 0:2], net, add_thickness, plot_invalid_shapes)))
    return (props[:, 0:-6] * k.unsqueeze(1)).to(dtype=x_in.dtype), theta
