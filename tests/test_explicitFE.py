"""Day 2: stateless corotational beam (src/explicitFE.py) against the legacy frameFE3 stiffness."""
import math

import numpy as np
import pytest
import torch

from explicitFE import Frame, rodrigues, lumped_mass, stable_dt

torch.set_default_dtype(torch.float64)
E, NU = 6.7, 0.3
G = E / (2 * (1 + NU))


def sections(ne, Iy=0.8, Iz=2.3):
    """Non-circular on purpose (Iy != Iz) so a swapped local axis would show up."""
    A = torch.tensor([[3.0, 2.6, 2.4]]).repeat(ne, 1)
    I = torch.tensor([[1.1, Iy, Iz]]).repeat(ne, 1)
    return A, I


def legacy_K(X, conn, A, I, elemSize):
    from frameFE3 import FrameFE
    from boundaryCondition import boundaryConditionEx
    bc = boundaryConditionEx('1fz', X, conn)
    fe = FrameFE({'nodeXY': X, 'connectivity': conn, 'numElemsPerBeam': 1, 'elemSize': elemSize, 'ElemType': 'TS'},
                 bc, {'Type': 'Linear', 'solver': 'pypardiso', 'matrixType': 'Sparse', 'Section': 'circle', 'nu': NU})
    u0 = torch.zeros(fe.ndof)
    K, _ = fe.assembleKNonLin(u0, u0, torch.tensor([E]), A[:1].repeat(fe.numEle, 1), I[:1].repeat(fe.numEle, 1))
    return K.to_dense(), fe.nodeXY, fe.connectivity


def our_K(X, conn, A, I):
    fr = Frame(X, conn)
    law = lambda d: fr.elastic_law(d, A, I, E, G)

    def f(q):
        q = q.reshape(fr.N, 6)
        fu, fth, _, _ = fr.internal_force(q[:, :3], rodrigues(q[:, 3:]), law)
        return torch.cat((fu, fth), dim=1).reshape(-1)
    return torch.autograd.functional.jacobian(f, torch.zeros(6 * fr.N))


GEOMS = {
    'skew_beam': (np.array([[0.0, 0, 0], [7.0, -4.0, 5.0]]), np.array([[0, 1]]), 3.0),
    'bent_3_beams': (np.array([0, 0, 0, 0, 0, 30, 30, 0, 30, 30, 30, 30.0]).reshape(-1, 3), np.array([[0, 1], [1, 2], [2, 3]]), 10.0),
}


@pytest.mark.parametrize('name', GEOMS)
def test_small_strain_tangent_equals_legacy_stiffness(name):
    X, conn, h = GEOMS[name]
    Kleg, Xr, cr = legacy_K(X, conn, *sections(1), h)
    Kour = our_K(Xr, cr, *sections(len(cr)))
    scale = Kleg.abs().max()
    assert (Kour - Kleg).abs().max() / scale < 1e-9, (Kour - Kleg).abs().max() / scale


def random_rotation(angle, seed=0):
    ax = torch.randn(3, generator=torch.Generator().manual_seed(seed))
    return rodrigues((angle * ax / ax.norm())[None])[0]


@pytest.mark.parametrize('angle', [math.pi / 2, 2.5, 3.0])
def test_rigid_body_motion_is_force_free(angle):
    X, conn, _ = GEOMS['bent_3_beams']
    fr = Frame(X, conn)
    A, I = sections(fr.ne)
    Q = random_rotation(angle)
    u = fr.X @ Q.T + torch.tensor([3.0, -1.0, 2.0]) - fr.X
    R = Q.expand(fr.N, 3, 3).contiguous()
    fu, fth, d, _ = fr.internal_force(u, R, lambda d: fr.elastic_law(d, A, I, E, G))
    assert d.abs().max() < 1e-12
    assert max(fu.abs().max(), fth.abs().max()) < 1e-10


def test_uniform_stretch_gives_axial_force_only():
    X, conn, _ = GEOMS['skew_beam']
    fr = Frame(X, conn)
    A, I = sections(1)
    eps = 0.01
    u = eps * (fr.X - fr.X[0])
    R = torch.eye(3).expand(fr.N, 3, 3).contiguous()
    fu, fth, d, f_loc = fr.internal_force(u, R, lambda d: fr.elastic_law(d, A, I, E, G))
    assert torch.allclose(f_loc[0, 0], torch.tensor(E * 3.0 * eps))
    assert f_loc[0, 1:].abs().max() < 1e-12
    e1 = (fr.X[1] - fr.X[0]) / fr.L0[0]
    assert torch.allclose(fu[1], E * 3.0 * eps * e1) and torch.allclose(fu[0], -fu[1])


def test_force_differentiable_wrt_section_and_state():
    """d/dA and d/du of a scalar of f_int at a deformed, rotated state: autograd vs central FD."""
    X, conn, _ = GEOMS['bent_3_beams']
    fr = Frame(X, conn)
    g = torch.Generator().manual_seed(1)
    u0 = 0.5 * torch.randn(fr.N, 3, generator=g)
    th = 0.3 * torch.randn(fr.N, 3, generator=g)
    w = torch.randn(fr.N, 6, generator=g)

    def scalar(Ascale, u):
        A, I = sections(fr.ne)
        A = A * Ascale
        fu, fth, _, _ = fr.internal_force(u, rodrigues(th), lambda d: fr.elastic_law(d, A, I, E, G))
        return (torch.cat((fu, fth), 1) * w).sum()

    a = torch.tensor(1.0, requires_grad=True)
    u = u0.clone().requires_grad_(True)
    ga, gu = torch.autograd.grad(scalar(a, u), (a, u))
    h = 1e-6
    fd_a = (scalar(torch.tensor(1 + h), u0) - scalar(torch.tensor(1 - h), u0)) / (2 * h)
    assert abs(ga - fd_a) < 1e-6 * abs(fd_a)
    for i, k in ((0, 0), (2, 1), (3, 2)):
        up, um = u0.clone(), u0.clone()
        up[i, k] += h
        um[i, k] -= h
        fd = (scalar(torch.tensor(1.0), up) - scalar(torch.tensor(1.0), um)) / (2 * h)
        assert abs(gu[i, k] - fd) < 1e-6 * max(abs(fd), 1.0), (gu[i, k], fd)


def test_stable_dt_matches_dense_eigenvalue():
    X, conn, _ = GEOMS['bent_3_beams']
    from mesher import MeshFrame
    Xr, cr = MeshFrame().refineFrameMeshElemSize(X, conn, 10.0)
    fr = Frame(Xr, cr)
    A, I = sections(fr.ne)
    rho = 1.24e-9
    m, j = lumped_mass(fr, A, I, rho)
    law = lambda d: fr.elastic_law(d, A, I, E, G)
    dt, w = stable_dt(fr, law, m, j, courant=1.0, iters=300)
    K = our_K(Xr, cr, A, I)
    s = torch.stack((m, m, m, j, j, j), 1).reshape(-1) ** -0.5
    w_dense = torch.linalg.eigvalsh(s[:, None] * K * s[None, :]).max().sqrt().item()
    assert abs(w - w_dense) / w_dense < 1e-3
    # with rotary-inertia scaling the axial wave governs: dt ~ L/c (corner nodes are heavier -> slightly above)
    dt_axial = fr.L0.min().item() / math.sqrt(E / rho)
    assert 1.0 <= dt / dt_axial < 1.1, dt / dt_axial
    # without scaling the rotational DOFs are stiffer/lighter and cut the step
    m2, j2 = lumped_mass(fr, A, I, rho, rot_scale=False)
    dt2, _ = stable_dt(fr, law, m2, j2, courant=1.0, iters=300)
    assert dt2 < dt
