"""Shell element checks (analytic): rigid-body invariance, membrane patch, pure bending incl. offset, cantilever."""
import numpy as np
import torch

from shellFE import Shell, vertex_normals
from explicitFE import rodrigues

torch.set_default_dtype(torch.float64)
E_, NU, T_ = 2300.0, 0.38, 2.0


def plate(nx=6, ny=4, Lx=30.0, Ly=20.0, jitter=0.0, seed=0):
    xs, ys = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1), indexing='ij')
    X = np.stack((xs.ravel(), ys.ravel(), 0 * xs.ravel()), 1)
    if jitter:
        rng = np.random.default_rng(seed)
        inner = (X[:, 0] > 0) & (X[:, 0] < Lx) & (X[:, 1] > 0) & (X[:, 1] < Ly)
        X[inner, :2] += jitter * rng.standard_normal((inner.sum(), 2))
    idx = lambda i, j: i * (ny + 1) + j
    tri = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tri += [[a, b, c], [a, c, d]]
    return X, np.array(tri)


def make(X, tri, o=0.0, t=T_):
    D = np.tile([0.0, 0.0, 1.0], (len(X), 1))
    return Shell(X, tri, D, t, E_, NU, 1.2e-9, offset=o)


def test_rigid_motion_zero_energy():
    X, tri = plate(jitter=0.5)
    sh = make(X, tri, o=-1.0)
    N = len(X)
    w = torch.tensor([0.3, -0.5, 0.8])
    Q = rodrigues(w[None])[0]
    x = (torch.as_tensor(X) @ Q.T) + torch.tensor([5.0, -2.0, 7.0])
    R = (Q @ torch.eye(3)).expand(N, 3, 3).contiguous()
    W = sh.element_energy(x, R).sum()
    assert W.abs() < 1e-14, W
    fu, fth, _ = sh.internal_force(x - sh.X, R)
    assert fu.abs().max() < 1e-10 and fth.abs().max() < 1e-10


def test_membrane_patch_uniform_stretch():
    X, tri = plate(jitter=0.6)
    sh = make(X, tri)
    eps = 1e-3
    x = torch.as_tensor(X).clone()
    x[:, 0] *= 1 + eps
    R = torch.eye(3).expand(len(X), 3, 3).contiguous()
    W = sh.element_energy(x, R).sum().item()
    A = sh.area.sum().item()
    Eg = eps + 0.5 * eps**2
    exact = 0.5 * sh.Cm * Eg**2 * A
    assert abs(W - exact) < 1e-6 * exact


def _bend(sh, X, kap, o):
    """Circular bending of the mid-surface about y with curvature kap: reference nodes on the plane at distance -o
    below... returns energy."""
    x0 = torch.as_tensor(X)
    N = len(X)
    s = x0[:, 0]
    th = kap * s
    # mid-surface point at arc s: (sin th/kap, 0, (1-cos th)/kap); director d = (-sin th, 0, cos th)... bending about y
    xm = torch.stack((torch.sin(th) / kap, x0[:, 1], (1 - torch.cos(th)) / kap), 1)
    d = torch.stack((-torch.sin(th), torch.zeros_like(th), torch.cos(th)), 1)
    x = xm - o * d
    R = torch.stack([rodrigues(torch.tensor([0.0, -float(a), 0.0])[None])[0] for a in th])
    return sh.element_energy(x, R).sum().item()


def test_pure_bending_with_and_without_offset():
    X, tri = plate(nx=12, ny=6)
    kap = 1e-3
    A = 30.0 * 20.0
    Db = E_ * T_**3 / (12 * (1 - NU**2))
    for o in (0.0, -1.0, 0.7):
        sh = make(X, tri, o=o)
        W = _bend(sh, X, kap, o)
        exact = 0.5 * Db * kap**2 * A
        assert abs(W - exact) < 0.03 * exact, (o, W, exact)


def test_cantilever_tip_load():
    """Clamped-free strip (b >> ?) : narrow plate treated as a beam with plane-stress modulus E/(1-nu^2) for a wide
    plate; here a strip of width b = 8 with free sides: Euler-Bernoulli + shear."""
    L, b, t = 40.0, 8.0, T_
    X, tri = plate(nx=20, ny=4, Lx=L, Ly=b)
    sh = make(X, tri)
    N = len(X)
    P = 1.0
    fixed = np.where(X[:, 0] < 1e-9)[0]
    free = np.setdiff1d(np.arange(N), fixed)
    tip = np.where(np.isclose(X[:, 0], L))[0]
    # solve K q = f with the dense Hessian of the energy (small problem) over dofs (u, rotation vector) of free nodes
    def Wfun(q):
        qq = q.reshape(N, 6)
        R = rodrigues(qq[:, 3:])
        return sh.element_energy(sh.X + qq[:, :3], R).sum()
    q0 = torch.zeros(N * 6)
    H = torch.autograd.functional.hessian(Wfun, q0)
    dof = np.concatenate([np.arange(6 * n, 6 * n + 6) for n in free])
    f = torch.zeros(N * 6)
    for n in tip:
        f[6 * n + 2] = P / len(tip)
    Hf = H[dof][:, dof] + 1e-9 * torch.eye(len(dof))
    q = torch.zeros(N * 6)
    q[dof] = torch.linalg.solve(Hf, f[dof])
    defl = q[[6 * n + 2 for n in tip]].mean().item()
    Ieff = b * t**3 / 12
    EIs = E_ * Ieff
    kGA = (5 / 6) * E_ / (2 * (1 + NU)) * b * t
    exact = P * L**3 / (3 * EIs) + P * L / kGA
    assert abs(defl - exact) < 0.10 * exact, (defl, exact)
