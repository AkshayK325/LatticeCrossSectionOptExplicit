"""Closed-form, differentiable joint stiffening from the strut angles and radii at each node (notes/18).

Same series (compliance) average and smooth zone length as latticeSetup.joint_factors (kept from there), with
three changes found from the Abaqus solids B3 / B5 / B6 (new FHS geometry):
  * every neighbour at a strut end is weighted by w = sin^p(theta): collinear neighbours count nothing, oblique ones
    little, orthogonal ones fully (an oblique diagonal at a plate corner must not lengthen the column's zone);
  * the end stiffening is k_end = 1 + mu a W, W = sum over neighbours (r_b / r_a)^2 w (added area relative to the own
    section; no constant k per topology);
  * the zone length is l_end = zl * smooth-max over neighbours of w (r_b + r_a cos theta) / sin theta, times
    (1 + b max(W - 2, 0)) (longer zone at high-valence nodes), capped at L/2.
Everything is a torch expression of the element radii, so design gradients (adjoint / autograd) see how a radius
change moves the stiffening. Parameters (a, p, zl, b) are fitted in abaqus/fit_rule.py; mu is the material factor
(1 for linear elastic, fitted for EPU 45)."""
import numpy as np
import torch

DEFAULT = dict(a=1.0, p=4.0, zl=1.0, b=0.0, mu=1.0, pz=None, gamma=1.0)


def rule_factors(lat, r_elem, a=1.0, p=4.0, zl=1.0, b=0.0, mu=1.0, pz=None, gamma=1.0, tau=0.02):
    """Per-element stiffness factors (ne,) >= 1, differentiable in r_elem (ne,) and in (a, p, zl, b, mu, pz, gamma) if tensors.
    p: angle exponent of the stiffness weight sin^p; pz: of the zone-length weight (default = p); a negative exponent weights
    ACUTE neighbours more (B7: the solid stiffens more the more acute the neighbour). gamma in [0, 1]: weight of a pair of struts
    that both LEAVE a plate at a plate node (B6: the oblique diagonal at a plate corner adds less than in-plate struts do)."""
    X, C = lat['X'], lat['conn']
    Xr, connr = lat['Xr'], lat['connr']
    boe_np = np.asarray(lat['beam_of_elem'])
    boe = torch.as_tensor(boe_np)
    nb = len(C)
    dt = r_elem.dtype
    cnt = torch.zeros(nb, dtype=dt).index_add(0, boe, torch.ones_like(r_elem))
    rb = torch.zeros(nb, dtype=dt).index_add(0, boe, r_elem) / cnt
    Lb = np.linalg.norm(X[C[:, 1]] - X[C[:, 0]], axis=1)
    ends, nbrs, cs, sn, gm = [], [], [], [], []
    zlo, zhi = float(X[:, 2].min()), float(X[:, 2].max())
    for b_, (i, j) in enumerate(C):
        for side, (node, other) in enumerate(((i, j), (j, i))):
            ta = (X[other] - X[node]) / Lb[b_]
            for c in np.nonzero((C[:, 0] == node) | (C[:, 1] == node))[0]:
                if c == b_:
                    continue
                o = C[c, 1] if C[c, 0] == node else C[c, 0]
                tb = (X[o] - X[node]) / np.linalg.norm(X[o] - X[node])
                ct = float(np.clip(ta @ tb, -1.0, 1.0))
                st = float(np.sqrt(max(1.0 - ct * ct, 0.0)))
                if st < 1e-6:                                       # collinear: the same strut continuing
                    continue
                at_plate = abs(X[node, 2] - zlo) < 1e-9 or abs(X[node, 2] - zhi) < 1e-9
                in_plane = lambda u: abs(u[2]) < 1e-9 * max(np.linalg.norm(u), 1.0)         # strut lies in the plate plane
                both_out = at_plate and not in_plane(X[other] - X[node]) and not in_plane(X[o] - X[node])
                ends.append(2 * b_ + side); nbrs.append(c); cs.append(ct); sn.append(st); gm.append(both_out)
    if not ends:
        return torch.ones(len(connr), dtype=dt)
    ends, nbrs = torch.as_tensor(ends), torch.as_tensor(nbrs)
    cs, sn = torch.tensor(cs, dtype=dt), torch.tensor(sn, dtype=dt)
    ra = rb[ends // 2]
    pz = p if pz is None else pz
    gmt = torch.as_tensor(gm)
    gfac = torch.where(gmt, torch.as_tensor(gamma, dtype=dt), torch.ones((), dtype=dt))
    w = sn ** p * gfac
    wz = sn ** pz * gfac
    q = rb[nbrs] / ra
    W = torch.zeros(2 * nb, dtype=dt).index_add(0, ends, q * q * w)                   # per strut end
    li = zl * wz * torch.clamp((rb[nbrs] + ra * cs) / sn, min=0.0)
    wt = tau * ra
    m0 = torch.full((2 * nb,), -torch.inf, dtype=dt).scatter_reduce(0, ends, li.detach(), 'amax')
    lse = torch.zeros(2 * nb, dtype=dt).index_add(0, ends, torch.exp((li - m0[ends]) / wt))
    wb = tau * torch.repeat_interleave(rb, 2)
    ell = m0 + wb * torch.log(lse.clamp_min(1e-300))                                  # smooth max over neighbours
    ell = torch.where(torch.isfinite(m0), ell, torch.zeros_like(ell))
    ell = ell * (1.0 + b * torch.clamp(W - 2.0, min=0.0))
    ell = torch.minimum(ell, torch.as_tensor(np.repeat(0.5 * Lb, 2), dtype=dt))
    k = 1.0 + mu * a * W
    P0, tb_ = X[C[boe_np, 0]], (X[C[boe_np, 1]] - X[C[boe_np, 0]]) / Lb[boe_np, None]
    s1 = ((Xr[connr[:, 0]] - P0) * tb_).sum(1)
    s2 = ((Xr[connr[:, 1]] - P0) * tb_).sum(1)
    lo = torch.as_tensor(np.minimum(s1, s2), dtype=dt)
    hi = torch.as_tensor(np.maximum(s1, s2), dtype=dt)
    le = hi - lo
    Lbt = torch.as_tensor(Lb[boe_np], dtype=dt)
    l0, l1 = ell[2 * boe], ell[2 * boe + 1]
    k0, k1 = k[2 * boe], k[2 * boe + 1]
    ov0 = torch.clamp(torch.minimum(hi, l0) - lo, min=0.0)
    ov1 = torch.clamp(hi - torch.maximum(lo, Lbt - l1), min=0.0)
    comp = (le - ov0 - ov1) + ov0 / k0 + ov1 / k1
    return le / comp
