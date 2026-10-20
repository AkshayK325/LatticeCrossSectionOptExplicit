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

DEFAULT = dict(a=1.0, p=4.0, zl=1.0, b=0.0, mu=1.0, pz=None, gamma=1.0, cap=0.5, wsat=None, lmin=0.0, taper=0.0, plate_scale=1.0, short=0.0, tail=0.0, tail_len=2.0, tail_pow=1.0, ang_cos=False, tail_mult=False, short_gain=0.0, short_l0=5.5, smooth=0.0)


def _sp(x, w):
    """Smooth positive part w log(1 + exp(x / w)) (>= 0, -> max(x, 0) as w -> 0)."""
    return w * torch.nn.functional.softplus(x / w)


def _smax(a, b, w):
    m = torch.maximum(a, b).detach()
    return m + w * torch.log(torch.exp((a - m) / w) + torch.exp((b - m) / w))


def _smin(a, b, w):
    return -_smax(-a, -b, w)

def _rule_core(lat, r_elem, a=1.0, p=4.0, zl=1.0, b=0.0, mu=1.0, pz=None, gamma=1.0, cap=0.5, wsat=None, lmin=0.0, taper=0.0, plate_scale=1.0, short=0.0, tail=0.0, tail_len=2.0, tail_pow=1.0, ang_cos=False, tail_mult=False, smooth=0.0, tau=0.02):
    """Per-element stiffness factors (ne,) >= 1, differentiable in the element radii r_elem (ne,) and in every numeric parameter (pass tensors to fit through the solver).

    PRODUCTION parameters (the shipped PRESETS use only these + short_gain / short_l0 + smooth):
      a, zl, b, p, gamma, wsat, mu, smooth.
        p / pz: angle exponent of the stiffness / zone-length weight sin^p (negative: acute neighbours weigh more, B7); gamma in [0, 1]: weight of a pair of struts that both LEAVE a plate
        at a plate node (B6); wsat: the neighbour sum W saturates smoothly at wsat (B9); b: zone growth with W above 2 (centre nodes, B8); mu: material factor (1: not transferable, notes/18 9);
        smooth > 0: every kink (clamps, min / max, the zone edge inside an element) is replaced by a smooth version of that width, so autograd equals finite differences at any radii
        (tests/test_joint_rule_smooth.py); the zone edge is blended in the COMPLIANCE domain, which reproduces the hard rule to 0.2 %.
    EXPERIMENTAL options (tested and NOT adopted, kept for the record, notes/18 sections 14-17): cap, lmin, taper, plate_scale, short, tail, tail_len, tail_pow, tail_mult, ang_cos."""
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
    plate_end = np.zeros(2 * nb, bool)                                  # strut end lies on a plate
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
                plate_end[2 * b_ + side] = at_plate
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
    if ang_cos:           # ((1 + cos theta) / 2)^p: 1 for acute pairs, 0.5^p at 90 deg, -> 0 for near-collinear (obtuse pairs of kinks add little)
        ca = ((1.0 + cs) / 2.0).clamp_min(1e-6)
        w, wz = ca ** p * gfac, ca ** pz * gfac
    else:
        w = sn ** p * gfac
        wz = sn ** pz * gfac
    q = rb[nbrs] / ra
    W = torch.zeros(2 * nb, dtype=dt).index_add(0, ends, q * q * w)                   # per strut end
    geo = (rb[nbrs] + ra * cs) / sn
    if smooth:            # every kink replaced by a smooth version of width `smooth` (x own radius): exact, well-conditioned design gradients everywhere
        geo = _sp(geo, smooth * ra)
        if lmin:
            geo = _smax(geo, lmin * ra, smooth * ra)
    else:
        geo = torch.clamp(geo, min=0.0)
        if lmin:
            geo = torch.maximum(geo, lmin * ra)
    li = zl * wz * geo
    wt = tau * ra
    m0 = torch.full((2 * nb,), -torch.inf, dtype=dt).scatter_reduce(0, ends, li.detach(), 'amax')
    lse = torch.zeros(2 * nb, dtype=dt).index_add(0, ends, torch.exp((li - m0[ends]) / wt))
    wb = tau * torch.repeat_interleave(rb, 2)
    ell = m0 + wb * torch.log(lse.clamp_min(1e-300))                                  # smooth max over neighbours
    ell = torch.where(torch.isfinite(m0), ell, torch.zeros_like(ell))
    ell = ell * (1.0 + b * (_sp(W - 2.0, smooth) if smooth else torch.clamp(W - 2.0, min=0.0)))
    if short:             # short struts (L / r < 5) are almost all joint: their zones grow (B10: C4 short strut L / r = 3.3)
        ell = ell * (1.0 + short * torch.clamp(5.0 - torch.as_tensor(np.repeat(Lb, 2), dtype=dt) / torch.repeat_interleave(rb, 2), min=0.0))
    capL = torch.as_tensor(np.repeat(cap * Lb, 2), dtype=dt)
    ell = _smin(ell, capL, smooth * torch.repeat_interleave(rb, 2)) if smooth else torch.minimum(ell, capL)
    if wsat is not None:
        W = wsat * torch.tanh(W / wsat)
    k = 1.0 + mu * a * W
    k = 1.0 + (k - 1.0) * torch.where(torch.as_tensor(plate_end), torch.as_tensor(plate_scale, dtype=dt), torch.ones((), dtype=dt))
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
    if tail > 0 or smooth:  # step zone (smoothed over 0.1 r) + exponential TAIL of height `tail` and length tail_len * r from each end that has neighbours:
        # overlapping tails stiffen a short strut (L / r < ~5) between two joints strongly and a long strut little (B10: the C4 short struts act ~3 x stiffer)
        nq, nsub = 8, 6
        xg, wg = np.polynomial.legendre.leggauss(nq)
        xg, wg = torch.as_tensor(xg, dtype=dt), torch.as_tensor(wg, dtype=dt)
        edges = torch.linspace(0.0, 1.0, nsub + 1, dtype=dt)
        a_, b_ = edges[:-1][:, None], edges[1:][:, None]
        u = (a_ + (b_ - a_) * (xg[None, :] + 1) / 2).reshape(-1)                              # (nsub nq) points in [0, 1]
        wu = ((b_ - a_) / 2 * wg[None, :]).reshape(-1)
        sg = lo[:, None] + (hi - lo)[:, None] * u[None, :]                                    # (ne, nq nsub) distance from node 0
        ra_e = rb[boe]                                                                        # own radius per element
        w_s = (2.0 * smooth if smooth else 0.1) * ra_e[:, None]
        phi0 = torch.sigmoid((l0[:, None] - sg) / w_s)
        phi1 = torch.sigmoid((l1[:, None] - (Lbt[:, None] - sg)) / w_s)
        t0 = tail * (l0 > 0).to(dt)[:, None] * torch.exp(-(sg / (tail_len * ra_e[:, None])) ** tail_pow)
        t1 = tail * (l1 > 0).to(dt)[:, None] * torch.exp(-((Lbt[:, None] - sg) / (tail_len * ra_e[:, None])) ** tail_pow)
        if not tail:      # smooth step only: blend the COMPLIANCE, 1/beta = 1 - (1 - 1/k0) phi0 - (1 - 1/k1) phi1. (A sigmoid in beta shifts the effective zone edge outward by
            # ~ w ln k, +27 % on the short C4 struts; the compliance blend is symmetric and reproduces the hard step exactly for an isolated zone.)
            cden = 1.0 - (1.0 - 1.0 / k0)[:, None] * phi0 - (1.0 - 1.0 / k1)[:, None] * phi1
            cden = torch.clamp(cden, min=1e-3)
            return le / ((le[:, None] * wu[None, :]) * cden).sum(1)
        beta0 = 1 + (k0 - 1)[:, None] * phi0 + t0
        beta1 = 1 + (k1 - 1)[:, None] * phi1 + t1
        beta = beta0 * beta1 if tail_mult else beta0 + beta1 - 1          # tail_mult: overlapping joint regions multiply (merged blobs are stiffer than the sum)
        return le / ((le[:, None] * wu[None, :]) / beta).sum(1)
    if taper > 0:         # smooth profile: beta(s) = 1 + (k - 1) (1 - s / l)+^taper from each end, compliance = int ds / beta (Gauss-Legendre)
        xg, wg = np.polynomial.legendre.leggauss(8)
        xg, wg = torch.as_tensor(xg, dtype=dt), torch.as_tensor(wg, dtype=dt)
        sg = lo[:, None] + (hi - lo)[:, None] * (xg[None, :] + 1) / 2                    # (ne, 8) distance from node 0
        phi0 = torch.clamp(1 - sg / l0.clamp_min(1e-12)[:, None], min=0.0) ** taper
        phi1 = torch.clamp(1 - (Lbt[:, None] - sg) / l1.clamp_min(1e-12)[:, None], min=0.0) ** taper
        beta = 1 + (k0 - 1)[:, None] * phi0 + (k1 - 1)[:, None] * phi1
        return le / ((le[:, None] / 2 * wg[None, :]) / beta).sum(1)
    comp = (le - ov0 - ov1) + ov0 / k0 + ov1 / k1
    return le / comp


def rule_factors(lat, r_elem, short_gain=0.0, short_l0=5.5, smooth=0.0, **kw):
    """_rule_core (above) times an optional SHORT-STRUT correction: the factors of the elements of a strut of length L and own radius r are multiplied by
    1 + short_gain * max(short_l0 - L / r, 0). Calibration on the C4 solids (abaqus/short_strut_calibration.py): the short (2.63 mm) C4 struts need an extra x2.1 at L / r = 4.4
    and x3.2 at L / r = 3.3 on top of R5 (gain ~ 1, l0 ~ 5.5), but C12 (L / r 4.6-5.7, valence 2-3) needs none: a C4-specific stopgap, off by default (notes/18 section 18)."""
    s = _rule_core(lat, r_elem, smooth=smooth, **kw)
    if short_gain:
        X, C = lat['X'], lat['conn']
        boe = torch.as_tensor(np.asarray(lat['beam_of_elem']))
        Lb = torch.as_tensor(np.linalg.norm(X[C[:, 1]] - X[C[:, 0]], axis=1), dtype=r_elem.dtype)
        nb = len(C)
        cnt = torch.zeros(nb, dtype=r_elem.dtype).index_add(0, boe, torch.ones_like(r_elem))
        rb = torch.zeros(nb, dtype=r_elem.dtype).index_add(0, boe, r_elem) / cnt
        x = short_l0 - Lb[boe] / rb[boe]
        s = s * (1.0 + short_gain * (_sp(x, 0.03) if smooth else torch.clamp(x, min=0.0)))   # onset smoothed over 0.03 in L / r only (a wider one leaks into C12's L / r = 4.64 struts)
    return s


# Ready-to-use joint configurations: drop_rig(..., joint=jointRule.PRESETS['R5']) (also latticeSetup.drop_rig / cases.b3_ours). Values = the fit files in abaqus/results.
#   'R5'       fitted on 13 linear specimens (B3, B5, B6, B7, B8, B9a-c); held-out rms 5 %, max 10 %; unequal radii (B11) within +-6 %; C4 cells 16-24 % too soft (notes/18 sections 13-18).
#   'R5_all22' refitted on all 22 specimens (adds B9d, B10, B11): rms 7.0 %, max 17 %; trades the long-strut cells for C4 (B10a -17 %); use only if C4 statics matter most.
# mu = 1: the EPU 45 factor fitted on the fast prescribed ramps does not transfer to impact (notes/18 section 9).
PRESETS = {
    'R5': dict(mode='rule', a=3.4462742365415906, zl=0.7732454540044637, gamma=0.002556182131193757, b=0.15346978546753595, p=-0.771172326210512, wsat=2.4686884460414977, smooth=0.05),
    # R5 + the short-strut law (gain 1.5, onset L / r = 4.6), calibrated on the C4 solids B10a-d and tested BLIND on B13a-d (decoupled short / long radii): errors +2.1, +7.5, -8.6, -1.7 %
    # (rms 5.9 %); leave-one-out over the eight C4 solids 6.1 % rms, 10.5 % worst; the 22-specimen controls (C12, unequal-radius T-joints, octet) unchanged. Use for C4-type lattices (valence-3 nodes,
    # 2.6 mm short struts). Known limits: the C4 impact peak force is worse (+49 % / +25 %), the multiplier also depends on the long struts' radius at the largest short-strut radius (notes/18 section 20).
    'R5_C4': dict(mode='rule', a=3.4462742365415906, zl=0.7732454540044637, gamma=0.002556182131193757, b=0.15346978546753595, p=-0.771172326210512, wsat=2.4686884460414977, short_gain=1.5, short_l0=4.6, smooth=0.05),
    'R5_all22': dict(mode='rule', a=2.993269531223047, zl=0.9297752637234049, gamma=0.061641144032001334, b=0.12074977714682034, p=-0.29195371444849794, wsat=3.017669602138089, smooth=0.05),
}
