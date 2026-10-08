"""Topology-aware joint stiffening from the fused-section geometry (no fitted constant).

Each strut is sliced perpendicular to its axis. In a slice at axial position s the solid material is the union
of the strut's own circle, the cross-sections of the neighbouring struts' cylinders (all struts sharing either
end node) and of the joint spheres, trimmed to the plate slab. Only the part CONNECTED to the strut's own
section is kept (a separate blob of neighbour material does not stiffen it). Its second moment of area about
its centroid, relative to the bare strut, gives the bending stiffening beta(s) >= 1 ('thicker strut end',
Smith et al. 2013, Labeas & Sunaric 2010, generalised from a constant end radius to the actual fused section).
An element gets the series (compliance) average s_e = l_e / int ds / beta(s), so the result is mesh
independent. Far from the nodes the slice is the strut's own circle and beta = 1 exactly.
"""
import numpy as np
from scipy import ndimage


def _slice_beta(Pn, t, e1, e2, ra, cyl, sph, z_lo, z_hi, n_grid, half):
    """beta (mean of the two principal second moments of the connected union, / pi ra^4 / 4) for one slice.
    Pn: slice centre on the axis (3,), t/e1/e2 strut frame, cyl: list of (p0, tb, Lb, rb), sph: list (c, rs)."""
    u = np.linspace(-half, half, n_grid)
    Y, Z = np.meshgrid(u, u, indexing='ij')
    P = Pn[None, None, :] + Y[..., None] * e1 + Z[..., None] * e2
    inside = Y * Y + Z * Z <= ra * ra
    for p0, tb, Lb, rb in cyl:
        d = P - p0
        tt = d @ tb
        perp = d - tt[..., None] * tb
        inside |= (tt >= 0) & (tt <= Lb) & ((perp * perp).sum(-1) <= rb * rb)
    for c, rs in sph:
        inside |= ((P - c) ** 2).sum(-1) <= rs * rs
    inside &= (P[..., 2] >= z_lo - 1e-9) & (P[..., 2] <= z_hi + 1e-9)
    return _moments(inside, Y, Z, u, ra, n_grid, decay)


def _moments(inside, Y, Z, u, ra, n_grid, decay):
    """Area / second-moment ratios (to the bare strut) of the connected part of the slice mask that contains the
    strut axis, with optional Saint-Venant weights exp(-d / (decay ra)) outside the strut circle."""
    lab, _ = ndimage.label(inside)
    c0 = lab[n_grid // 2, n_grid // 2]
    if c0 == 0:
        return 1.0
    m = lab == c0
    dA = (u[1] - u[0]) ** 2
    A = m.sum() * dA
    yc, zc = (Y[m].sum() * dA) / A, (Z[m].sum() * dA) / A
    Iyy = (((Z[m] - zc) ** 2).sum()) * dA
    Izz = (((Y[m] - yc) ** 2).sum()) * dA
    Iyz = (((Y[m] - yc) * (Z[m] - zc)).sum()) * dA
    tr, det = Iyy + Izz, Iyy * Izz - Iyz * Iyz
    lmin = 0.5 * tr - np.sqrt(max(0.25 * tr * tr - det, 0.0))
    lmax = tr - lmin
    I0 = np.pi * ra ** 4 / 4
    return max(0.5 * (lmin + lmax) / I0, 1.0), max(lmin / I0, 1.0)


def strut_beta(lat, r_beam, ds_frac=0.05, n_grid=81, half_factor=3.0):
    """beta profiles along every strut: list of (s (k,), beta_mean (k,), beta_min (k,)). r_beam (nb,) radii."""
    X, C = lat['X'], lat['conn']
    z_lo, z_hi = X[:, 2].min(), X[:, 2].max()
    nb = len(C)
    rn = np.zeros(len(X))
    for b, (i, j) in enumerate(C):
        rn[i], rn[j] = max(rn[i], r_beam[b]), max(rn[j], r_beam[b])
    out = []
    for b, (i, j) in enumerate(C):
        p, q = X[i], X[j]
        L = np.linalg.norm(q - p)
        t = (q - p) / L
        a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(t, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        cyl = []
        for c in np.nonzero(np.isin(C[:, 0], (i, j)) | np.isin(C[:, 1], (i, j)))[0]:
            if c == b:
                continue
            p0, q0 = X[C[c, 0]], X[C[c, 1]]
            Lc = np.linalg.norm(q0 - p0)
            cyl.append((p0, (q0 - p0) / Lc, Lc, r_beam[c]))
        sph = [(X[i], rn[i]), (X[j], rn[j])]
        ra = r_beam[b]
        half = half_factor * max(ra, rn[i], rn[j])
        s = np.arange(0.0, L + 1e-12, ds_frac * ra)
        bm, bn = [], []
        for sv in s:
            r_ = _slice_beta(p + sv * t, t, e1, e2, ra, cyl, sph, z_lo, z_hi, n_grid, half)
            bm.append(r_[0]); bn.append(r_[1])
        out.append((s, np.array(bm), np.array(bn)))
    return out


def element_factors(lat, r_beam, which='mean', **kw):
    """Per-element stiffness factors s_e = l_e / int ds / beta(s) over each element (mesh independent)."""
    X, C, Xr, connr = lat['X'], lat['conn'], lat['Xr'], lat['connr']
    boe = np.asarray(lat['beam_of_elem'])
    prof = strut_beta(lat, r_beam, **kw)
    se = np.ones(len(connr))
    for e, (a, b) in enumerate(connr):
        k = boe[e]
        p, q = X[C[k, 0]], X[C[k, 1]]
        t = (q - p) / np.linalg.norm(q - p)
        s1, s2 = sorted(((Xr[a] - p) @ t, (Xr[b] - p) @ t))
        s, bm, bn = prof[k]
        beta = bm if which == 'mean' else bn
        ss = np.linspace(s1, s2, 41)
        inv = np.interp(ss, s, 1.0 / beta)
        se[e] = (s2 - s1) / np.trapezoid(inv, ss)
    return se, prof


def _slice_moments(Pn, e1, e2, ra, cyl, sph, z_lo, z_hi, n_grid, half, decay=None, hulls=(), own=True):
    """Connected fused section in the slice (frame e1 = element y, e2 = element z): ratios to the bare strut of
    (area, I about y = int z^2, I about z = int y^2) about the section centroid, each >= 1.
    decay: None (all fused material fully effective: plane sections through the whole union, an upper bound) or a
    length lam (in units of ra): fused material outside the strut's own circle counts with weight exp(-d / lam ra),
    d = its distance from the circle ('Saint-Venant participation': bending stress diffuses into side-attached
    material only over a length of the order of the section size, as for the effective width of flanges)."""
    u = np.linspace(-half, half, n_grid)
    Y, Z = np.meshgrid(u, u, indexing='ij')
    P = Pn[None, None, :] + Y[..., None] * e1 + Z[..., None] * e2
    inside = (Y * Y + Z * Z <= ra * ra) if own else np.zeros(Y.shape, dtype=bool)
    for p0, tb, Lb, rb in cyl:
        d = P - p0
        tt = d @ tb
        perp = d - tt[..., None] * tb
        inside |= (tt >= 0) & (tt <= Lb) & ((perp * perp).sum(-1) <= rb * rb)
    for c, rs in sph:
        inside |= ((P - c) ** 2).sum(-1) <= rs * rs
    for hd in hulls:                                       # beamBuild joints: convex hull of the strut end caps
        inside |= (hd.find_simplex(P.reshape(-1, 3)) >= 0).reshape(inside.shape)
    inside &= (P[..., 2] >= z_lo - 1e-9) & (P[..., 2] <= z_hi + 1e-9)
    return _moments(inside, Y, Z, u, ra, n_grid, decay)


def _moments(inside, Y, Z, u, ra, n_grid, decay):
    """Area / second-moment ratios (to the bare strut) of the connected part of the slice mask that contains the
    strut axis, with optional Saint-Venant weights exp(-d / (decay ra)) outside the strut circle."""
    lab, _ = ndimage.label(inside)
    c0 = lab[n_grid // 2, n_grid // 2]
    if c0 == 0:
        return 1.0, 1.0, 1.0
    m = lab == c0
    dA = (u[1] - u[0]) ** 2
    w = np.ones(m.sum())
    if decay is not None:
        dist = np.maximum(np.sqrt(Y[m] ** 2 + Z[m] ** 2) - ra, 0.0)
        w = np.exp(-dist / (decay * ra))
    A = w.sum() * dA
    yc, zc = (w * Y[m]).sum() / w.sum(), (w * Z[m]).sum() / w.sum()
    Iy = (w * (Z[m] - zc) ** 2).sum() * dA                 # about local y (stress varies with z)
    Iz = (w * (Y[m] - yc) ** 2).sum() * dA
    A0, I0 = np.pi * ra ** 2, np.pi * ra ** 4 / 4
    return max(A / A0, 1.0), max(Iy / I0, 1.0), max(Iz / I0, 1.0)


def _node_hulls(X, C, r_beam, n_seg=32, extension=0.0, rounded=False):
    """beamBuild.BeamStructureSTL joints: per node, the convex hull (Delaunay) of the circular end caps (own
    radius each) of all struts meeting there, each cap normal to its strut and placed at the strut end, which is
    moved by extension * L along the strut (negative = shortened into the strut: the hull then spans the gap).
    rounded: the hull bulged as beamBuild._round_hull (membership by the hull of the rounded vertices)."""
    from scipy.spatial import Delaunay
    th = np.linspace(0, 2 * np.pi, n_seg + 1)[:-1]
    hulls = {}
    for n in range(len(X)):
        inc = np.nonzero((C[:, 0] == n) | (C[:, 1] == n))[0]
        if len(inc) < 2:
            continue
        pts = []
        for b in inc:
            o = C[b, 1] if C[b, 0] == n else C[b, 0]
            L = np.linalg.norm(X[o] - X[n])
            t = (X[o] - X[n]) / L
            a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
            e1 = np.cross(t, a); e1 /= np.linalg.norm(e1)
            e2 = np.cross(t, e1)
            c0 = X[n] - extension * L * t                      # extension < 0: cap moved into the strut
            pts.append(c0 + r_beam[b] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2))
        P = np.vstack(pts)
        if rounded:
            import trimesh
            from beamBuild import BeamStructureSTL
            P = BeamStructureSTL._round_hull(trimesh.convex.convex_hull(P)).vertices
        hulls[n] = Delaunay(P)
    return hulls


def _fhs_slicer(lat, r_beam, z_lo, z_hi):
    """FHS print solid of the lattice (nodeJoint.lattice_fhs, trimmed to the plates) and a function giving, for strut
    b at axial position s, the exact slice mask on the (Y, Z) grid in the strut frame (x, y, z): the solid is rotated
    so that the strut axis is the slicing direction, cut by manifold3d, and the polygons rasterised (even-odd)."""
    import manifold3d as m3
    from matplotlib.path import Path
    from nodeJoint import lattice_fhs
    X, C = lat['X'], lat['conn']
    man, _ = lattice_fhs(X, C, r_beam, z_range=(z_lo, z_hi))
    cache = {}

    def mask(b, p, x, y, z, s, Y, Z):
        if b not in cache:                                  # world -> (y, z, x) about p: axis along local z
            R = np.vstack((y, z, x))
            M = np.column_stack((R, -R @ p))
            cache[b] = man.transform(M.tolist())
        pts = np.column_stack((Y.ravel(), Z.ravel()))
        inside = np.zeros(len(pts), bool)
        for poly in cache[b].slice(float(s)).to_polygons():
            inside ^= Path(np.asarray(poly)).contains_points(pts)
        return inside.reshape(Y.shape)
    return mask


def element_factors_aniso(lat, r_beam, ds_frac=0.05, n_grid=81, half_factor=3.0, decay=None, joint_shape='sphere',
                          extension=0.0, rounded=False):
    """joint_shape: 'sphere' (gmsh/Abaqus validation solids: sphere of the largest radius at each node), 'hull'
    (beamBuild print STLs: convex hull of the strut end caps, which always lies inside that sphere) or 'fhs' (the
    FHS print joints, nodeJoint.lattice_fhs: exact slices of the FHS solid)."""
    """Per-element (ne, 6) factors for the resultants [N, T, My1, Mz1, My2, Mz2]: series averages over each
    element of the fused-section ratios in the element's own local axes (explicitFE.Frame rule)."""
    X, C, Xr, connr = lat['X'], lat['conn'], lat['Xr'], lat['connr']
    boe = np.asarray(lat['beam_of_elem'])
    z_lo, z_hi = X[:, 2].min(), X[:, 2].max()
    rn = np.zeros(len(X))
    for b, (i, j) in enumerate(C):
        rn[i], rn[j] = max(rn[i], r_beam[b]), max(rn[j], r_beam[b])
    prof = []
    node_hulls = _node_hulls(X, C, r_beam, extension=extension, rounded=rounded) if joint_shape == 'hull' else {}
    fhs = _fhs_slicer(lat, r_beam, z_lo, z_hi) if joint_shape == 'fhs' else None
    ext_l = -min(extension, 0.0) if joint_shape == 'hull' else 0.0      # print struts stop this fraction short
    for b, (i, j) in enumerate(C):
        p, q = X[i], X[j]
        L = np.linalg.norm(q - p)
        x = (q - p) / L
        v = np.eye(3)[np.argmin(np.abs(x))]                  # Frame: global axis least aligned with the beam
        y = np.cross(v, x); y /= np.linalg.norm(y)
        z = np.cross(x, y)
        cyl = []
        for c in np.nonzero(np.isin(C[:, 0], (i, j)) | np.isin(C[:, 1], (i, j)))[0]:
            p0, q0 = X[C[c, 0]], X[C[c, 1]]
            Lc = np.linalg.norm(q0 - p0)
            tc = (q0 - p0) / Lc
            if c == b and ext_l == 0.0:
                continue
            # print geometry: each strut only between its shortened ends (the strut itself too, if shortened)
            cyl.append((p0 + ext_l * Lc * tc, tc, (1 - 2 * ext_l) * Lc, r_beam[c]))
        sph = [(X[i], rn[i]), (X[j], rn[j])] if joint_shape == 'sphere' else []
        hl = [node_hulls[n] for n in (i, j) if n in node_hulls]
        ra = r_beam[b]
        half = half_factor * max(ra, rn[i], rn[j])
        sgrid = np.arange(0.0, L + 1e-12, ds_frac * ra)
        if fhs is not None:                                  # exact FHS slices (the strut itself is part of the solid)
            u = np.linspace(-half, half, n_grid)
            Y, Z = np.meshgrid(u, u, indexing='ij')
            R = np.array([_moments(fhs(b, p, x, y, z, min(max(sv, 1e-6), L - 1e-6), Y, Z), Y, Z, u, ra, n_grid, decay)
                          for sv in sgrid])
        else:
            R = np.array([_slice_moments(p + sv * x, y, z, ra, cyl, sph, z_lo, z_hi, n_grid, half, decay, hl,
                                         own=ext_l == 0.0) for sv in sgrid])
        prof.append((sgrid, R))
    F = np.ones((len(connr), 6))
    for e, (a, b) in enumerate(connr):
        k = boe[e]
        p, q = X[C[k, 0]], X[C[k, 1]]
        t = (q - p) / np.linalg.norm(q - p)
        s1, s2 = sorted(((Xr[a] - p) @ t, (Xr[b] - p) @ t))
        sgrid, R = prof[k]
        ss = np.linspace(s1, s2, 41)
        avg = lambda col: (s2 - s1) / np.trapezoid(np.interp(ss, sgrid, 1.0 / col), ss)
        fA, fY, fZ = avg(R[:, 0]), avg(R[:, 1]), avg(R[:, 2])
        F[e] = (fA, 0.5 * (fY + fZ), fY, fZ, fY, fZ)
    return F, prof
