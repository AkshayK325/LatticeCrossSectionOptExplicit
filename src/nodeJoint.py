"""One lattice node, studied in isolation: beams meeting at a point, trimmed so that none enters another, and the
methods that connect the trimmed ends (plan and log: notes/12_ThreeBeamJointStudy.md).
"""
import numpy as np
import manifold3d as m3
import trimesh


def three_beam_case():
    """Big / medium / small beam meeting in 3D (unequal angles 107 / 117 / 87 deg, not coplanar), flat far ends.
    Returns (node (3,), unit directions (3,3), radii (3,), lengths (3,))."""
    E = np.array([[1.0, 0, 0], [-0.30, 0.80, 0.52], [-0.45, -0.55, 0.70]])
    E /= np.linalg.norm(E, axis=1)[:, None]
    return np.zeros(3), E, np.array([1.10, 0.75, 0.40]), np.full(3, 6.0)


def _frame(e):
    u = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(e, u)


def _rotation(e):
    """Proper rotation with columns (u, v, e): maps the local z axis to e."""
    u, v = _frame(e)
    return np.column_stack((u, v, e))


def cylinder(node, e, r, L, seg=64, start=0.0):
    """Solid circular beam along e from start to L (flat ends), as a manifold."""
    c = m3.Manifold.cylinder(float(L - start), float(r), float(r), seg)
    return c.transform(np.column_stack((_rotation(e), node + start * e)).tolist())


def trim_radius(dirs, radii, L, dt=0.002, nphi=180):
    """Smallest sphere radius about the node outside which no beam enters another: the largest distance from the
    node of any point that lies on the surface of one beam and inside another (beams as solid cylinders from the
    node along their axes). Geometric, no free parameter. Returns (R_s, per-pair values {(i, j): R_ij})."""
    t = np.arange(0.0, L.max(), dt)
    ph = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    pair = {}
    for i in range(len(dirs)):
        u, v = _frame(dirs[i])
        P = (t[:, None, None] * dirs[i] + radii[i] * (np.cos(ph)[None, :, None] * u + np.sin(ph)[None, :, None] * v))
        P = P.reshape(-1, 3)
        for j in range(len(dirs)):
            if j == i:
                continue
            z = P @ dirs[j]
            inside = (z > 0) & (z < L[j]) & (np.linalg.norm(P - np.outer(z, dirs[j]), axis=1) < radii[j])
            if inside.any():
                key = tuple(sorted((i, j)))
                pair[key] = max(pair.get(key, 0.0), float(np.linalg.norm(P[inside], axis=1).max()))
    return max(pair.values()), pair


def to_stl_mesh(man):
    """The manifold as an STL stores it: float32 vertices welded, collapsed triangles dropped, re-validated
    (smoothJoint._float32_weld), so the written STL is exactly a closed 2-manifold."""
    from smoothJoint import _float32_weld
    return _float32_weld(man)


def export(mesh, path_stem, threemf=True):
    """Write <path_stem>.stl (binary) and, optionally, <path_stem>.3mf (indexed + zip: ~3x smaller, same mesh)."""
    mesh.export(f'{path_stem}.stl')
    if threemf:
        mesh.export(f'{path_stem}.3mf')


def to_trimesh(man):
    m = man.to_mesh64()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts, np.int64), process=False)


def trimmed_beams(node, dirs, radii, L, R_s, seg=64):
    """The beams with everything inside the trim sphere removed: each ends on the sphere with its own section, and
    no two touch (method B; the node region is left empty for the connection). One manifold of 3 disjoint parts."""
    ball = m3.Manifold.sphere(float(R_s), 4 * seg).translate(tuple(map(float, node)))
    parts = [cylinder(node, e, r, l, seg) - ball for e, r, l in zip(dirs, radii, L)]
    return m3.Manifold.batch_boolean(parts, m3.OpType.Add)


def ring_offsets(dirs, radii, R_s, margin=0.02, iters=50, core=False):
    """Distance from the node of each beam's node ring: outside the trim sphere (sqrt(R_s^2 - r_i^2)), and far enough
    that no other ring reaches beyond its plane, t_i >= t_j cos(theta_ij) + r_j sin(theta_ij) + margin * r_i, so
    every ring is a face of the hull of all rings (else the hull wraps over a beam, e.g. a thick beam whose radius is
    close to R_s leaves the sphere almost at its equator). Geometric: no free parameter (margin = 2 % of r_i)."""
    t = np.sqrt(np.maximum(R_s ** 2 - radii ** 2, 0.0))
    c = np.clip(dirs @ dirs.T, -1, 1)
    sn = np.sqrt(1 - c * c)
    for _ in range(iters):
        need = (t[None, :] * c + radii[None, :] * sn)                 # [i, j]: ring j's reach along e_i
        if core:                                                      # core ring j at the node reaches r_j sin
            need = np.maximum(need, radii[None, :] * sn)
        np.fill_diagonal(need, -np.inf)
        t_new = np.maximum(t, need.max(1) + margin * radii)
        if np.allclose(t_new, t):
            break
        t = t_new
    return t


def ring_count(r, r_min, n_min=16):
    """Vertices on a beam ring: proportional to the radius (similar edge lengths on every beam), >= n_min."""
    return max(n_min, int(round(n_min * r / r_min)))


def _tube_faces(a, b, n):
    """Quad strip (as triangles) between ring index blocks a and b of n vertices, outward for rings ordered
    counter-clockwise about the tube axis and b further along the axis."""
    i = np.arange(n)
    j = (i + 1) % n
    return np.vstack((np.column_stack((a + i, b + i, a + j)), np.column_stack((a + j, b + i, b + j))))


def loop_limit(V, F):
    """Loop subdivision limit positions (closed mesh): v_lim = (1 - n chi) v + chi * sum(neighbours), with
    chi = 1 / (3 / (8 beta) + n) and beta the Loop weight of valence n. After any number of levels this puts every
    vertex exactly on the smooth limit surface, so a coarser level loses only the chord between vertices."""
    import scipy.sparse as sp
    E = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V),) * 2).tocsr()
    A = ((A + A.T) > 0).astype(float)
    n = np.asarray(A.sum(1)).ravel()
    beta = (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2 * np.pi / n)) ** 2) / n
    chi = 1.0 / (3.0 / (8.0 * beta) + n)
    return (1 - n * chi)[:, None] * V + chi[:, None] * (A @ V)


def loop_shrink(n, spacing_ratio=1.0, levels=3, limit=False):
    """Radius factor by which Loop subdivision shrinks a long regular n-gon tube (rings `spacing_ratio` edge lengths
    apart): measured numerically on a test tube, so that control rings can be enlarged by 1 / factor."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dz = spacing_ratio * 2 * np.sin(np.pi / n)
    rings = [np.column_stack((np.cos(t), np.sin(t), np.full(n, k * dz))) for k in range(40)]
    V = np.vstack(rings)
    F = np.vstack([_tube_faces(k * n, (k + 1) * n, n) for k in range(39)])
    V2, F2 = trimesh.remesh.subdivide_loop(V, F, iterations=levels)
    if limit:
        V2 = loop_limit(V2, F2)
    z = V2[:, 2]
    mid = np.abs(z - z.mean()) < 3 * dz
    return float(np.linalg.norm(V2[mid, :2], axis=1).mean())


def _fibonacci(n=64):
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    ph = np.pi * (1 + 5 ** 0.5) * k
    return np.column_stack((np.sqrt(1 - z * z) * np.cos(ph), np.sqrt(1 - z * z) * np.sin(ph), z))


def ball_points(P_hull, center, rho, margin, dirs, t_ring):
    """Points of the node ball (radius rho, Fibonacci directions) outside the hull of P_hull by more than margin:
    the back of a node whose beams lie in a narrow cone (notes/17 P3.4). Points inside or near the hull are dropped
    (no near-duplicate hull vertices), and so are points not behind every node ring's plane by margin
    ((p - c) . e_i <= t_i - margin), so each ring stays a face of the hull (the tubes attach there; else the cap
    removal leaves a non-manifold edge)."""
    from scipy.spatial import ConvexHull
    Pb = center + rho * _fibonacci()
    H0 = ConvexHull(P_hull)
    out = (Pb @ H0.equations[:, :3].T + H0.equations[:, 3]).max(1) > margin
    behind = (((Pb - center) @ np.asarray(dirs).T) <= np.asarray(t_ring) - margin).all(1)
    return Pb[out & behind]


def section_ring(xy, n, phase=0.0):
    """n points along the closed section contour xy (m x 2, in the beam frame u, v), equally spaced by arclength,
    counter-clockwise, starting where the polar angle is `phase` (so neighbouring rings of one beam correspond)."""
    xy = np.asarray(xy, float)
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if (np.roll(xy[:, 0], -1) * xy[:, 1] - xy[:, 0] * np.roll(xy[:, 1], -1)).sum() > 0:   # clockwise: reverse
        xy = xy[::-1]
    k0 = np.argmin(np.abs(np.angle(np.exp(1j * (np.arctan2(xy[:, 1], xy[:, 0]) - phase)))))
    xy = np.roll(xy, -k0, axis=0)
    P = np.vstack((xy, xy[:1]))
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack((np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])))


_RING_OP = {}


def ring_limit_operator(n, levels=3, limit=False, spacing_ratio=1.0):
    """Linear map T (complex, n 2^levels x n) from a long tube's control ring (identical rings spacing_ratio edge
    lengths apart) to its subdivided (or limit) cross-section, ordered counter-clockwise from control vertex 0.
    Measured on a test tube (Loop is linear, and the same in x and y); T is circulant by construction."""
    key = (n, levels, limit, spacing_ratio)
    if key in _RING_OP:
        return _RING_OP[key]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dz = spacing_ratio * 2 * np.sin(np.pi / n)
    nr = 9                                                          # = 24 to machine precision (finite Loop support)
    F = np.vstack([_tube_faces(k * n, (k + 1) * n, n) for k in range(nr - 1)])

    def run(ring):
        V = np.vstack([np.column_stack((ring.real, ring.imag, np.full(n, k * dz))) for k in range(nr)])
        V2, F2 = trimesh.remesh.subdivide_loop(V, F, iterations=levels)
        if limit:
            V2 = loop_limit(V2, F2)
        return V2
    Vc = run(np.exp(1j * t))                                       # unit circle: find the mid ring and its order
    zm = (nr // 2) * dz
    sel = np.where(np.abs(Vc[:, 2] - zm) < 1e-9)[0]
    sel = sel[np.argsort(np.mod(np.arctan2(Vc[sel, 1], Vc[sel, 0]) + 1e-9, 2 * np.pi))]
    m = len(sel)
    if m != n * 2 ** levels:
        raise RuntimeError(f'ring_limit_operator: mid ring has {m} points, expected {n * 2 ** levels}')
    e0 = np.zeros(n, complex)
    e0[0] = 1.0
    Vd = run(e0)
    col = Vd[sel, 0] + 1j * Vd[sel, 1]
    T = np.column_stack([np.roll(col, j * 2 ** levels) for j in range(n)])
    _RING_OP[key] = T
    return T


def control_ring(xy, n, levels=3, limit=False, phase=0.0, smooth=0.0, spacing_ratio=1.0):
    """Control ring (n x 2) whose subdivided tube has the section xy: least squares T c = the contour sampled at the
    n 2^levels limit points (arclength), plus smooth * |T|^2 |D2 c|^2 (D2: circular second difference). The plain fit
    rings along flat sides (curvature stripes along the strut); smooth = 1e-2 halves that ripple at < 0.15 % r_eq
    more deviation (notes/17 P6). For a circle (either way) it is the radius / loop_shrink compensation."""
    T = ring_limit_operator(n, levels, limit, spacing_ratio)
    tgt = section_ring(xy, T.shape[0], phase)
    b = tgt[:, 0] + 1j * tgt[:, 1]
    if smooth > 0:
        I = np.eye(n)
        D = np.roll(I, 1, 1) - 2 * I + np.roll(I, -1, 1)
        A = np.vstack((T, np.sqrt(smooth) * np.linalg.norm(T, 2) * D))
        b = np.r_[b, np.zeros(n)]
    else:
        A = T
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.column_stack((c.real, c.imag))


LEAD_SECTION = 0.5   # lead (x r_eq) of non-circular struts: the section is within 1 % r_eq ~1.0 r_eq from the node
                     # ring (1.25-1.75 with lead 1.0), creases no worse (notes/17 P7)


def _blend(s):
    """Smoothstep 0 -> 1 on s in [0, 1] (clamped)."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def strictly_convex_ring(c, sag=0.03):
    """The ring c (n x 2, counter-clockwise) made strictly convex, keeping its vertex count and order: vertices off
    the ring's convex hull are moved onto a parabolic arc over their hull edge, bulging outward. Used where a ring
    must be a face of the node hull (first ring, core ring): an inverse-subdivision control ring rings (Gibbs-like)
    along flat sides and is not convex there (notes/17 P6). The bulge of each edge is sag x its length, capped at a
    tenth of the exterior angles at its two ends, so the turn stays positive where two arcs meet; the result is
    checked (every vertex a hull vertex) and an error is raised otherwise."""
    from scipy.spatial import ConvexHull
    n = len(c)
    hv = np.sort(ConvexHull(c).vertices)
    m = len(hv)
    ext = np.empty(m)                                              # exterior angle at each hull vertex
    for q in range(m):
        a, b, d = c[hv[q - 1]], c[hv[q]], c[hv[(q + 1) % m]]
        u1, u2 = b - a, d - b
        ext[q] = np.arctan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2)
    out = c.copy()
    c0 = c.mean(0)
    for q in range(m):
        a, b = hv[q], hv[(q + 1) % m]
        k = (b - a) % n
        if k < 2:
            continue
        idx = (a + np.arange(1, k)) % n
        A, B = c[a], c[b]
        ab = B - A
        sg = min(sag, 0.1 * min(ext[q], ext[(q + 1) % m]))       # end slope 4 sg < exterior angle / 2
        # parameter on the edge where the ray from the centroid through each vertex meets it, so that blending a
        # ring into c moves its vertices radially (even spacing crossed neighbouring rings: folds, notes/17 P7.3)
        P = c[idx] - c0
        den = P[:, 0] * ab[1] - P[:, 1] * ab[0]
        num_ = (A - c0)[0] * P[:, 1] - (A - c0)[1] * P[:, 0]
        tt = np.where(np.abs(den) > 1e-12, num_ / np.where(np.abs(den) > 1e-12, den, 1.0), np.nan)
        radial_ok = (np.all(np.isfinite(tt)) and tt.min() > 1e-6 and tt.max() < 1 - 1e-6
                     and np.all(np.diff(tt) > 1e-6))
        if not radial_ok:                                          # not star-shaped from the centroid here
            tt = np.linspace(0, 1, k + 1)[1:-1]
        nrm = np.array([ab[1], -ab[0]]) / np.linalg.norm(ab)        # outward for a counter-clockwise ring
        out[idx] = A + tt[:, None] * ab + (4 * sg * np.linalg.norm(ab) * tt * (1 - tt))[:, None] * nrm
    if len(ConvexHull(out).vertices) != n:
        raise RuntimeError('strictly_convex_ring: could not make the hull ring strictly convex')
    return out


def section_hausdorff(xy, c, levels=3, limit=False, spacing_ratio=1.0):
    """Hausdorff distance between the subdivided tube section of control ring c and the contour xy."""
    from shapely.geometry import LineString
    T = ring_limit_operator(len(c), levels, limit, spacing_ratio)
    lim = T @ (c[:, 0] + 1j * c[:, 1])
    a = LineString(np.vstack((np.c_[lim.real, lim.imag], [[lim[0].real, lim[0].imag]])))
    return a.hausdorff_distance(LineString(np.vstack((xy, xy[:1]))))


def section_control_ring(xy, n_min=16, levels=3, limit=False, tol=0.01, n_max=128, phase=0.0, smooth=0.0,
                         spacing_ratio=1.0):
    """Control ring for a general (convex or not) section: the smallest ring count n >= n_min, a multiple of 4
    (doubly symmetric sections), whose subdivided section is within tol * r_eq (r_eq = sqrt(A / pi)) of xy.
    Returns (control ring n x 2, n, Hausdorff / r_eq)."""
    from shapely.geometry import Polygon
    xy = np.asarray(xy, float)
    req = np.sqrt(Polygon(xy).area / np.pi)
    n = int(4 * np.ceil(n_min / 4))
    while True:
        c = control_ring(xy, n, levels, limit, phase, smooth, spacing_ratio)
        h = section_hausdorff(xy, c, levels, limit, spacing_ratio) / req
        if h <= tol or n >= n_max:
            return c, n, h
        n += 4


def hull_subdivision_node(node, dirs, radii, L, R_s, levels=3, n_min=16, lead=0.0, core=False, ball=True,
                          sections=None):
    """Method C1: organic node by subdivision of a hull control mesh (notes/12).
    Control mesh = convex hull of one ring per beam (at ring_offsets: outside the trim sphere and each ring a face of
    the hull), with the ring caps removed,
    plus a control tube per beam (rings one edge length apart, past the far end, capped). Loop subdivision gives a
    smooth (C2 away from extraordinary vertices) surface whose tubes converge to the beams (control rings enlarged
    by the measured Loop shrink); the far ends are then trimmed flat. ball: also hull points on the node ball
    (radius r_max, Loop-shrink compensated), so the hull has a back behind the node point when all beams lie in a
    narrow cone (else the core rings are nearly coplanar, the hull is flat there and the surface dents and creases
    inside the Boolean node's ball, notes/17 P3.4); points the beams already cover drop out of the hull.
    sections: per beam None (circle of radius r_i) or a convex section contour (m x 2, in the frame _frame(e_i),
    roll applied) with r_i its equivalent radius; its control ring is the inverse-subdivision fit
    (`section_control_ring`, within 1 % r_eq), and the ring offsets / ball use its circumradius (notes/17 P6).
    Returns the watertight manifold."""
    from scipy.spatial import ConvexHull
    r_min = radii.min()
    ctrl_rings = []                                               # per beam: control ring in (u, v), n
    for i, r in enumerate(radii):
        sec = None if sections is None else sections[i]
        if sec is None:
            n = ring_count(r, r_min, n_min)
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            ctrl_rings.append((r / loop_shrink(n, levels=levels) * np.column_stack((np.cos(t), np.sin(t))), n, None))
        else:
            c, n, _ = section_control_ring(sec, n_min=ring_count(r, r_min, n_min), levels=levels)
            ctrl_rings.append((c, n, sec))
    hull_rings = [c if sec is None else strictly_convex_ring(c) for c, _, sec in ctrl_rings]   # hull members
    r_out = np.array([max(np.linalg.norm(c, axis=1).max(), np.linalg.norm(h, axis=1).max())
                      for (c, _, _), h in zip(ctrl_rings, hull_rings)])                  # control circumradius
    toff = ring_offsets(dirs, r_out, R_s, core=core)
    V, F, ring0 = [], [], []
    nv = 0
    for (e, r, l), (c, n, sec), ch in zip(zip(dirs, radii, L), ctrl_rings, hull_rings):
        u, v = _frame(e)
        circ = np.outer(c[:, 0], u) + np.outer(c[:, 1], v)
        circ0 = np.outer(ch[:, 0], u) + np.outer(ch[:, 1], v)       # first ring: a hull face, strictly convex
        if sec is None:
            dz = 2 * r * np.sin(np.pi / n)
        else:                                                       # the section's own edge length
            P_ = np.vstack((sec, sec[:1]))
            dz = np.linalg.norm(np.diff(P_, axis=0), axis=1).sum() / n
        ti = toff[len(ring0)]
        # control rings: the first two gaps `lead` * r long (a longer blend from the hull into the beam), then one
        # edge length apart
        ld = lead if sec is None else min(lead, LEAD_SECTION)         # sections recover their shape sooner
        z0 = [ti] + ([ti + ld * r, ti + 2 * ld * r] if ld > 0 else [])
        zs = np.concatenate((z0, np.arange(z0[-1] + dz, l + 3 * dz, dz)))
        start = nv
        for kz, z in enumerate(zs):
            if sec is None:
                V.append(node + z * e + (circ0 if kz == 0 else circ))
            else:                                                  # hull ring -> section over a physical length
                b = _blend((z - ti) / (LEAD_SECTION * r))
                V.append(node + z * e + (1 - b) * circ0 + b * circ)
            nv += n
        ring0.append((start, n))
        for k in range(len(zs) - 1):
            F.append(_tube_faces(start + k * n, start + (k + 1) * n, n))
        V.append((node + (zs[-1] + 0.5 * dz) * e)[None])          # far cap (trimmed away later)
        last = start + (len(zs) - 1) * n
        i = np.arange(n)
        F.append(np.column_stack((last + i, np.full(n, nv), last + (i + 1) % n)))
        nv += 1
    cores = []
    if core:     # each beam's own section at the node: the hull then contains every beam up to the node point
        for (e, r), (c, n, sec) in zip(zip(dirs, radii), ctrl_rings):
            u, v = _frame(e)
            if sec is None:
                t = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
                rc = r / loop_shrink(n, levels=levels)             # same Loop-shrink compensation as the tubes
                V.append(node + rc * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)))
            else:                                                   # the hull ring, half a step rotated
                ch = strictly_convex_ring(c)
                cm = 0.5 * (ch + np.roll(ch, -1, axis=0))
                cm *= (np.linalg.norm(ch, axis=1).mean() / np.linalg.norm(cm, axis=1).mean())
                V.append(node + np.outer(cm[:, 0], u) + np.outer(cm[:, 1], v))
            cores.append(np.arange(nv, nv + n))
            nv += n
    if ball:
        Vr = np.vstack(V)
        rho = (radii.max() / loop_shrink(ring_count(radii.max(), r_min, n_min), levels=levels) if sections is None
               else r_out.max())
        Pb = ball_points(Vr[np.concatenate([np.arange(s, s + n) for s, n in ring0] + cores)], node, rho,
                         0.05 * r_min, dirs, toff)
        if len(Pb):
            V.append(Pb)
            cores.append(np.arange(nv, nv + len(Pb)))
            nv += len(Pb)
    V = np.vstack(V)
    # node: hull of the first ring of every beam (+ the core rings), without the ring caps
    idx = np.concatenate([np.arange(s, s + n) for s, n in ring0] + cores)
    H = ConvexHull(V[idx])
    S = idx[H.simplices]
    ring_of = np.full(len(V), -1)
    for q, (s, n) in enumerate(ring0):
        ring_of[s:s + n] = q
    rs = ring_of[S]
    S = S[~((rs[:, 0] >= 0) & (rs[:, 0] == rs[:, 1]) & (rs[:, 1] == rs[:, 2]))]   # rim caps (core rings: -1)
    c = V[idx].mean(0)                                             # outward orientation
    nrm = np.cross(V[S[:, 1]] - V[S[:, 0]], V[S[:, 2]] - V[S[:, 0]])
    flip = (nrm * (V[S].mean(1) - c)).sum(1) < 0
    S[flip] = S[flip][:, ::-1]
    F = np.vstack(F + [S])
    ctrl = trimesh.Trimesh(V, F, process=False)
    ctrl.remove_unreferenced_vertices()                             # core points inside the hull
    if not ctrl.is_winding_consistent:
        trimesh.repair.fix_winding(ctrl)
    if ctrl.volume < 0:
        ctrl.invert()
    V2, F2 = trimesh.remesh.subdivide_loop(ctrl.vertices, ctrl.faces, iterations=levels)
    sm = m3.Manifold(m3.Mesh(np.ascontiguousarray(V2, np.float32), np.ascontiguousarray(F2, np.uint32)))
    if sm.status() != m3.Error.NoError:
        raise RuntimeError(f'subdivided node is not a manifold: {sm.status()}')
    for e, l in zip(dirs, L):                                     # flat far ends
        sm = sm.trim_by_plane(tuple(map(float, -e)), float(-(node @ e) - l))
    return sm, ctrl


def cotan_laplacian(V, F):
    """Cotangent Laplacian M^-1 C (C: cotangent weights, M: lumped area), the geometric discrete Laplace-Beltrami."""
    import scipy.sparse as sp
    C = sp.csr_matrix((len(V), len(V)))
    rows, cols, vals = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):            # angle at c is opposite edge a-b
        u = V[F[:, a]] - V[F[:, c]]
        w = V[F[:, b]] - V[F[:, c]]
        cot = (u * w).sum(1) / np.maximum(np.linalg.norm(np.cross(u, w), axis=1), 1e-16)
        rows += [F[:, a], F[:, b]]
        cols += [F[:, b], F[:, a]]
        vals += [0.5 * cot, 0.5 * cot]
    W = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(V),) * 2).tocsr()
    Cm = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    M = np.bincount(F.ravel(), weights=np.repeat(area / 3, 3), minlength=len(V))
    return sp.diags(1.0 / M) @ Cm


_SDF_CACHE = {}


def polygon_sdf(Q, poly, grad=False):
    """Signed distance from 2D points Q (k x 2) to the closed polygon poly (m x 2), < 0 inside, and optionally its
    gradient. The contour is resampled at ~perimeter / 2000 (k-d tree, cached per polygon; distance error < 0.1 % of
    perimeter / 2pi); the sign is that of (q - s) . n_s, s the nearest sample and n_s its outward normal (the SDF
    gradient points along the normal at the nearest point), not a point-in-polygon test (83 % of the time before,
    notes/17 P7.2)."""
    from scipy.spatial import cKDTree
    key = (poly.shape, poly.tobytes())
    if key not in _SDF_CACHE:
        P = np.asarray(poly, float)
        if (np.roll(P[:, 0], -1) * P[:, 1] - P[:, 0] * np.roll(P[:, 1], -1)).sum() > 0:   # make counter-clockwise
            P = P[::-1]
        Pc = np.vstack((P, P[:1]))
        sl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
        t = np.linspace(0, sl[-1], 2000, endpoint=False)
        D = np.column_stack((np.interp(t, sl, Pc[:, 0]), np.interp(t, sl, Pc[:, 1])))
        tg = np.roll(D, -1, axis=0) - np.roll(D, 1, axis=0)
        N = np.column_stack((tg[:, 1], -tg[:, 0]))
        N /= np.linalg.norm(N, axis=1, keepdims=True)                 # outward (counter-clockwise contour)
        _SDF_CACHE[key] = (cKDTree(D), D, N)
    tree, D, N = _SDF_CACHE[key]
    d, j = tree.query(Q)
    R = Q - D[j]
    sg = np.where((R * N[j]).sum(1) < 0, -1.0, 1.0)
    if not grad:
        return sg * d
    g = np.where((d > 1e-12)[:, None], sg[:, None] * R / np.maximum(d, 1e-12)[:, None], N[j])
    return sg * d, g


def core_distance(P, node, e, r, reach, section=None, grad=False):
    """Signed distance from P to one beam's core, the beam continued to the node point (axial 0 .. reach), and
    optionally its gradient. Circle (section None): the capsule of radius r (rounded ends; with the other capsules it
    is the Boolean node with its ball). Section (contour in the beam frame _frame(e)): the prism of that section with
    flat ends."""
    x = (P - node) @ e
    w = np.clip(x, 0.0, reach)
    D = (P - node) - np.outer(w, e)
    if section is None:
        nD = np.linalg.norm(D, axis=1)
        if not grad:
            return nD - r
        return nD - r, D / np.maximum(nD, 1e-12)[:, None]
    u, v = _frame(e)
    a = x - w                                                       # signed: < 0 behind the node, > 0 past reach
    q = np.column_stack((D @ u, D @ v))
    if grad:
        d2, g2 = polygon_sdf(q, np.asarray(section, float), grad=True)
        g3 = np.outer(g2[:, 0], u) + np.outer(g2[:, 1], v)
    else:
        d2 = polygon_sdf(q, np.asarray(section, float))
    out = (d2 > 0) | (a != 0)
    dp = np.maximum(d2, 0.0)
    d = np.where(out, np.sqrt(dp ** 2 + a ** 2), d2)
    if not grad:
        return d
    go = (dp[:, None] * g3 + a[:, None] * e) / np.maximum(d, 1e-12)[:, None]
    return d, np.where(out[:, None], go, g3)


def core_clearance(P, node, dirs, radii, reach, sections=None, grad=False):
    """Smooth distance from P to the union of the beam cores (each beam's capsule from the node point to reach_i):
    cubic smooth minimum (width 0.5 r_min) of the capsule distances, so it has no jumps and no creases where two
    cores meet."""
    k = 0.5 * radii.min()
    out = g = None
    for i, (e, r, l) in enumerate(zip(dirs, radii, reach)):
        sec = None if sections is None else sections[i]
        if grad:
            d, gd = core_distance(P, node, e, r, l, sec, grad=True)
        else:
            d = core_distance(P, node, e, r, l, sec)
        if out is None:
            out, g = d, (gd if grad else None)
            continue
        hh = np.maximum(k - np.abs(out - d), 0.0) / k
        if grad:                                                    # d/d(out) of the cubic smooth minimum
            wa = np.where(out < d, 1.0 - 0.5 * hh ** 2, 0.5 * hh ** 2)
            g = wa[:, None] * g + (1.0 - wa)[:, None] * gd
        out = np.minimum(out, d) - hh ** 3 * k / 6.0
    return (out, g) if grad else out


def flank_weights(V, node, dirs, radii, reach, sections=None):
    """Anchor weight per vertex for the anchored fairing, in [0, 1]: ~1 on a beam's own flank (close to its core,
    i.e. the beam cylinder continued to the node point, and away from every other core) where the thin plate would
    neck the beam; ~0 in the crevices (comparably close to two cores), which stay free to fair."""
    D = [core_distance(V, node, e, r, l, None if sections is None else sections[i])
         for i, (e, r, l) in enumerate(zip(dirs, radii, reach))]
    D = np.sort(np.array(D), axis=0)
    a = 0.3 * radii.min()
    b = 0.5 * radii.min()
    return np.exp(-(np.maximum(D[0], 0.0) / a) ** 2) * (1.0 - np.exp(-((D[1] - D[0]) / b) ** 2))


def capsule_depth(P, A, B, R):
    """Signed distance from points P to the union of capsules (segments A_k-B_k, radius R_k; < 0 inside) and, per
    point, its projection onto the nearest capsule's surface. The struts' capsules from the node point are the
    Boolean node (cylinders + a radius-r ball at the node)."""
    best = np.full(len(P), np.inf)
    proj = P.copy()
    for a, b, r in zip(A, B, R):
        ab = b - a
        t = np.clip((P - a) @ ab / (ab @ ab), 0.0, 1.0)
        q = a + t[:, None] * ab
        d = P - q
        n = np.linalg.norm(d, axis=1)
        sd = n - r
        s = sd < best
        best[s] = sd[s]
        ok = s & (n > 1e-12)
        proj[ok] = q[ok] + r * d[ok] / n[ok, None]
    return best, proj


def smooth_core_depth(P, node, dirs, radii, reach, h=1e-4, sections=None):
    """core_clearance (smooth union of the beam capsules: no crease where two cores meet) and each point's
    projection onto that surface along the clearance gradient (analytic; was 6 extra evaluations)."""
    d, g = core_clearance(P, node, dirs, radii, reach, sections, grad=True)
    gn = np.linalg.norm(g, axis=1)
    ok = gn > 1e-9
    proj = P.copy()
    proj[ok] = P[ok] - (d[ok] / gn[ok] ** 2)[:, None] * g[ok]
    return d, proj


def contained_fair_solve(K, wv, V0, fixed, depth, tol, strong, max_iter=12):
    """min x^T K x + sum w_v |x_v - x0_v|^2 subject to depth(x) >= -tol (the surface stays outside the cores), by an
    active set: solve, pin every new violator firmly (weight `strong`) to its projection on the core surface,
    re-solve, until no new violator. The thin plate then fairs around the pinned points. Returns (V, n_pinned,
    iterations)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    fr, fx = np.where(~fixed)[0], np.where(fixed)[0]
    wc = np.zeros(len(V0))
    T = np.zeros_like(V0)
    for it in range(max_iter):
        A = (K + sp.diags(wv + wc)).tocsr()
        solve = spl.factorized(A[fr][:, fr].tocsc())
        Vn = V0.copy()
        for k in range(3):
            Vn[fr, k] = solve(wv[fr] * V0[fr, k] + wc[fr] * T[fr, k] - A[fr][:, fx] @ V0[fx, k])
        d = np.full(len(Vn), np.inf)                               # only the free vertices can violate
        proj = Vn.copy()
        d[fr], proj[fr] = depth(Vn[fr])
        new = (d < -tol) & ~fixed & (wc == 0)
        if not new.any():
            contained_fair_solve.pinned_mask = wc > 0
            return Vn, int((wc > 0).sum()), it + 1
        wc[new] = strong[new]
        T[new] = proj[new]
    contained_fair_solve.pinned_mask = wc > 0
    return Vn, int((wc > 0).sum()), max_iter


def fair_node_anchored(mesh, node, dirs, radii, t_ring, free_len=1.0, anchor=10.0, contain='smooth', tol=0.02,
                       strong=1e1, sections=None):
    """FHS fairing (Faired Hull-Subdivision node, anchored): least-squares fairing, min x^T K x + sum_v w_v |x_v - x0_v|^2, with the thin-plate
    energy K = C^T M^-1 C (cotangent C, lumped area M), the beams fixed beyond t_ring_i + free_len * r_i, and soft
    anchors w = anchor * area / r_min^4 * flank_weights on the beams' flanks, so the crevices fair freely but no beam
    necks. One linear solve (no clamping, no folds). contain: also keep the surface outside every beam's core (its
    cylinder continued to the node point + a radius-r ball there = the Boolean node), within tol * r_min, by an active
    set (`contained_fair_solve`); without it the thin plate can cut into the cores behind the node point of an acute
    valence-3 node (section 0.82 r at 30 deg, notes/17 P3.4). Returns (mesh, fixed mask)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    fixed = np.zeros(len(V), bool)
    for i, (e, r, t) in enumerate(zip(dirs, radii, t_ring)):
        w = (V - node) @ e
        far = w > t + free_len * r                                  # on the beam's own tube past the free length
        sec = None if sections is None else sections[i]
        fixed[far] |= np.abs(core_distance(V[far], node, e, r, np.inf, sec)) < 0.05 * r
    Lc = cotan_laplacian(V, F)                                        # M^-1 C
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    M = sp.diags(np.bincount(F.ravel(), weights=np.repeat(area / 3, 3), minlength=len(V)))
    K = (Lc.T @ M @ Lc).tocsr()                                       # = C^T M^-1 C
    # data term anchor * integral |x - x0|^2 / r_min^4 dA, consistent with the thin-plate energy integral |Lap x|^2 dA
    wv = anchor * M.diagonal() / radii.min() ** 4 * flank_weights(V, node, dirs, radii,
                                                                   t_ring + free_len * radii + 2 * radii, sections)
    if contain:
        reach = t_ring + free_len * radii + 2 * radii
        depth = ((lambda X: smooth_core_depth(X, node, dirs, radii, reach, sections=sections)) if contain == 'smooth' else
                 (lambda X: capsule_depth(X, np.tile(node, (len(dirs), 1)), node + reach[:, None] * dirs, radii)))
        Vn, fair_node_anchored.pinned, _ = contained_fair_solve(
            K, wv, V, fixed, depth, tol * radii.min(), strong * anchor * M.diagonal() / radii.min() ** 4)
        return trimesh.Trimesh(Vn, F, process=False), fixed
    A = (K + sp.diags(wv)).tocsr()
    fr = np.where(~fixed)[0]
    fx = np.where(fixed)[0]
    solve = spl.factorized(A[fr][:, fr].tocsc())
    Vn = V.copy()
    for k in range(3):
        Vn[fr, k] = solve(wv[fr] * V[fr, k] - A[fr][:, fx] @ V[fx, k])
    return trimesh.Trimesh(Vn, F, process=False), fixed


def fair_node(mesh, node, dirs, radii, t_ring, free_len=1.5, cotan=True, core=True, limit=False):
    """Unanchored FHS fairing (first version; necks thick beams, kept for the study): variational fairing of the node. The vertices of the node region (everything that is not beam
    beyond t_ring_i + free_len * r_i) are moved to the thin-plate (minimal bending) surface: bi-Laplacian
    L^2 x = 0 with the beam vertices fixed (two rings of fixed beam give a tangent-continuous join). One sparse
    solve; spreads the curvature of the crevices instead of concentrating it at the beam rings. core: the node keeps
    every beam's own cylinder up to the node point (vertices that the fairing pulls into a core are projected onto it
    and fixed, then the rest is solved again; folds, not used). limit: keep only the normal displacement, limited
    (smooth max) by the clearance to the beam cores: crevices fill, no beam necks on its way into the node.
    Returns (mesh, fixed mask)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    fixed = np.zeros(len(V), bool)
    for e, r, t in zip(dirs, radii, t_ring):
        w = (V - node) @ e
        rad = np.linalg.norm((V - node) - np.outer(w, e), axis=1)
        fixed |= (w > t + free_len * r) & (np.abs(rad - r) < 0.05 * r)
    E = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V),) * 2).tocsr()
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    Lap = cotan_laplacian(V, F) if cotan else sp.eye(len(V)) - sp.diags(1.0 / deg) @ A
    L2 = (Lap @ Lap).tocsr()
    Vn = V.copy()
    for it in range(30):                     # active set: the node may not cut into any beam's core cylinder
        fr = np.where(~fixed)[0]
        fx = np.where(fixed)[0]
        rhs = -L2[fr][:, fx] @ Vn[fx]
        solve = spl.factorized(L2[fr][:, fr].tocsc())
        for k in range(3):
            Vn[fr, k] = solve(rhs[:, k])
        if not core:
            break
        hit = np.zeros(len(V), bool)
        for e, r, t in zip(dirs, radii, t_ring):
            w = (Vn - node) @ e
            q = (Vn - node) - np.outer(w, e)
            rad = np.linalg.norm(q, axis=1)
            inside = (~fixed) & (w > 0) & (w < t + free_len * r) & (rad < r)
            Vn[inside] = node + np.outer(w[inside], e) + q[inside] * (r / np.maximum(rad[inside], 1e-12))[:, None]
            hit |= inside
        if not hit.any():
            break
        fixed |= hit
    if limit:
        # normal displacement only, and never deeper than the clearance to the beam cores (smooth max): the
        # fairing fills the crevices freely but cannot neck a beam on its way into the node
        nrm = mesh.vertex_normals
        dn = ((Vn - V) * nrm).sum(1)
        eps = 0.05 * radii.min()
        cl = core_clearance(V, node, dirs, radii, t_ring + free_len * radii + 2 * radii)
        # inward bound: the clearance where the vertex is outside the cores, ~0 where it is already inside (no
        # outward push: the cores overlap near the node and the crevices of C1 already lie inside their union)
        g = -eps * np.logaddexp(0.0, cl / eps)
        dn = 0.5 * (dn + g + np.sqrt((dn - g) ** 2 + eps * eps))
        dn[fixed] = 0.0
        Vn = V + dn[:, None] * nrm
    return trimesh.Trimesh(Vn, F, process=False), fixed


# ---------------------------------------------------------------- FHS on a whole lattice
def _strut_rings(z_lo, z_hi, r, dz, lead, mid=1.0):
    """Axial positions of a strut's control rings between its two node rings: the first / last two gaps lead * r
    long, one edge length apart in between."""
    a = [z_lo] + ([z_lo + lead * r, z_lo + 2 * lead * r] if lead > 0 else [])
    b = ([z_hi - 2 * lead * r, z_hi - lead * r] if lead > 0 else []) + [z_hi]
    if a[-1] >= b[0] - 0.5 * dz:
        return np.linspace(z_lo, z_hi, max(int(np.ceil((z_hi - z_lo) / dz)), 1) + 1)
    n = max(int(np.ceil((b[0] - a[-1]) / (mid * dz))), 1)          # middle rings `mid` edge lengths apart
    return np.concatenate((a, np.linspace(a[-1], b[0], n + 1)[1:-1], b))


def merge_straight_chains(X, conn, radii, cos_tol=1e-6, r_tol=1e-9, tags=None):
    """Merge struts that continue straight through a degree-2 node with the same radius (element subdivisions, not
    joints): such a node would otherwise get a node patch and its dense blend rings for nothing. Returns
    (conn, radii) of the merged struts (node indices unchanged); with tags (the section of each strut, None for a
    circle), only circular struts merge (a section is given in its strut's frame, which a merged strut may reverse),
    and (conn, radii, tags) is returned."""
    conn = [tuple(c) for c in np.asarray(conn, int)]
    radii = list(np.broadcast_to(np.asarray(radii, float), (len(conn),)))
    tg = None if tags is None else list(tags)

    def same(k0, k1):
        return tg is None or (tg[k0] is None and tg[k1] is None)
    while True:
        deg = {}
        for k, (a, b) in enumerate(conn):
            deg.setdefault(a, []).append(k)
            deg.setdefault(b, []).append(k)
        for n, ks in deg.items():
            if len(ks) != 2 or abs(radii[ks[0]] - radii[ks[1]]) > r_tol or not same(ks[0], ks[1]):
                continue
            ends = [conn[k][1] if conn[k][0] == n else conn[k][0] for k in ks]
            u, v = X[ends[0]] - X[n], X[ends[1]] - X[n]
            if u @ v / (np.linalg.norm(u) * np.linalg.norm(v)) < -1 + cos_tol:
                keep, drop = sorted(ks)
                conn[keep] = (ends[0], ends[1])
                del conn[drop], radii[drop]
                if tg is not None:
                    del tg[drop]
                break
        else:
            return (np.array(conn), np.array(radii)) if tg is None else (np.array(conn), np.array(radii), tg)


PLATE_GAP = 0.005         # mm: no FHS vertex closer than this to a plate plane before the trim


def mirror_at_plates(X, conn, radii, z_range, tol=1e-9):
    """Struts at plate nodes mirrored across their plate plane (struts lying in the plane are already symmetric):
    with the mirrored half added, the FHS node at a plate is symmetric about it, so the trimmed surface meets the
    plate at 90 deg. Without it, a plate node's hull can meet the plate almost tangentially: a grazing wedge that
    gives sliver tets and overlapping facets when the STL is remeshed for FE. It is also the joint of a stacked
    lattice, where a plate node joins the mirrored cell. Returns (X, conn, radii) extended."""
    X = np.asarray(X, float)
    conn = np.asarray(conn, int)
    radii = np.broadcast_to(np.asarray(radii, float), (len(conn),))
    Xn, cn, rn = [X], [conn], [radii]
    nv = len(X)
    for z0 in z_range:
        on = np.abs(X[:, 2] - z0) < 1e-6
        for (a, b), r in zip(conn, radii):
            for p, q in ((a, b), (b, a)):
                if on[p] and abs(X[q, 2] - z0) > 1e-6:          # strut leaves the plate: add its mirror image
                    m = X[q].copy()
                    m[2] = 2 * z0 - m[2]
                    Xn.append(m[None])
                    cn.append(np.array([[p, nv]]))
                    rn.append(np.array([r]))
                    nv += 1
    return np.vstack(Xn), np.vstack(cn), np.concatenate(rn)


def lattice_fhs(X, conn, radii, z_range=None, levels=2, lead=1.0, free_len=1.0, anchor=10.0, n_min=16,
                limit=True, mid=3.0, mirror=False, ball=True, contain=True, tol=0.02, strong=1e1, sections=None):
    """FHS (Faired Hull-Subdivision) node joints for a whole lattice (algorithm: notes/13_FHS_NodeAlgorithm.md). Every node of degree >= 2 gets a node
    patch (convex hull of its node rings + core rings, ring caps removed); every strut a control tube between the
    node rings at its two ends; the whole control mesh is Loop-subdivided, the node regions are faired in one
    anchored least-squares solve, and the result is trimmed to the plate planes z_range. mirror: the struts at
    plate nodes are mirrored across the plate first (symmetric plate joints, meeting the plate at 90 deg; off by
    default: the exactly symmetric node puts vertices on the plate plane and the trim then leaves coincident
    facets that break the FE remesh, 2026-10-08).
    Mesh size (notes/14): 2 Loop levels with the vertices moved onto the exact limit surface (limit), and the
    strut-middle control rings mid edge lengths apart: ~5x fewer faces than 3 plain levels, 99 % of the surface
    within ~10 um of it. sections: per strut (conn order) None (circle of radius radii_i) or a convex section
    contour (m x 2) in the strut frame _frame(e_i), e_i from conn[i][0] to conn[i][1], roll applied; radii_i is then
    its equivalent radius sqrt(A / pi) (notes/17 P6). Returns (watertight manifold, info dict)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from scipy.spatial import ConvexHull, cKDTree
    X = np.asarray(X, float)
    if z_range is not None and mirror:                                 # symmetric joints at the plates
        if sections is not None:
            raise ValueError('lattice_fhs: mirror=True is not supported with sections')
        X, conn, radii = mirror_at_plates(X, conn, np.broadcast_to(np.asarray(radii, float), (len(conn),)), z_range)
    if sections is None:
        conn, radii = merge_straight_chains(X, conn, radii)             # element splits are not joints
        secs = [None] * len(conn)
    else:
        conn, radii, secs = merge_straight_chains(X, conn, radii, tags=sections)
    r_min = radii.min()
    nring = np.array([ring_count(r, r_min, n_min) for r in radii])
    shrink = {n: loop_shrink(n, spacing_ratio=mid, levels=levels, limit=limit) for n in set(nring.tolist())}
    rc = np.array([r / shrink[n] for r, n in zip(radii, nring)])
    cring, hring = [], []                                           # per strut: control ring, hull (end) ring
    for i, (r, n) in enumerate(zip(radii, nring)):
        if secs[i] is None:
            t = np.linspace(0, 2 * np.pi, n, endpoint=False)
            c = rc[i] * np.column_stack((np.cos(t), np.sin(t)))
            cring.append(c)
            hring.append(c)
        else:
            c, nring[i], _ = section_control_ring(secs[i], n_min=n, levels=levels, limit=limit, spacing_ratio=mid)
            cring.append(c)
            hring.append(strictly_convex_ring(c))
    rc = np.array([max(np.linalg.norm(c, axis=1).max(), np.linalg.norm(h, axis=1).max()) for c, h in zip(cring, hring)])
    length = np.linalg.norm(X[conn[:, 1]] - X[conn[:, 0]], axis=1)
    E = (X[conn[:, 1]] - X[conn[:, 0]]) / length[:, None]

    def sec_at(i, k):                                               # strut i's section in the frame of its end k
        if secs[i] is None or k == 0:
            return secs[i]
        u, v = _frame(E[i])
        u2, v2 = _frame(-E[i])
        return np.asarray(secs[i]) @ np.array([[u @ u2, u @ v2], [v @ u2, v @ v2]])
    inc = [[] for _ in X]
    for i, (a, b) in enumerate(conn):
        inc[a].append((i, 0))
        inc[b].append((i, 1))
    # 1-2. trim sphere and node-ring offsets per node
    toff = {}
    for n, ends in enumerate(inc):
        if len(ends) < 2:
            for i, k in ends:
                toff[i, k] = 0.0
            continue
        dirs = np.array([E[i] if k == 0 else -E[i] for i, k in ends])
        rr = np.array([radii[i] if secs[i] is None else np.linalg.norm(secs[i], axis=1).max() for i, _ in ends])
        try:
            R_s, _ = trim_radius(dirs, rr, np.array([length[i] for i, _ in ends]))
        except ValueError:                                         # no overlap at all (e.g. collinear struts)
            R_s = 1.5 * rr.max()
        t = ring_offsets(dirs, np.array([rc[i] for i, _ in ends]), R_s, core=True)
        for (i, k), ti in zip(ends, t):
            toff[i, k] = float(ti)
    for i in range(len(conn)):
        if toff[i, 0] + toff[i, 1] > 0.95 * length[i]:
            raise ValueError(f'strut {i}: node rings overlap ({toff[i, 0]:.2f} + {toff[i, 1]:.2f} > '
                             f'0.95 * {length[i]:.2f})')
    # 3. control tubes
    V, F, nv = [], [], 0
    ring_at = {}                                                   # (strut, end) -> (first vertex index, n)
    for i, (a, b) in enumerate(conn):
        n, e, r = nring[i], E[i], radii[i]
        u, v = _frame(e)
        circ = np.outer(cring[i][:, 0], u) + np.outer(cring[i][:, 1], v)
        circ_h = np.outer(hring[i][:, 0], u) + np.outer(hring[i][:, 1], v)     # end rings: hull faces
        if secs[i] is None:
            dz = 2 * r * np.sin(np.pi / n)
        else:
            P_ = np.vstack((secs[i], secs[i][:1]))
            dz = np.linalg.norm(np.diff(P_, axis=0), axis=1).sum() / n
        zs = _strut_rings(toff[i, 0], length[i] - toff[i, 1], r, dz,
                          lead if secs[i] is None else min(lead, LEAD_SECTION), mid)
        start = nv
        for kz, z in enumerate(zs):
            if secs[i] is None:
                V.append(X[a] + z * e + (circ_h if kz in (0, len(zs) - 1) else circ))
            else:                                                  # blend from both node rings (short struts:
                bl = LEAD_SECTION * r                              # the two blends overlap, no full section)
                bb = _blend((z - zs[0]) / bl) * _blend((zs[-1] - z) / bl)
                V.append(X[a] + z * e + (1 - bb) * circ_h + bb * circ)
            nv += n
        for q in range(len(zs) - 1):
            F.append(_tube_faces(start + q * n, start + (q + 1) * n, n))
        ring_at[i, 0] = (start, n)
        ring_at[i, 1] = (start + (len(zs) - 1) * n, n)
    # 4. node patches
    for nd, ends in enumerate(inc):
        if len(ends) == 1:                                         # free end: flat cap fan
            i, k = ends[0]
            s0, n = ring_at[i, k]
            V.append(X[nd][None])
            j = np.arange(n)
            fan = np.column_stack((s0 + j, np.full(n, nv), s0 + (j + 1) % n))
            F.append(fan if k == 1 else fan[:, ::-1])
            nv += 1
            continue
        idx, ring_of, cores = [], [], []
        for q, (i, k) in enumerate(ends):
            s0, n = ring_at[i, k]
            idx.append(np.arange(s0, s0 + n))
            ring_of.append(np.full(n, q))
        for i, k in ends:                                          # core rings at the node point
            e = E[i] if k == 0 else -E[i]
            n = nring[i]
            if secs[i] is None:
                u, v = _frame(e)
                t = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
                V.append(X[nd] + rc[i] * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)))
            else:                                                  # the hull ring, half a step on, strut frame
                u, v = _frame(E[i])
                ch = hring[i]
                cm = 0.5 * (ch + np.roll(ch, -1, axis=0))
                cm *= np.linalg.norm(ch, axis=1).mean() / np.linalg.norm(cm, axis=1).mean()
                V.append(X[nd] + np.outer(cm[:, 0], u) + np.outer(cm[:, 1], v))
            idx.append(np.arange(nv, nv + n))
            ring_of.append(np.full(n, -1))
            nv += n
        if ball:                                                   # the node's back (notes/17 P3.4)
            Vt = np.vstack(V)
            Pb = ball_points(Vt[np.concatenate(idx)], X[nd], max(rc[i] for i, _ in ends),
                             0.05 * min(radii[i] for i, _ in ends),
                             [E[i] if k == 0 else -E[i] for i, k in ends], [toff[i, k] for i, k in ends])
            if len(Pb):
                V.append(Pb)
                idx.append(np.arange(nv, nv + len(Pb)))
                ring_of.append(np.full(len(Pb), -1))
                nv += len(Pb)
        Vall = np.vstack(V)
        idx = np.concatenate(idx)
        ring_of = np.concatenate(ring_of)
        H = ConvexHull(Vall[idx])
        S = H.simplices
        ro = ring_of[S]
        S = idx[S[~((ro[:, 0] >= 0) & (ro[:, 0] == ro[:, 1]) & (ro[:, 1] == ro[:, 2]))]]
        c = Vall[idx].mean(0)
        nrm = np.cross(Vall[S[:, 1]] - Vall[S[:, 0]], Vall[S[:, 2]] - Vall[S[:, 0]])
        flip = (nrm * (Vall[S].mean(1) - c)).sum(1) < 0
        S[flip] = S[flip][:, ::-1]
        F.append(S)
        V = [Vall]
    ctrl = trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False)
    ctrl.remove_unreferenced_vertices()
    if not ctrl.is_watertight:
        raise RuntimeError('lattice control mesh is not closed')
    trimesh.repair.fix_winding(ctrl)
    if ctrl.volume < 0:
        ctrl.invert()
    # 5. Loop subdivision
    V2, F2 = trimesh.remesh.subdivide_loop(ctrl.vertices, ctrl.faces, iterations=levels)
    if limit:                                                      # vertices onto the limit surface
        V2 = loop_limit(V2, F2)
    # 6. anchored fairing of all node regions in one solve
    fixed = np.zeros(len(V2), bool)
    for i, (a, b) in enumerate(conn):
        w = (V2 - X[a]) @ E[i]
        band = (w > toff[i, 0] + free_len * radii[i]) & (w < length[i] - toff[i, 1] - free_len * radii[i])
        fixed[band] |= np.abs(core_distance(V2[band], X[a], E[i], radii[i], np.inf, secs[i])) < 0.05 * radii[i]
    for nd, ends in enumerate(inc):                                # free ends stay as subdivided
        if len(ends) == 1:
            fixed |= np.linalg.norm(V2 - X[nd], axis=1) < 3 * radii[ends[0][0]]
    flank = np.zeros(len(V2))
    groups = []
    owner = cKDTree(X).query(V2)[1]
    for nd, ends in enumerate(inc):
        sel = np.where(owner == nd)[0]
        if len(ends) < 2 or len(sel) == 0:
            continue
        dirs = np.array([E[i] if k == 0 else -E[i] for i, k in ends])
        rr = np.array([radii[i] for i, _ in ends])
        reach = np.array([toff[i, k] for i, k in ends]) + free_len * rr + 2 * rr
        ss = None if all(secs[i] is None for i, _ in ends) else [sec_at(i, k) for i, k in ends]
        flank[sel] = flank_weights(V2[sel], X[nd], dirs, rr, reach, ss)
        groups.append((nd, sel, dirs, rr, reach, ss))
    Lc = cotan_laplacian(V2, F2)
    area = 0.5 * np.linalg.norm(np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]]), axis=1)
    M = sp.diags(np.bincount(F2.ravel(), weights=np.repeat(area / 3, 3), minlength=len(V2)))
    K = (Lc.T @ M @ Lc).tocsr()
    wv = anchor * M.diagonal() / r_min ** 4 * flank
    fr, fx = np.where(~fixed)[0], np.where(fixed)[0]
    if contain:                                                    # stay outside each node's smooth core union
        pos = np.full(len(V2), -1)                                  # global vertex -> index among the free ones
        pos[fr] = np.arange(len(fr))

        def depth(P):                                               # P: the free vertices (fr order)
            d = np.full(len(P), np.inf)
            proj = P.copy()
            for nd, sel, dirs, rr, reach, ss in groups:
                q = pos[sel]
                q = q[q >= 0]
                if len(q):
                    d[q], proj[q] = smooth_core_depth(P[q], X[nd], dirs, rr, reach, sections=ss)
            return d, proj
        Vn, n_pin, _ = contained_fair_solve(K, wv, V2, fixed, depth, tol * r_min,
                                            strong * anchor * M.diagonal() / r_min ** 4)
    else:
        A = (K + sp.diags(wv)).tocsr()
        solve = spl.factorized(A[fr][:, fr].tocsc())
        Vn = V2.copy()
        for k in range(3):
            Vn[fr, k] = solve(wv[fr] * V2[fr, k] - A[fr][:, fx] @ V2[fx, k])
        n_pin = 0
    if z_range is not None:
        # cut planes: the plates, or (mirror) 2 PLATE_GAP inside the material, where the symmetric surface crosses
        # cleanly instead of having vertices exactly on the symmetry plane
        cuts = [z_range[0] + 2 * PLATE_GAP, z_range[1] - 2 * PLATE_GAP] if mirror else list(z_range)
        # keep vertices >= PLATE_GAP off the cut planes (pushed away on their own side): the trim then never cuts
        # within 5 um of a vertex, so it leaves no micro-edges (they made gmsh's surface reparametrisation hang)
        for z0 in cuts:
            d = Vn[:, 2] - z0
            near = np.abs(d) < PLATE_GAP
            Vn[near, 2] = z0 + np.where(d[near] >= 0, PLATE_GAP, -PLATE_GAP)
    man = m3.Manifold(m3.Mesh(np.ascontiguousarray(Vn, np.float32), np.ascontiguousarray(F2, np.uint32)))
    if man.status() != m3.Error.NoError:
        raise RuntimeError(f'faired lattice is not a manifold: {man.status()}')
    if z_range is not None:                                        # 7. plate planes
        man = man.trim_by_plane((0.0, 0.0, 1.0), float(cuts[0])).trim_by_plane((0.0, 0.0, -1.0), -float(cuts[1]))
        if mirror:                                                 # flat caps back onto the plate planes
            mesh = man.to_mesh64()
            V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
            for zc, z0 in zip(cuts, z_range):
                V[np.abs(V[:, 2] - zc) < 1e-7, 2] = z0
            man = m3.Manifold(m3.Mesh64(np.array(V, np.float64), np.array(mesh.tri_verts, np.uint64)))   # writable copies
    return man, dict(control_faces=len(ctrl.faces), free_vertices=len(fr), ring_offsets=toff, pinned=n_pin)


# ---------------------------------------------------------------- mesh size: deviation between two surfaces
def _point_triangle_dist(P, A, B, C):
    """Distance from points P (n,3) to triangles (A, B, C) (n,3 each), vectorised (Ericson, closest point on
    triangle)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1e-30)
    v = vb / den
    w = vc / den
    Q = A + v[..., None] * ab + w[..., None] * ac                       # interior
    with np.errstate(divide='ignore', invalid='ignore'):
        tab = np.clip(d1 / np.where(d1 - d3 != 0, d1 - d3, 1e-30), 0, 1)
        tac = np.clip(d2 / np.where(d2 - d6 != 0, d2 - d6, 1e-30), 0, 1)
        tbc = np.clip((d4 - d3) / np.where((d4 - d3) + (d5 - d6) != 0, (d4 - d3) + (d5 - d6), 1e-30), 0, 1)
    E1 = A + tab[..., None] * ab
    E2 = A + tac[..., None] * ac
    E3 = B + tbc[..., None] * (C - B)
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    dist = np.where(inside, np.linalg.norm(P - Q, axis=-1), np.inf)
    for E in (E1, E2, E3):
        dist = np.minimum(dist, np.where(inside, np.inf, np.linalg.norm(P - E, axis=-1)))
    return np.where(inside, np.linalg.norm(P - Q, axis=-1), dist)


def surface_distance(P, mesh, ring=2):
    """Distance from points P to the surface of `mesh` (smooth, fine): exact point-triangle distances to the faces
    around the mesh's nearest vertices (`ring` nearest vertices' one-rings). Returns (n,)."""
    from scipy.spatial import cKDTree
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    vf = mesh.vertex_faces                                               # (nv, max valence), -1 padded
    _, nn = cKDTree(V).query(P, k=ring)
    nn = np.atleast_2d(nn.T).T if ring > 1 else nn[:, None]
    best = np.full(len(P), np.inf)
    for j in range(nn.shape[1]):
        fl = vf[nn[:, j]]
        for c in range(fl.shape[1]):
            f = fl[:, c]
            ok = f >= 0
            if not ok.any():
                continue
            T = F[f[ok]]
            d = _point_triangle_dist(P[ok], V[T[:, 0]], V[T[:, 1]], V[T[:, 2]])
            best[ok] = np.minimum(best[ok], d)
    return best


def deviation(dense, coarse, n_samples=None, seed=0, planes_z=()):
    """Two-sided max distance between the dense and the coarse surface: dense vertices to the coarse surface, and
    samples on the coarse faces (several per face) to the dense surface. Points on the plate planes planes_z
    (identical flat faces in both, but large triangles the vertex-ring search cannot cover) are left out."""
    def off(P):
        keep = np.ones(len(P), bool)
        for z in planes_z:
            keep &= np.abs(P[:, 2] - z) > 1e-4
        return keep
    Pd = np.asarray(dense.vertices)
    d1 = surface_distance(Pd[off(Pd)], coarse)
    rng = np.random.default_rng(seed)
    n = n_samples or 4 * len(coarse.faces)
    fi = rng.integers(0, len(coarse.faces), n)
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    T = coarse.vertices[coarse.faces[fi]]
    P = T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])
    d2 = surface_distance(P[off(P)], dense)
    return float(max(d1.max(), d2.max()))
