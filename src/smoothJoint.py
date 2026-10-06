"""Smooth (filleted) lattice joints for printing: rolling-ball fillets on the signed-distance-field union of the struts.

Each strut is its own section swept along the beam (circle: a capsule of its own radius; polygon: the section with
its corners rounded, swept, with a smooth rounded end), with no max-radius joint sphere. The joint is the
morphological closing of the strut union by a ball of radius k = `fillet` × the smallest strut radius at the joint
(constant-radius rolling-ball fillet; Rossignac & Requicha). It is evaluated per point in closed form, without a
second distance transform:
- each strut is linearised at the point by the tangent half-space of its exact (convex) distance field;
- the field is k - (distance to the intersection of the offset half-spaces {a_i >= k}, the allowed ball centres),
  found by projection onto 1-3 planes (_ball_fill).
Concave crevices get an exact constant-radius fillet for locally planar surfaces at any angle. Convex sides and
tangent / collinear junctions keep exactly the plain union, so no material is added where none is needed (a smooth
min / round union bulges there: +2 % for two collinear struts merging). The result does not depend on the strut
order and is C1.

Adaptive surface (default): marching cubes only in a patch around every joint (cell = smallest strut size there / 10).
The vertices are relaxed (tangential smoothing + projection onto the exact field) to remove the marching-cubes
slivers. Each patch is cut flat where the fillets have died out, and the strut middles are prisms stitched between
the cut loops of the two end patches. The mesh is therefore fine at the joints and coarse along the straight struts.
Nodes joined by a strut too short for two patches are meshed as one patch. adaptive=False (and struts that come
close away from a node) uses one uniform grid. manifold3d does the plate-plane trim and the watertight check.
Method notes, iterations and metrics: notes/11_PrintJointGeometry.md.

User parameter: `fillet` (fillet radius / smallest strut radius at the joint, default 0.5). `resolution` (cell size)
is optional.
"""
import numpy as np

DIV = 10            # joint-patch cells per strut size (adaptive); the uniform grid uses 8


# ------------------------------------------------------------------ strut SDFs
def _capsule_sdf(P, a, b, r):
    """Signed distance (negative inside) of points P (n,3) to the capsule a-b of radius r."""
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - r


def _polygon_sdf(Q, S):
    """Signed distance of 2D points Q (n,2) to the simple polygon S (m,2), negative inside."""
    d2 = np.full(len(Q), np.inf)
    inside = np.zeros(len(Q), bool)
    for p, q in zip(S, np.roll(S, -1, 0)):
        e = q - p
        w = Q - p
        t = np.clip((w @ e) / (e @ e), 0.0, 1.0)
        d2 = np.minimum(d2, ((w - t[:, None] * e) ** 2).sum(1))
        crosses = (p[1] > Q[:, 1]) != (q[1] > Q[:, 1])
        xint = p[0] + (Q[:, 1] - p[1]) * e[0] / np.where(e[1] == 0, 1.0, e[1])
        inside ^= crosses & (Q[:, 0] < xint)
    return np.where(inside, -1.0, 1.0) * np.sqrt(d2)


def _strut_sdf_grad(P, s, a, b):
    """(SDF, unit gradient) of strut s between a and b: analytic for a capsule, finite differences for a polygon."""
    if s['S'] is None:
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        w = P - (a + t[:, None] * ab)
        n = np.linalg.norm(w, axis=1)
        return n - s['R'], w / np.maximum(n, 1e-12)[:, None]
    d = _strut_sdf(P, s, a, b)
    eps = 1e-4 * s['size']
    g = np.column_stack([(_strut_sdf(P + eps * np.eye(3)[k], s, a, b) - d) / eps for k in range(3)])
    return d, g / np.maximum(np.linalg.norm(g, axis=1), 1e-12)[:, None]


def _strut_sdf(P, s, a=None, b=None):
    """SDF of strut s (dict from _struts) between a and b (default its own end points)."""
    a = s['a'] if a is None else a
    b = s['b'] if b is None else b
    if s['S'] is None:
        return _capsule_sdf(P, a, b, s['R'])
    e = (b - a) / np.linalg.norm(b - a)
    W = P - a
    t = W @ e
    d = s['f2'](np.column_stack((W @ s['ex'], W @ s['ey'])))
    L = np.linalg.norm(b - a)
    # smooth rounded end past the end points (C1, the capsule for a circle): the section inset by rho, swept and
    # rounded by rho, rho = equivalent radius (>= inradius, so d + rho >= 0 and it equals d along the strut)
    rho = s['size']
    x = np.maximum(d + rho, 0.0)
    tt = np.where(t < 0, t, np.maximum(t - L, 0.0))
    return np.sqrt(x * x + tt * tt) - rho


def _round_section(S, frac=0.25):
    """Section corners (convex and concave) rounded with radius frac * equivalent radius = 2.5 joint-mesh cells
    (closing + opening offsets): a smooth surface the mesh can resolve, so the relaxation cannot fold at a sharp
    edge, and no re-entrant section corner. Area change (1 - pi/4) * rho^2 per 90-deg corner (rectangle 1.2 x 0.6:
    -1.4 %); features finer than rho (thin slots, sharp lobes) are smoothed out. Returns (polygon, area)."""
    import manifold3d as m3
    cs = m3.CrossSection([S.tolist()], m3.FillRule.NonZero)
    rho = frac * np.sqrt(cs.area() / np.pi)
    cs = cs.offset(rho, m3.JoinType.Round, 2.0, 32).offset(-2 * rho, m3.JoinType.Round, 2.0, 32) \
           .offset(rho, m3.JoinType.Round, 2.0, 32)
    P = np.asarray(max(cs.to_polygons(), key=len), float)
    d = np.linalg.norm(np.diff(np.vstack((P, P[:1])), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(d)))
    t = np.linspace(0, s[-1], 257)[:-1]                          # 256 points by arc length
    Pc = np.vstack((P, P[:1]))
    return np.column_stack([np.interp(t, s, Pc[:, j]) for j in (0, 1)]), float(cs.area())


def _section_field(S, R, size, n=384):
    """2D signed distance of section S, precomputed on an n x n raster (half-width 2R + size) and evaluated by cubic
    interpolation: smooth (C2) and fast. Outside the raster: |q| - R (a lower bound; far beyond any fillet)."""
    from scipy.ndimage import map_coordinates, spline_filter
    key = S.tobytes()
    if key not in _FIELDS:
        E = 2 * R + size
        g = np.linspace(-E, E, n)
        G = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
        coef = spline_filter(_polygon_sdf(G, S).reshape(n, n), order=3, mode='nearest')
        step = g[1] - g[0]

        def f(Q):
            d = map_coordinates(coef, ((Q + E) / step).T, order=3, mode='nearest', prefilter=False)
            return np.where((np.abs(Q) > E - 2 * step).any(1), np.linalg.norm(Q, axis=1) - R, d)
        _FIELDS[key] = f
    return _FIELDS[key]


_FIELDS = {}            # section fields by polygon (struts with the same section share one)


def _struts(X, conn, sections, orientations=None):
    """Strut dicts: end points, section S (None = circle), frame, size (equivalent radius), R (circumradius).
    sections[i]: a radius, or a polygon ((m,2) or flattened) in the frame of orientations[i]."""
    out = []
    for i, (n0, n1) in enumerate(conn):
        a, b = X[n0], X[n1]
        if np.ndim(sections[i]) == 0:
            r = float(sections[i])
            out.append(dict(a=a, b=b, S=None, size=r, R=r))
            continue
        S = np.asarray(sections[i], float).reshape(-1, 2)
        ez = (b - a) / np.linalg.norm(b - a)
        v = np.asarray(orientations[i], float) if orientations is not None else np.array([0.0, 0.0, 1.0])
        v = v - (v @ ez) * ez
        if np.linalg.norm(v) < 1e-8:
            v = np.cross(ez, [1, 0, 0] if abs(ez[0]) < 0.9 else [0, 1, 0])
        ey = v / np.linalg.norm(v)
        ex = np.cross(ey, ez)                                    # same frame as BeamStructureSTL.local_frame
        S, A = _round_section(S)
        size, R = float(np.sqrt(A / np.pi)), float(np.linalg.norm(S, axis=1).max())
        out.append(dict(a=a, b=b, S=S, ex=ex, ey=ey, size=size, R=R, f2=_section_field(S, R, size)))
    return out


# ------------------------------------------------------------------ grid SDF + marching cubes
def _fillet_k(struts, fillet):
    """Fillet radius of a joint: fillet * the smallest strut size there. One radius per joint keeps the field smooth
    (a radius switching between struts inside a three-strut blend leaves a crease)."""
    return max(fillet * min(s['size'] for s in struts), 1e-9)


def _ball_fill(P, struts, k, ends=None):
    """Rolling-ball filleted union at points P (n,3): the morphological closing of the strut union by a ball of
    radius k, with each strut linearised at the point (tangent half-space of its exact distance field). The allowed
    ball centres are W = intersection of {a_i >= k}. The result is k - dist(P, W) outside W and min_i a_i inside it.
    - Concave crevices get the exact constant-radius fillet for locally planar surfaces, at any angle.
    - Convex sides and tangent junctions keep exactly the plain union, so no material is added where none is needed.
    - Symmetric in the struts (no blending order). C1 outside W.
    - Each distance field is convex, so the tangent half-space lies inside the true {a_i >= k}: the linearised W is
      conservative (fills at least the true closing near the joint).
    dist(P, W) is the nearest feasible projection y onto the planes of 1, 2 or 3 struts within 2k of the point (near
    the surface |y - x| ~ k, so a farther strut cannot bind; deeper inside D stays negative either way).
    Returns (D, unit gradient): (y - x) / |y - x| in the fillet, the nearest strut's normal elsewhere."""
    from itertools import combinations
    n, m = len(P), len(struts)
    A = np.full((m, n), np.inf)
    Nr = np.zeros((m, n, 3))
    for i, s in enumerate(struts):
        a, b = ends[i] if ends is not None else (s['a'], s['b'])
        pad = 1.25 * s['R'] + 3 * k
        sel = np.where(((P >= np.minimum(a, b) - pad) & (P <= np.maximum(a, b) + pad)).all(1))[0]
        A[i, sel], Nr[i, sel] = _strut_sdf_grad(P[sel], s, a, b)
    j = A.argmin(0)
    D = A[j, np.arange(n)]
    G = Nr[j, np.arange(n)]
    pq = k - A                                             # > 0: the point is inside that strut's k-dilation
    act = A < 2.05 * k
    need = np.where((pq > 0).any(0))[0]
    if len(need) == 0:
        return D, G
    best = np.full(len(need), np.inf)
    bdy = np.zeros((len(need), 3))
    pq_n, act_n, N_n = pq[:, need], act[:, need], Nr[:, need]
    tol = 1e-9 * k
    for size in (1, 2, 3):
        for S in combinations(range(m), size):
            rows = np.where(act_n[list(S)].all(0))[0]
            if len(rows) == 0:
                continue
            ns = N_n[list(S)][:, rows]                     # (size, r, 3)
            ps = pq_n[list(S)][:, rows]
            if size == 1:
                dy, ok = ps[0][:, None] * ns[0], np.ones(len(rows), bool)
            elif size == 2:                                # midpoint / half-difference form: no 1/(1 - c^2) loss
                mid, half = 0.5 * (ns[0] + ns[1]), 0.5 * (ns[0] - ns[1])
                om = 2.0 * (half * half).sum(-1)           # 1 - c
                ok = om > 1e-12
                om = np.where(ok, om, 1.0)
                dy = ((ps[0] + ps[1]) / (2.0 - om))[:, None] * mid + ((ps[0] - ps[1]) / om)[:, None] * half
            else:
                M = np.transpose(ns, (1, 0, 2))            # rows = plane normals
                ok = np.abs(np.linalg.det(M)) > 1e-6
                Ms = np.where(ok[:, None, None], M, np.eye(3))
                dy = np.linalg.solve(Ms, ps.T[:, :, None])[:, :, 0]
            # feasible: every active plane satisfied at y = x + dy
            feas = ok & (((np.einsum('mrj,rj->mr', N_n[:, rows], dy) - pq_n[:, rows]) >= -tol) | ~act_n[:, rows]).all(0)
            dist = np.where(feas, np.linalg.norm(dy, axis=1), np.inf)
            better = dist < best[rows]
            best[rows[better]] = dist[better]
            bdy[rows[better]] = dy[better]
    f = np.isfinite(best) & (best > 0)
    D[need[f]] = k - best[f]
    G[need[f]] = bdy[f] / best[f][:, None]
    return D, G


def _sdf_points(P, struts, fillet, ends=None, k=None, grad=False):
    """Filleted union at points P (n,3) (and its unit gradient if grad); k defaults to the joint fillet radius."""
    k = _fillet_k(struts, fillet) if k is None else k
    D = np.ones(len(P))                                    # far from every strut (never on the surface)
    G = np.zeros((len(P), 3))
    for c0 in range(0, len(P), 200000):
        sl = slice(c0, c0 + 200000)
        d, g = _ball_fill(P[sl], struts, k, ends)
        fin = np.isfinite(d)
        D[sl] = np.where(fin, d, 1.0)
        G[sl] = np.where(fin[:, None], g, 0.0)
    return (D, G) if grad else D


def _sdf_grid(struts, lo, h, n, fillet, ends=None):
    """Filleted union on the grid lo + h*index (shape n), block by block (only the struts near each block)."""
    k = _fillet_k(struts, fillet)
    D = np.full(n, 10 * h)
    B = 24
    ab = [ends[i] if ends is not None else (s['a'], s['b']) for i, s in enumerate(struts)]
    for i0 in range(0, n[0], B):
        for j0 in range(0, n[1], B):
            for k0 in range(0, n[2], B):
                idx = [np.arange(o, min(o + B, n[d])) for d, o in enumerate((i0, j0, k0))]
                blo = lo + h * np.array([x[0] for x in idx])
                bhi = lo + h * np.array([x[-1] for x in idx])
                near = [i for i, s in enumerate(struts)
                        if (np.minimum(*ab[i]) - 1.25 * s['R'] - 3 * k <= bhi).all()
                        and (np.maximum(*ab[i]) + 1.25 * s['R'] + 3 * k >= blo).all()]
                if not near:
                    continue
                g = np.stack(np.meshgrid(*[lo[d] + h * idx[d] for d in range(3)], indexing='ij'), -1)
                d = _sdf_points(g.reshape(-1, 3), [struts[i] for i in near], fillet, [ab[i] for i in near], k)
                D[np.ix_(*idx)] = d.reshape(g.shape[:3])
    return D


def _relax(V, F, sdf, h, n_iter=8, loops=()):
    """Marching-cubes clean-up: tangential smoothing (removes the sliver triangles that make spurious creases), each step followed by a projection back onto the exact zero level set of sdf. loops: boundary loops
    [(vertex indices in order, plane normal e, plane offset c)] whose vertices slide along the loop in their plane."""
    import scipy.sparse as sp
    M = sp.csr_matrix((np.ones(F.size), (F.ravel(), np.repeat(np.arange(len(F)), 3))), shape=(len(V), len(F)))

    def project(V):
        for _ in range(2 if loops else 1):
            f, g = sdf(V)
            V = V - (f / np.maximum((g * g).sum(1), 1e-12))[:, None] * g
            for lp, e, c in loops:
                V[lp] -= np.outer(V[lp] @ e - c, e)
        return V, g

    V, g = project(V)
    for _ in range(n_iter):
        n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        # area-weighted centroid of the incident faces (a uniform Laplacian collapses triangles at valence-4 vertices)
        T = V[F]
        w = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        L = (M @ (w[:, None] * T.mean(1))) / np.maximum(M @ w, 1e-30)[:, None] - V
        L -= (L * n).sum(1, keepdims=True) * n
        for lp, e, c in loops:
            Ll = 0.5 * (V[np.roll(lp, 1)] + V[np.roll(lp, -1)]) - V[lp]
            L[lp] = Ll - np.outer(Ll @ e, e)
        V, g = project(V + 0.5 * L)
    return V


def _mc_manifold(D, lo, h, sdf=None):
    """Closed manifold of {D < 0} by marching cubes (grid origin lo, spacing h), relaxed onto sdf if given."""
    import manifold3d as m3
    from skimage.measure import marching_cubes
    D = np.pad(D, 1, constant_values=max(D.max(), h))            # closed surface at the box faces
    # keep the surface off the grid points (|D| >= 0.1h): no near-degenerate triangles, which smoothing cannot open
    # (the relaxation projects the vertices back onto the exact surface)
    D = np.where(np.abs(D) < 0.1 * h, np.where(D < 0, -0.1 * h, 0.1 * h), D)
    V, F, _, _ = marching_cubes(D, level=0.0, spacing=(h, h, h))
    V = V + lo - h
    if sdf is not None:
        V = _relax(V, F, sdf, h)
    # skimage orders the triangles with normals toward increasing D (outside), i.e. outward already
    solid = m3.Manifold(m3.Mesh64(np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.uint64)))
    if solid.status() != m3.Error.NoError or solid.is_empty():
        raise RuntimeError(f'marching-cubes surface is not a valid manifold: {solid.status()}')
    return solid


# ------------------------------------------------------------------ adaptive: joint patches + stitched tubes
def _frame(e):
    u = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(e, u)


def _cut_length(node, s, e, others, k, h):
    """Distance from the node at which the fillets on strut s have died out and a cut disk (radius R + 2h, depth
    R + 6h) holds strut s only. others = [(strut, a, b)]: the other struts of the patch; k: its fillet radius."""
    u, v = _frame(e)
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.vstack([np.zeros((1, 3))] + [rho * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), v))
                                           for rho in (s['R'], s['R'] + 2 * h)])
    L = h
    while L < 50 * s['R']:
        P = np.vstack([node + (L + dt) * e + ring for dt in np.linspace(0, s['R'] + 6 * h, 4)])
        if all(_strut_sdf(P, t, ta, tb).min() > 2 * k + h for t, ta, tb in others):   # beyond any ball fill
            return L
        L += 0.5 * h
    return L


def _boundary_loops(F):
    """Ordered boundary loops (vertex index lists, in the faces' edge direction) of an open triangle mesh."""
    E = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    have = set(map(tuple, E))
    nxt = {a: b for a, b in E if (b, a) not in have}
    loops = []
    while nxt:
        a0, b = nxt.popitem()
        loop = [a0]
        while b != a0:
            loop.append(b)
            b = nxt.pop(b)
        loops.append(loop)
    return loops


def _clip_ears(F, loops):
    """Remove boundary 'ears': a loop vertex in a single triangle (with its two loop neighbours). Once its three
    vertices sit on the cut plane that triangle is degenerate, so it is dropped and the loop skips the vertex."""
    F = np.asarray(F)
    while True:
        cnt = np.bincount(F.ravel(), minlength=F.max() + 1)
        drop, new = set(), {}
        for ie, lp in loops.items():
            ear = cnt[lp] == 1
            ear &= ~(ear & np.roll(ear, 1))          # never two neighbouring ears in one pass
            new[ie] = lp[~ear]
            drop.update(lp[ear].tolist())
        if not drop:
            return F, loops
        F = F[~np.isin(F, list(drop)).any(1)]
        loops = new


def _collapse_near_cut(V, F, loops, planes, tol):
    """Collapse interior vertices lying within tol of their loop's cut plane onto the adjacent loop vertex: the
    boolean cut leaves sliver triangles there (marching-cubes vertices just off the plane). Link condition checked."""
    F = np.asarray(F).copy()
    onloop = np.zeros(len(V), bool)
    for lp in loops.values():
        onloop[lp] = True
    near = onloop.copy()
    for ie in loops:
        e, c = planes[ie]
        near |= np.abs(V @ e - c) < tol
    for _ in range(10):
        sub = np.where(near[F].any(1))[0]                       # faces of the loop / near-plane vertices only
        nbr, vf = {}, {}
        for k in sub:
            for a in F[k]:
                nbr.setdefault(a, set()).update(F[k])
                vf.setdefault(a, []).append(k)
        repl, busy, dead = {}, set(), set()
        for ie, lp in loops.items():
            e, c = planes[ie]
            for b in lp:
                for v in nbr[b] - {b}:
                    if onloop[v] or v in busy or b in busy or abs(V[v] @ e - c) >= tol:
                        continue
                    shared = [k for k in vf[v] if b in F[k]]
                    if len(shared) != 2 or len((nbr[v] & nbr[b]) - {v, b}) != 2:
                        continue
                    repl[v] = b
                    dead.update(shared)
                    busy.update(nbr[v] | nbr[b])
        if not repl:
            return F
        keep = np.ones(len(F), bool)
        keep[list(dead)] = False
        F = F[keep]
        for v, b in repl.items():
            F[F == v] = b
    return F


def _strip(P, sP, Q, sQ):
    """Triangle strip between closed loops P and Q (index arrays, arc-length parameters with the closing 1.0
    appended), both starting at parameter 0 and running the same way round. Tube edges P_i -> P_i+1, Q_j+1 -> Q_j."""
    nP, nQ = len(P), len(Q)
    F, i, j = [], 0, 0
    while i < nP or j < nQ:
        if j == nQ or (i < nP and sP[i + 1] <= sQ[j + 1]):
            F.append((P[i % nP], P[(i + 1) % nP], Q[j % nQ]))
            i += 1
        else:
            F.append((P[i % nP], Q[(j + 1) % nQ], Q[j % nQ]))
            j += 1
    return F


def _tube(P, Q, V, u, v, e, delta, nv):
    """Strut middle between the cut loops P (patch at the start, tube edges P_i -> P_i+1) and Q (patch at the end).
    Two inner rings at +-delta from the cuts carry the merged arc-length parameters of both loops, so each loop is
    stitched to a near-aligned ring and the straight middle is an exact prism of aligned quads (no zigzag across
    the corners of a non-circular section). Returns (new vertices, faces); new vertex indices start at nv."""
    def start(I):
        W = V[I] - V[I].mean(0)
        I = np.roll(I, -int(np.argmin(np.abs(np.arctan2(W @ v, W @ u)))))
        d = np.linalg.norm(np.diff(V[np.append(I, I[0])], axis=0), axis=1)
        return I, np.concatenate(([0.0], np.cumsum(d))) / d.sum()
    P, sP = start(np.asarray(P))
    Q, sQ = start(np.asarray(Q))
    sM = np.unique(np.round(np.concatenate((sP[:-1], sQ[:-1])), 12))

    def ring(I, s_, shift):
        C = V[np.append(I, I[0])]
        return np.column_stack([np.interp(sM, s_, C[:, k]) for k in range(3)]) + shift * e
    M1 = nv + np.arange(len(sM))
    M2 = M1 + len(sM)
    sMc = np.append(sM, 1.0)
    F = _strip(P, sP, M1, sMc) + _strip(M1, sMc, M2, sMc) + _strip(M2, sMc, Q, sQ)
    return np.vstack((ring(P, sP, delta), ring(Q, sQ, -delta))), F


class _Fallback(Exception):
    pass


def _check_isolated(X, conn, struts, fillet):
    """Struts may only blend at shared nodes (else the patches would miss a fusion): raise _Fallback otherwise.
    Two strut surfaces blend where their axes are closer than R_i + R_j + fillet * min(size)."""
    A, B = np.array([s['a'] for s in struts]), np.array([s['b'] for s in struts])
    R = np.array([s['R'] for s in struts])
    size = np.array([s['size'] for s in struts])
    m = R * (1 + fillet)
    lo, hi = np.minimum(A, B) - m[:, None], np.maximum(A, B) + m[:, None]
    t = np.linspace(0, 1, 65)
    for i in range(len(struts)):
        cand = np.where((lo <= hi[i]).all(1) & (hi >= lo[i]).all(1))[0]
        cand = [j for j in cand if j > i and not set(conn[i]) & set(conn[j])]
        P = A[i] + t[:, None] * (B[i] - A[i])
        for j in cand:
            if _capsule_sdf(P, A[j], B[j], 0.0).min() < R[i] + R[j] + fillet * min(size[i], size[j]):
                raise _Fallback(f'struts {i} and {j} come close away from a node')


def _adaptive(X, conn, struts, fillet, resolution):
    """Joint patches (fine marching cubes, cut flat where the fillets end) + ruled tubes between the cuts. Nodes
    joined by a strut too short for two patches are meshed as one patch (cluster)."""
    import manifold3d as m3
    _check_isolated(X, conn, struts, fillet)
    nn = len(X)
    root = list(range(nn))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    length = [np.linalg.norm(s['b'] - s['a']) for s in struts]
    dirs = {}
    for i, s in enumerate(struts):
        e = (s['b'] - s['a']) / length[i]
        dirs[i, 0], dirs[i, 1] = e, -e
    while True:                                                    # grow clusters until every tube has length
        groups = {}
        for n in range(nn):
            groups.setdefault(find(n), []).append(n)
        patches, cuts = [], {}
        for nodes in groups.values():
            ns = set(nodes)
            inner = [i for i, (a, b) in enumerate(conn) if a in ns and b in ns]
            ends = [(i, k) for i, ab in enumerate(conn) for k in (0, 1) if ab[k] in ns and ab[1 - k] not in ns]
            if not inner and not ends:
                continue
            h = float(resolution) if resolution else min(struts[i]['size'] for i in inner + [i for i, _ in ends]) / DIV
            seg = [(struts[i], struts[i]['a'], struts[i]['b']) for i in inner]
            seg += [(struts[i], X[conn[i][k]], X[conn[i][k]] + 10 * struts[i]['R'] * dirs[i, k]) for i, k in ends]
            cut = {}
            for q, (i, k) in enumerate(ends):
                cut[i, k] = _cut_length(X[conn[i][k]], struts[i], dirs[i, k], seg[:len(inner) + q] +
                                        seg[len(inner) + q + 1:], _fillet_k([t for t, _, _ in seg], fillet), h)
            cuts.update(cut)
            patches.append((nodes, inner, ends, cut, h))
        patch_h = [p[4] for p in patches]
        short = [i for i in range(len(struts)) if (i, 0) in cuts and (i, 1) in cuts and
                 cuts[i, 0] + cuts[i, 1] > length[i] - 0.5 * min(patch_h)]
        if not short:
            break
        for i in short:
            root[find(conn[i][0])] = find(conn[i][1])

    Vs, Fs, loops, nv = [], [], {}, 0
    for nodes, inner, ends, cut, h in patches:
        loc = [struts[i] for i in inner] + [struts[i] for i, _ in ends]
        tips = [(struts[i]['a'], struts[i]['b']) for i in inner]
        tips += [(X[conn[i][k]], X[conn[i][k]] + (cut[i, k] + 3 * h) * dirs[i, k]) for i, k in ends]
        ext = max([cut[ie] + struts[ie[0]]['R'] + 6 * h for ie in ends] +
                  [struts[i]['R'] * (1 + fillet) + 3 * h for i in inner])
        ext = np.ceil(ext / h) * h
        lo = X[nodes].min(0) - ext - 0.3819660 * h                 # no grid plane on a node or cut plane
        n = np.ceil((X[nodes].max(0) + ext - lo) / h).astype(int) + 2
        sdf = lambda P, loc=loc, tips=tips: _sdf_points(P, loc, fillet, tips, grad=True)
        J = _mc_manifold(_sdf_grid(loc, lo, h, n, fillet, tips), lo, h, sdf)
        planes = {ie: (dirs[ie], X[conn[ie[0]][ie[1]]] @ dirs[ie] + cut[ie]) for ie in ends}
        if ends:
            cutters = []
            for ie in ends:
                R = struts[ie[0]]['R']
                u, v = _frame(dirs[ie])
                c = m3.Manifold.cylinder(R + 5 * h, R + 2 * h, R + 2 * h, 48)
                o = X[conn[ie[0]][ie[1]]] + cut[ie] * dirs[ie]
                cutters.append(c.transform(np.column_stack((u, v, dirs[ie], o)).tolist()))
            J = J - m3.Manifold.batch_boolean(cutters, m3.OpType.Add)
        mesh = J.to_mesh64()
        V = np.asarray(mesh.vert_properties)[:, :3].astype(float)
        F = np.asarray(mesh.tri_verts, np.int64)
        on = {ie: np.abs(V @ e - c) < 1e-6 * h for ie, (e, c) in planes.items()}
        cap = np.zeros(len(F), bool)
        for ie in ends:
            cap |= on[ie][F].all(1)
        F = F[~cap]
        found = {}
        for lp in _boundary_loops(F):
            ie = [ie for ie in ends if on[ie][lp].all()]
            if len(ie) != 1 or ie[0] in found:
                raise _Fallback(f'nodes {nodes}: unexpected cut loop')
            found[ie[0]] = np.asarray(lp)
        if len(found) != len(ends):
            raise _Fallback(f'nodes {nodes}: missing cut loop')
        F = _collapse_near_cut(V, F, found, planes, 0.2 * h)
        F, found = _clip_ears(F, found)
        used = np.unique(F)
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        V, F = V[used], remap[F]
        found = {ie: remap[lp] for ie, lp in found.items()}
        if ends:                                                   # the cut leaves slivers along the loops
            V = _relax(V, F, sdf, h, loops=[(found[ie], *planes[ie]) for ie in ends])
        loops.update({ie: lp + nv for ie, lp in found.items()})
        Vs.append(V)
        Fs.append(F + nv)
        nv += len(V)
    V = np.vstack(Vs)
    hs = {ie: h for _, _, ends, _, h in patches for ie in ends}
    for i in range(len(struts)):
        if (i, 0) in loops:
            u, v = _frame(dirs[i, 0])
            Pl, Ql = loops[i, 0][::-1], loops[i, 1]
            gap = length[i] - cuts[i, 0] - cuts[i, 1]
            Vt, Ft = _tube(Pl, Ql, V, u, v, dirs[i, 0], min(hs[i, 0], hs[i, 1], gap / 3), len(V))
            V = np.vstack((V, Vt))
            Fs.append(np.asarray(Ft, np.int64))
    F = np.vstack(Fs)
    solid = m3.Manifold(m3.Mesh64(np.ascontiguousarray(V), np.ascontiguousarray(F, np.uint64)))
    if solid.status() != m3.Error.NoError or solid.is_empty():
        raise _Fallback(f'stitched mesh invalid: {solid.status()}')
    return solid


def build_smooth_stl(X, conn, sections, stl_filename=None, fillet=0.5, resolution=None, z_range=None,
                     orientations=None, adaptive=True):
    """Watertight STL of the lattice with filleted joints. sections: radius per strut (or one radius), or a polygon
    (m,2) per strut in the frame of `orientations` (as BeamStructureSTL). Returns the trimesh (re-read if written)."""
    import manifold3d as m3
    import trimesh
    X = np.asarray(X, float)
    conn = np.asarray(conn, int)
    if np.isscalar(sections):
        sections = np.full(len(conn), float(sections))
    struts = _struts(X, conn, sections, orientations)
    if adaptive:
        try:
            solid = _adaptive(X, conn, struts, fillet, resolution)
        except _Fallback as err:
            print(f'smooth joints: uniform grid ({err})')
            adaptive = False
    if not adaptive:
        h = float(resolution) if resolution else min(s['size'] for s in struts) / 8.0
        pad = max(s['R'] for s in struts) * (1 + fillet) + 2 * h
        lo = X.min(0) - pad - 0.3819660 * h
        n = np.ceil((X.max(0) + pad - lo) / h).astype(int) + 1
        solid = _mc_manifold(_sdf_grid(struts, lo, h, n, fillet), lo, h, lambda P: _sdf_points(P, struts, fillet, grad=True))
    if z_range is not None:
        bb = solid.bounding_box()
        slab = m3.Manifold.cube((bb[3] - bb[0] + 2, bb[4] - bb[1] + 2, z_range[1] - z_range[0]))
        solid = solid ^ slab.translate((bb[0] - 1, bb[1] - 1, z_range[0]))
    if solid.status() != m3.Error.NoError or solid.is_empty():
        raise RuntimeError(f'smooth lattice is not a valid manifold: {solid.status()}')
    # collapse edges below float32 resolution: an STL stores float32, and such an edge would be merged on re-read
    solid = solid.simplify(4 * float(np.spacing(np.float32(np.abs(X).max() + 1.0))))
    mesh = solid.to_mesh64()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts, np.int64), process=False)
    if stl_filename is not None:
        tm.export(stl_filename)
        tm = trimesh.load(stl_filename, process=True)
    return tm
