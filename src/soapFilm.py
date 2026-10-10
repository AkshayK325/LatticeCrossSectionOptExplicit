"""Soap-film node (SF), our implementation of G. Luo, Q. Zou, "Soap Film-inspired Subdivisional Lattice Structure
Construction", arXiv:2504.08847 (2025), notes/Publications/SoapBubbleMethod.pdf. Used as the published baseline
for FHS (notes/16_SoapFilm_vs_FHS.md). Paper section in brackets.

    minimum cutting [3.2] -> spherical Voronoi film [3.3.1] -> 3-row upsampling + Laplacian fairing [3.3.2]
    -> PN-Loop subdivision x3 with circle sampling on the cut circles [3.4] -> + retained cylinders

The paper treats equal radii only. For unequal radii the pairwise cut uses the natural generalisation
d_ij = (r_i cos th + r_j) / sin th (= r / tan(th/2) for equal radii), and Voronoi points are put on the union of the
cylinders (ray exit), which equals the paper's line-cylinder intersection for equal radii.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def _frame(e):
    u = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(e, u)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _ang(a, b):
    return np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))


# ---------------------------------------------------------------------------------------------- [3.2] cutting
def cut_lengths(E, r, lam=0.3):
    """d_i = (1 + lam) max_j d_ij, d_ij = (r_i cos th + r_j) / sin th  (Eqs. 3-5; equal r: r / tan(th / 2))."""
    n = len(E)
    d = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                th = _ang(E[i], E[j])
                d[i] = max(d[i], (r[i] * np.cos(th) + r[j]) / np.sin(th))
    return (1 + lam) * d


# ------------------------------------------------------------------------------------ [3.3.1] spherical Voronoi
def spherical_voronoi(E, tol=1e-7):
    """Voronoi diagram of the unit directions E on the sphere: vertices (unit), and edges (a, b, i, j) between
    vertices a, b separating cells i, j. Co-circular generators (square Delaunay faces) give merged vertices; all
    generators on one circle (valence 3, planar nodes) give the two poles and lune cells."""
    n = len(E)
    c = E.mean(0)
    w, U = np.linalg.eigh((E - c).T @ (E - c))
    if n == 3 or w[0] < tol:                                       # all on one circle: poles + lunes
        p = U[:, 0] if (E @ U[:, 0]).mean() >= 0 else -U[:, 0]
        pv, pw = _frame(p)
        order = np.argsort(np.arctan2(E @ pw, E @ pv))
        V = np.vstack((p, -p))
        edges = [(0, 1, order[k], order[(k + 1) % n]) for k in range(n)]
        return V, edges
    from scipy.spatial import ConvexHull
    H = ConvexHull(E)
    S = H.simplices
    C = _unit(np.cross(E[S[:, 1]] - E[S[:, 0]], E[S[:, 2]] - E[S[:, 0]]))
    C *= np.sign((C * E[S[:, 0]]).sum(1))[:, None]
    # merge equal circumcentres (co-circular generators)
    vid = np.full(len(C), -1)
    V = []
    for f in range(len(C)):
        for k, v in enumerate(V):
            if np.linalg.norm(C[f] - v) < 1e-6:
                vid[f] = k
                break
        else:
            vid[f] = len(V)
            V.append(C[f])
    V = np.array(V)
    ef = {}
    for f, s in enumerate(S):
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            ef.setdefault((min(a, b), max(a, b)), []).append(f)
    edges = []
    for (i, j), fs in ef.items():
        a, b = vid[fs[0]], vid[fs[1]]
        if a != b:
            edges.append((a, b, i, j))
    return V, edges


def ray_exit(v, E, r):
    """Distance from the node along unit v to the boundary of the union of the half-infinite strut cylinders and
    the node sphere (radius max r): all convex and contain the node, so the exit is the max of the exits."""
    s = r.max()
    for e, ri in zip(E, r):
        c = v @ e
        if c > 1e-12:
            s = max(s, ri / np.sqrt(max(1 - c * c, 1e-12)))
    return s


def line_cylinder(v, E, r, cells):
    """Paper's update: the point on the line through the node along unit v that lies on the cylinder (infinite,
    axis e_k) of the cells meeting there, s = r_k / sqrt(1 - (v.e_k)^2); unequal radii: the outermost."""
    return max(r[k] / np.sqrt(max(1 - (v @ E[k]) ** 2, 1e-12)) for k in cells) * v


def film_points(E, r):
    """Voronoi loop of every cell: the Voronoi vertices moved onto the intersection lines of the cylinders
    (line-cylinder, Fig. 7a), plus three points inserted on each non-monotonic Voronoi edge (Fig. 7b). Returns
    points P (k, 3), and per cell the point ids in counter-clockwise order about e_i."""
    V, edges = spherical_voronoi(E)
    vcells = [set() for _ in V]
    for a, b, i, j in edges:
        vcells[a] |= {i, j}
        vcells[b] |= {i, j}
    P = [line_cylinder(v, E, r, vcells[a]) for a, v in enumerate(V)]
    cell = [set() for _ in E]
    for a, b, i, j in edges:
        cell[i] |= {a, b}
        cell[j] |= {a, b}
        n2 = _unit(E[i] + E[j])
        th = _ang(V[a], V[b])
        if _ang(V[a], n2) < th and _ang(V[b], n2) < th:            # the intersection curve bends: insert 3 points
            for dvec in (_unit(V[a] + n2), n2, _unit(V[b] + n2)):
                k = len(P)
                P.append(line_cylinder(dvec, E, r, (i, j)))
                cell[i].add(k)
                cell[j].add(k)
    P = np.array(P)
    loops = []
    for i, e in enumerate(E):
        u, v = _frame(e)
        ids = np.array(sorted(cell[i]))
        az = np.arctan2(P[ids] @ v, P[ids] @ u)
        loops.append(ids[np.argsort(az)])
    return P, loops


# ------------------------------------------------------------------------------- film mesh, upsampling, fairing
def film_mesh(E, r, d, rows=3, row_t=None):
    """Control mesh: per cell a quad strip from its cut circle (row 0) through `rows` rows on the cylinder to its
    Voronoi loop (shared with the neighbour cells). Returns V, F, circle id per vertex (-1 none), circle parameter
    u, and row index per vertex (0 = boundary, rows + 1 = Voronoi loop)."""
    P, loops = film_points(E, r)
    V = [P]
    circ = [np.full(len(P), -1)]
    upar = [np.zeros(len(P))]
    row = [np.full(len(P), rows + 1)]
    F = []
    nv = len(P)
    for i, (e, ri, di) in enumerate(zip(E, r, d)):
        u, v = _frame(e)
        W = P[loops[i]]
        ph = np.arctan2(W @ v, W @ u)
        ring = di * e + ri * (np.outer(np.cos(ph), u) + np.outer(np.sin(ph), v))
        m = len(W)
        ids = [None] * (rows + 2)
        ids[rows + 1] = loops[i]
        tk = [0.0] + list(row_t if row_t is not None else [k / (rows + 1) for k in range(1, rows + 1)])
        for k in range(rows + 1):                                   # row 0 = circle, rows 1..rows interpolated
            t = tk[k]
            Q = (1 - t) * ring + t * W
            if 0 < k:                                               # onto the cylinder, perpendicular to the axis
                ax = Q @ e
                rad = Q - np.outer(ax, e)
                Q = np.outer(ax, e) + ri * _unit(rad)
            V.append(Q)
            circ.append(np.full(m, i if k == 0 else -1))
            upar.append(ph if k == 0 else np.zeros(m))
            row.append(np.full(m, k))
            ids[k] = np.arange(nv, nv + m)
            nv += m
        a_ = np.arange(m)
        b_ = (a_ + 1) % m
        for k in range(rows + 1):                                   # outward: ccw about e, row k further out
            o, nr = ids[k], ids[k + 1]
            F.append(np.column_stack((nr[a_], o[a_], nr[b_])))
            F.append(np.column_stack((nr[b_], o[a_], o[b_])))
    return np.vstack(V), np.vstack(F), np.concatenate(circ), np.concatenate(upar), np.concatenate(row)


def _adjacency(F, n):
    E = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(float).tocsr()


def fair(V, F, fixed, order=2, cotan=False):
    """Discrete Laplacian fairing: L^order x = 0 on the free vertices, fixed ones as constraints (uniform weights,
    or cotangent M^-1 C)."""
    if cotan:
        from nodeJoint import cotan_laplacian
        L = cotan_laplacian(V, F)
    else:
        A = _adjacency(F, len(V))
        L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    if order == 12:                                                # L x = 0, then L^2 x = 0 from that result
        return fair(fair(V, F, fixed, 1, cotan), F, fixed, 2, cotan)
    K = L if order == 1 else (L @ L)
    K = K.tocsr()
    fr, fx = np.where(~fixed)[0], np.where(fixed)[0]
    solve = spl.factorized(K[fr][:, fr].tocsc())
    out = V.copy()
    for c in range(3):
        out[fr, c] = solve(-K[fr][:, fx] @ V[fx, c])
    return out


# ------------------------------------------------------------------------------- [3.4] PN-Loop + circle sampling
def _vertex_normals(V, F):
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])          # area weighted
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return _unit(N)


def _edges(F):
    """Unique edges, per face its three edge ids (edge k of face f = (f[k], f[k+1])), and per edge its opposite
    vertices (-1 if boundary)."""
    H = np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    opp = np.concatenate((F[:, 2], F[:, 0], F[:, 1]))
    key = np.sort(H, 1)
    E, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nf = len(F)
    fe = inv.reshape(3, nf).T
    o = np.full((len(E), 2), -1)
    first = np.full(len(E), True)
    for h in range(len(H)):
        e = inv[h]
        if first[e]:
            o[e, 0] = opp[h]
            first[e] = False
        else:
            o[e, 1] = opp[h]
    return E, fe, o


def _pn_offset(Vk, Nk, t, n1):
    """PN displacement term ((n_k + n') . (v_k - t)) / ((n_k + n') . n')."""
    s = Nk + n1
    return (s * (Vk - t)).sum(-1) / np.maximum((s * n1).sum(-1), 1e-12)


def pn_loop_step(V, N, F, circ, upar, centers, axes, radii, pn=True):
    """One PN-Loop step (Eqs. 10-11) with circle sampling (Eq. 6) on the boundary circles. Boundary vertices stay;
    boundary edge points are sampled on their circle at the mid parameter, normal = cylinder normal."""
    E, fe, o = _edges(F)
    nV = len(V)
    bnd_e = o[:, 1] < 0
    bnd_v = circ >= 0
    # --- vertex points (interior vertices)
    A = _adjacency(F, nV)
    n = np.asarray(A.sum(1)).ravel()
    beta = (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2 * np.pi / n)) ** 2) / n
    a0 = 1 - n * beta
    t = a0[:, None] * V + beta[:, None] * (A @ V)
    n1 = _unit(a0[:, None] * N + beta[:, None] * (A @ N))
    if pn:
        h = a0 * _pn_offset(V, N, t, n1)
        Ai = A.tocoo()
        hij = _pn_offset(V[Ai.col], N[Ai.col], t[Ai.row], n1[Ai.row])
        h += beta * np.bincount(Ai.row, weights=hij, minlength=nV)
        Vv = t + h[:, None] * n1
    else:
        Vv = t
    Vv[bnd_v], n1[bnd_v] = V[bnd_v], N[bnd_v]
    # --- edge points
    i, j, p, q = E[:, 0], E[:, 1], o[:, 0], np.where(o[:, 1] >= 0, o[:, 1], o[:, 0])
    te = 3 / 8 * (V[i] + V[j]) + 1 / 8 * (V[p] + V[q])
    ne = _unit(3 / 8 * (N[i] + N[j]) + 1 / 8 * (N[p] + N[q]))
    if pn:
        he = 3 / 8 * (_pn_offset(V[i], N[i], te, ne) + _pn_offset(V[j], N[j], te, ne)) + \
             1 / 8 * (_pn_offset(V[p], N[p], te, ne) + _pn_offset(V[q], N[q], te, ne))
        Ve = te + he[:, None] * ne
    else:
        Ve = te
    ce = np.full(len(E), -1)
    ue = np.zeros(len(E))
    b = np.where(bnd_e)[0]
    ci = circ[i[b]]
    du = np.angle(np.exp(1j * (upar[j[b]] - upar[i[b]])))
    um = upar[i[b]] + 0.5 * du
    for k, (c0, ax, rr) in enumerate(zip(centers, axes, radii)):
        s = ci == k
        if s.any():
            u_, v_ = _frame(ax)
            rad = np.outer(np.cos(um[s]), u_) + np.outer(np.sin(um[s]), v_)
            Ve[b[s]] = c0 + rr * rad
            ne[b[s]] = rad
    ce[b], ue[b] = ci, um
    # --- 1-to-4 split
    m = nV + fe
    a_, b_, c_ = F[:, 0], F[:, 1], F[:, 2]
    F2 = np.vstack((np.column_stack((a_, m[:, 0], m[:, 2])), np.column_stack((b_, m[:, 1], m[:, 0])),
                    np.column_stack((c_, m[:, 2], m[:, 1])), m))
    return (np.vstack((Vv, Ve)), np.vstack((n1, ne)), F2, np.concatenate((circ, ce)), np.concatenate((upar, ue)))


def soap_film_node(node, E, r, lam=0.3, rows=3, levels=3, fair_order=12, pn=True, smooth=True, fixed_rows=1,
                   cotan=True, row_t=None):
    """SF node patch at `node` for struts along unit E with radii r.
    [3.3.2] 3 rows inserted linearly along the axial edges (row_t=None: evenly, as Fig. 8a: circle + 2 rings fixed
    = ~half the strip); fairing: the discrete Laplacian equation, then the second-order one (fair_order=12), with
    the cotangent (Desbrun et al. 1999, the paper's [38]) Laplacian (cotan=True). Uniform weights pull the crease
    row inward toward the node on this irregular mesh (undershoot into the struts, wavy curvature; notes/16 R2).
    Fixed: the boundary + its first ring (fixed_rows=1, "2 rings" counted with the boundary): the reading that
    best reproduces Table 1 (regular-20 within 2 %); fixed_rows=2 (boundary + 2 rings) is 2-3x closer to the
    Boolean node than Table 1. Returns dict(V, F, circ, upar, d, ctrl=(V, F)):
    an open mesh whose boundary loops are exactly the cut circles (centre node + d_i e_i, radius r_i)."""
    E = _unit(np.asarray(E, float))
    r = np.broadcast_to(np.asarray(r, float), (len(E),)).copy()
    d = cut_lengths(E, r, lam)
    if row_t is not None:
        rows = len(row_t)
    V, F, circ, upar, row = film_mesh(E, r, d, rows, row_t)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if (fn * V[F].mean(1)).sum() < 0:                               # outward normals
        F = F[:, ::-1].copy()
    if smooth:
        V = fair(V, F, row <= fixed_rows, order=fair_order, cotan=cotan)   # boundary + its 2 rings fixed [3.3.2]
    N = _vertex_normals(V, F)
    cyl = row <= 1                                                  # boundary + 1-ring: cylinder normals [3.4.2]
    owner = _row_owner(V, E, r, d, row)
    for i, e in enumerate(E):
        s = cyl & (owner == i)
        rad = V[s] - np.outer(V[s] @ e, e)
        N[s] = _unit(rad)
    ctrl = (V + node, F.copy())
    centers, axes = [di * e for e, di in zip(E, d)], list(E)
    for _ in range(levels):
        V, N, F, circ, upar = pn_loop_step(V, N, F, circ, upar, centers, axes, r, pn=pn)
    return dict(V=V + node, F=F, N=N, circ=circ, upar=upar, d=d, E=E, r=r, ctrl=ctrl)


def _row_owner(V, E, r, d, row):
    """Strut owning each non-Voronoi-loop vertex (the one whose cylinder it lies on, nearest to its axis cut)."""
    owner = np.full(len(V), -1)
    best = np.full(len(V), np.inf)
    for i, (e, ri) in enumerate(zip(E, r)):
        ax = V @ e
        dev = np.abs(np.linalg.norm(V - np.outer(ax, e), axis=1) - ri) + (ax < 0) * 1e3
        s = dev < best
        owner[s], best[s] = i, dev[s]
    owner[row > 3] = -1
    return owner


# --------------------------------------------------------------------------------------------- full node solid
def node_solid(sf, L, spacing=None):
    """Closed triangle mesh: SF patch + each retained strut as a tube from its cut circle to the flat far end at
    L_i (rings at the boundary's mean edge length, same boundary vertices, so watertight), with a fan cap."""
    import trimesh
    V, F, circ, upar = sf['V'], sf['F'], sf['circ'], sf['upar']
    Vs, Fs = [V], [F]
    nv = len(V)
    for i, (e, ri, di) in enumerate(zip(sf['E'], sf['r'], sf['d'])):
        ids = np.where(circ == i)[0]
        ids = ids[np.argsort(np.mod(upar[ids], 2 * np.pi))]
        m = len(ids)
        c0 = V[ids] - di * e                                        # node + radial part
        h = spacing or 2 * np.pi * ri / m
        zs = np.linspace(di, L[i], max(2, int(np.ceil((L[i] - di) / h)) + 1))[1:]
        a = np.arange(m)
        b = (a + 1) % m
        prev = ids
        for z in zs:
            cur = nv + a
            Vs.append(c0 + z * e)
            Fs.append(np.column_stack((prev[a], cur, prev[b])))
            Fs.append(np.column_stack((prev[b], cur, cur[b])))
            prev = cur
            nv += m
        Vs.append((c0.mean(0) + L[i] * e)[None])
        Fs.append(np.column_stack((prev[a], np.full(m, nv), prev[b])))
        nv += 1
    mesh = trimesh.Trimesh(np.vstack(Vs), np.vstack(Fs), process=True)
    trimesh.repair.fix_winding(mesh)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


# ------------------------------------------------------------------------------------------ Savio et al. 2018
def catmull_clark(V, faces):
    """One Catmull-Clark step on a closed polygon mesh (faces: list of vertex-index lists). Returns (V, quads)."""
    nV = len(V)
    fp = np.array([V[f].mean(0) for f in faces])
    emap, edges, ef = {}, [], []
    for fi, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            k = (min(a, b), max(a, b))
            if k not in emap:
                emap[k] = len(edges)
                edges.append(k)
                ef.append([])
            ef[emap[k]].append(fi)
    edges = np.array(edges)
    if any(len(x) != 2 for x in ef):
        raise ValueError('Catmull-Clark: mesh is not closed / manifold')
    ef = np.array(ef)
    ep = (V[edges[:, 0]] + V[edges[:, 1]] + fp[ef[:, 0]] + fp[ef[:, 1]]) / 4
    n = np.zeros(nV)
    Q = np.zeros_like(V)
    R = np.zeros_like(V)
    nf = np.zeros(nV)
    for fi, f in enumerate(faces):
        for v in f:
            Q[v] += fp[fi]
            nf[v] += 1
    mid = 0.5 * (V[edges[:, 0]] + V[edges[:, 1]])
    for c in (0, 1):
        np.add.at(R, edges[:, c], mid)
        np.add.at(n, edges[:, c], 1)
    Q /= nf[:, None]
    R /= n[:, None]
    Vv = (Q + 2 * R + (n - 3)[:, None] * V) / n[:, None]
    Vn = np.vstack((Vv, ep, fp))
    eo, fo = nV, nV + len(edges)
    quads = []
    for fi, f in enumerate(faces):
        k = len(f)
        for j in range(k):
            a, b, c = f[j - 1], f[j], f[(j + 1) % k]
            quads.append([b, eo + emap[(min(b, c), max(b, c))], fo + fi, eo + emap[(min(a, b), max(a, b))]])
    return Vn, quads


def _cc_shrink(levels):
    """Mean radius / half-side of the Catmull-Clark limit of a long square prism (Savio: scale the section so the
    subdivided beam has the nominal diameter, Sec. 3.3)."""
    n_r = 12
    V = np.array([[s1, s2, z] for z in range(n_r) for s1, s2 in ((1, 1), (-1, 1), (-1, -1), (1, -1))], float)
    F = [[4 * k + j, 4 * k + (j + 1) % 4, 4 * (k + 1) + (j + 1) % 4, 4 * (k + 1) + j] for k in range(n_r - 1) for j in range(4)]
    F += [[3, 2, 1, 0], [4 * (n_r - 1) + j for j in range(4)]]
    for _ in range(levels):
        V, F = catmull_clark(V, F)
    mid = np.abs(V[:, 2] - (n_r - 1) / 2) < 1.0
    return float(np.linalg.norm(V[mid, :2], axis=1).mean())


def savio_node(node, E, r, L, levels=3, tol=1e-9):
    """Savio node; tries the exact Table I placement (touching node squares, shared corners: the cubic node) and,
    where squares touch without sharing corners (non-cubic nodes), a 2 % clearance so the hull closes the gaps."""
    try:
        return _savio_node(node, E, r, L, levels, 0.0)
    except ValueError:
        return _savio_node(node, E, r, L, levels, 0.02)


def _savio_node(node, E, r, L, levels, margin):
    """Savio, Meneghello, Concheri (2018), Rapid Prototyping J. 24(2) 351-360, generalised to one node of any
    valence (their Sec. 4 extension is only sketched; the SF paper does the same): each beam is a square-section
    frustum, from the node square (half-side a_h from the MAXIMUM radius at the node, Table I) to the beam's own
    square at mid-span, then a prism to the far end. Node squares sit at the hull-face distance (= a_h for the cubic
    node, where adjacent squares share their corners as in Table I) and the node is closed by the convex hull of all
    node squares (planar faces where a beam is missing). Catmull-Clark x levels, half-sides divided by the measured
    CC shrink (their diameter compensation), far ends trimmed flat. Returns (closed trimesh, control (V, faces))."""
    import trimesh
    import manifold3d as m3
    from scipy.spatial import ConvexHull
    E = _unit(np.asarray(E, float))
    r = np.broadcast_to(np.asarray(r, float), (len(E),)).copy()
    L = np.broadcast_to(np.asarray(L, float), (len(E),)).copy()
    k = _cc_shrink(levels)
    a_h = r.max() / k
    a = r / k
    frames = []
    for i, e in enumerate(E):                                      # square orientation: edges facing the nearest
        cs = [E[j] @ e for j in range(len(E)) if j != i]           # neighbour (cube: edges parallel to the axes)
        jn = [j for j in range(len(E)) if j != i][int(np.argmax(cs))]
        o = E[jn] - (E[jn] @ e) * e
        u = _unit(o) if np.linalg.norm(o) > 1e-9 else _frame(e)[0]
        frames.append((u, np.cross(e, u)))
    corner = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    def square(i, ti, ah):
        u, v = frames[i]
        return ti * E[i] + ah * (np.outer(corner[:, 0], u) + np.outer(corner[:, 1], v))
    t = np.full(len(E), a_h)
    for _ in range(200):                                           # no other node square reaches past square i's
        tn = t.copy()                                              # plane (equality = shared corners, cube)
        for i in range(len(E)):
            for j in range(len(E)):
                if i != j:
                    tn[i] = max(tn[i], float((square(j, t[j], a_h) @ E[i]).max()) + margin * a_h)
        if np.allclose(tn, t, atol=1e-12):
            break
        t = tn
    rings = [(square(i, t[i], a_h), *frames[i]) for i in range(len(E))]
    # merge shared corners (cubic node: Table I vertices common to three beams)
    allp = np.vstack([rg for rg, _, _ in rings])
    uniq, inv = [], []
    for p_ in allp:
        for q, w in enumerate(uniq):
            if np.linalg.norm(p_ - w) < 1e-7 * a_h:
                inv.append(q)
                break
        else:
            inv.append(len(uniq))
            uniq.append(p_)
    V = list(uniq)
    node_ring = [list(np.array(inv[4 * i:4 * i + 4])) for i in range(len(E))]
    faces = []
    H = ConvexHull(np.array(uniq))
    ring_sets = [set(rg) for rg in node_ring]
    # hull facets, merged by plane: drop the ones that are a node square (beam opening)
    from collections import defaultdict
    groups = defaultdict(list)
    for sx, eq in zip(H.simplices, H.equations):
        groups[tuple(np.round(eq, 7))].append(sx)
    for key, tris in groups.items():
        vs = set(np.concatenate(tris))
        if any(vs <= rs and len(vs) == 4 for rs in ring_sets):
            continue
        for tri in tris:                                           # keep as triangles (CC handles n-gons)
            tri = list(tri)
            nrm = np.cross(uniq[tri[1]] - uniq[tri[0]], uniq[tri[2]] - uniq[tri[0]])
            if nrm @ (uniq[tri[0]]) < 0:
                tri = tri[::-1]
            faces.append([int(x) for x in tri])
    for i, (e, ri) in enumerate(zip(E, r)):                        # frustum to mid-span, prism to past the end
        rg, u, v = rings[i]
        prev = node_ring[i]
        zs = [0.5 * L[i], L[i] + 2 * a[i]] if 0.5 * L[i] > t[i] + 1e-6 else [L[i] + 2 * a[i]]
        for z in zs:
            cur = []
            for ang in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
                cur.append(len(V))
                V.append(z * e + a[i] * np.sqrt(2) * (np.cos(ang) * u + np.sin(ang) * v))
            for j in range(4):
                faces.append([prev[j], cur[j], cur[(j + 1) % 4], prev[(j + 1) % 4]])
            prev = cur
        faces.append(prev[::-1])                                   # end cap (trimmed away)
    V = np.array(V)
    # consistent outward orientation of the control mesh
    tm = trimesh.Trimesh(V, [f if len(f) == 3 else f[:3] for f in faces], process=False)
    ctrl = (V + node, [list(f) for f in faces])
    Vs, Fs = V, faces
    for _ in range(levels):
        Vs, Fs = catmull_clark(Vs, Fs)
    Fq = np.array(Fs)
    T = np.vstack((Fq[:, [0, 1, 2]], Fq[:, [0, 2, 3]]))
    mesh = trimesh.Trimesh(Vs, T, process=True)
    trimesh.repair.fix_winding(mesh)
    if mesh.volume < 0:
        mesh.invert()
    man = m3.Manifold(m3.Mesh(np.ascontiguousarray(mesh.vertices, np.float32), np.ascontiguousarray(mesh.faces, np.uint32)))
    for e, l in zip(E, L):
        man = man.trim_by_plane(tuple(map(float, -e)), float(-l))
    out = man.to_mesh()
    m = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3] + node, np.asarray(out.tri_verts), process=True)
    return m, ctrl, dict(a_h=a_h, t=t, shrink=k)
