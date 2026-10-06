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


def hull_subdivision_node(node, dirs, radii, L, R_s, levels=3, n_min=16, lead=0.0, core=False):
    """Method C1: organic node by subdivision of a hull control mesh (notes/12).
    Control mesh = convex hull of one ring per beam (at ring_offsets: outside the trim sphere and each ring a face of
    the hull), with the ring caps removed,
    plus a control tube per beam (rings one edge length apart, past the far end, capped). Loop subdivision gives a
    smooth (C2 away from extraordinary vertices) surface whose tubes converge to the beams (control rings enlarged
    by the measured Loop shrink); the far ends are then trimmed flat. Returns the watertight manifold."""
    from scipy.spatial import ConvexHull
    r_min = radii.min()
    toff = ring_offsets(dirs, radii * np.array([1 / loop_shrink(ring_count(r, r_min, n_min), levels=levels)
                                                 for r in radii]), R_s, core=core)
    V, F, ring0 = [], [], []
    nv = 0
    for e, r, l in zip(dirs, radii, L):
        n = ring_count(r, r_min, n_min)
        rc = r / loop_shrink(n, levels=levels)
        u, v = _frame(e)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circ = rc * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
        dz = 2 * r * np.sin(np.pi / n)
        ti = toff[len(ring0)]
        # control rings: the first two gaps `lead` * r long (a longer blend from the hull into the beam), then one
        # edge length apart
        z0 = [ti] + ([ti + lead * r, ti + 2 * lead * r] if lead > 0 else [])
        zs = np.concatenate((z0, np.arange(z0[-1] + dz, l + 3 * dz, dz)))
        start = nv
        for z in zs:
            V.append(node + z * e + circ)
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
        for e, r in zip(dirs, radii):
            n = ring_count(r, r_min, n_min)
            u, v = _frame(e)
            t = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
            rc = r / loop_shrink(n, levels=levels)                 # same Loop-shrink compensation as the tubes
            V.append(node + rc * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)))
            cores.append(np.arange(nv, nv + n))
            nv += n
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


def core_clearance(P, node, dirs, radii, reach):
    """Smooth distance from P to the union of the beam cores (each beam's capsule from the node point to reach_i):
    cubic smooth minimum (width 0.5 r_min) of the capsule distances, so it has no jumps and no creases where two
    cores meet."""
    k = 0.5 * radii.min()
    out = None
    for e, r, l in zip(dirs, radii, reach):
        w = np.clip((P - node) @ e, 0.0, l)
        d = np.linalg.norm((P - node) - np.outer(w, e), axis=1) - r
        if out is None:
            out = d
        else:
            hh = np.maximum(k - np.abs(out - d), 0.0) / k
            out = np.minimum(out, d) - hh ** 3 * k / 6.0
    return out


def flank_weights(V, node, dirs, radii, reach):
    """Anchor weight per vertex for the anchored fairing, in [0, 1]: ~1 on a beam's own flank (close to its core,
    i.e. the beam cylinder continued to the node point, and away from every other core) where the thin plate would
    neck the beam; ~0 in the crevices (comparably close to two cores), which stay free to fair."""
    D = []
    for e, r, l in zip(dirs, radii, reach):
        w = np.clip((V - node) @ e, 0.0, l)
        D.append(np.linalg.norm((V - node) - np.outer(w, e), axis=1) - r)
    D = np.sort(np.array(D), axis=0)
    a = 0.3 * radii.min()
    b = 0.5 * radii.min()
    return np.exp(-(np.maximum(D[0], 0.0) / a) ** 2) * (1.0 - np.exp(-((D[1] - D[0]) / b) ** 2))


def fair_node_anchored(mesh, node, dirs, radii, t_ring, free_len=1.0, anchor=10.0):
    """FHS fairing (Faired Hull-Subdivision node, anchored): least-squares fairing, min x^T K x + sum_v w_v |x_v - x0_v|^2, with the thin-plate
    energy K = C^T M^-1 C (cotangent C, lumped area M), the beams fixed beyond t_ring_i + free_len * r_i, and soft
    anchors w = anchor * area / r_min^4 * flank_weights on the beams' flanks, so the crevices fair freely but no beam
    necks. One linear solve (no clamping, no folds). Returns (mesh, fixed mask)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces)
    fixed = np.zeros(len(V), bool)
    for e, r, t in zip(dirs, radii, t_ring):
        w = (V - node) @ e
        rad = np.linalg.norm((V - node) - np.outer(w, e), axis=1)
        fixed |= (w > t + free_len * r) & (np.abs(rad - r) < 0.05 * r)
    Lc = cotan_laplacian(V, F)                                        # M^-1 C
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    M = sp.diags(np.bincount(F.ravel(), weights=np.repeat(area / 3, 3), minlength=len(V)))
    K = (Lc.T @ M @ Lc).tocsr()                                       # = C^T M^-1 C
    # data term anchor * integral |x - x0|^2 / r_min^4 dA, consistent with the thin-plate energy integral |Lap x|^2 dA
    wv = anchor * M.diagonal() / radii.min() ** 4 * flank_weights(V, node, dirs, radii,
                                                                   t_ring + free_len * radii + 2 * radii)
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


def merge_straight_chains(X, conn, radii, cos_tol=1e-6, r_tol=1e-9):
    """Merge struts that continue straight through a degree-2 node with the same radius (element subdivisions, not
    joints): such a node would otherwise get a node patch and its dense blend rings for nothing. Returns
    (conn, radii) of the merged struts (node indices unchanged)."""
    conn = [tuple(c) for c in np.asarray(conn, int)]
    radii = list(np.broadcast_to(np.asarray(radii, float), (len(conn),)))
    while True:
        deg = {}
        for k, (a, b) in enumerate(conn):
            deg.setdefault(a, []).append(k)
            deg.setdefault(b, []).append(k)
        for n, ks in deg.items():
            if len(ks) != 2 or abs(radii[ks[0]] - radii[ks[1]]) > r_tol:
                continue
            ends = [conn[k][1] if conn[k][0] == n else conn[k][0] for k in ks]
            u, v = X[ends[0]] - X[n], X[ends[1]] - X[n]
            if u @ v / (np.linalg.norm(u) * np.linalg.norm(v)) < -1 + cos_tol:
                keep, drop = sorted(ks)
                conn[keep] = (ends[0], ends[1])
                del conn[drop], radii[drop]
                break
        else:
            return np.array(conn), np.array(radii)


def lattice_fhs(X, conn, radii, z_range=None, levels=2, lead=1.0, free_len=1.0, anchor=10.0, n_min=16,
                limit=True, mid=3.0):
    """FHS (Faired Hull-Subdivision) node joints for a whole lattice (algorithm: notes/13_FHS_NodeAlgorithm.md). Every node of degree >= 2 gets a node
    patch (convex hull of its node rings + core rings, ring caps removed); every strut a control tube between the
    node rings at its two ends; the whole control mesh is Loop-subdivided, the node regions are faired in one
    anchored least-squares solve, and the result is trimmed to the plate planes z_range.
    Mesh size (notes/14): 2 Loop levels with the vertices moved onto the exact limit surface (limit), and the
    strut-middle control rings mid edge lengths apart: ~5x fewer faces than 3 plain levels, 99 % of the surface
    within ~10 um of it. Returns (watertight manifold, info dict)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from scipy.spatial import ConvexHull, cKDTree
    X = np.asarray(X, float)
    conn, radii = merge_straight_chains(X, conn, radii)                 # element splits are not joints
    r_min = radii.min()
    nring = np.array([ring_count(r, r_min, n_min) for r in radii])
    shrink = {n: loop_shrink(n, spacing_ratio=mid, levels=levels, limit=limit) for n in set(nring.tolist())}
    rc = np.array([r / shrink[n] for r, n in zip(radii, nring)])
    length = np.linalg.norm(X[conn[:, 1]] - X[conn[:, 0]], axis=1)
    E = (X[conn[:, 1]] - X[conn[:, 0]]) / length[:, None]
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
        rr = np.array([radii[i] for i, _ in ends])
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
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circ = rc[i] * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
        zs = _strut_rings(toff[i, 0], length[i] - toff[i, 1], r, 2 * r * np.sin(np.pi / n), lead, mid)
        start = nv
        for z in zs:
            V.append(X[a] + z * e + circ)
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
            u, v = _frame(e)
            n = nring[i]
            t = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
            V.append(X[nd] + rc[i] * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)))
            idx.append(np.arange(nv, nv + n))
            ring_of.append(np.full(n, -1))
            nv += n
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
        rad = np.linalg.norm((V2 - X[a]) - np.outer(w, E[i]), axis=1)
        fixed |= ((w > toff[i, 0] + free_len * radii[i]) & (w < length[i] - toff[i, 1] - free_len * radii[i])
                  & (np.abs(rad - radii[i]) < 0.05 * radii[i]))
    for nd, ends in enumerate(inc):                                # free ends stay as subdivided
        if len(ends) == 1:
            fixed |= np.linalg.norm(V2 - X[nd], axis=1) < 3 * radii[ends[0][0]]
    flank = np.zeros(len(V2))
    owner = cKDTree(X).query(V2)[1]
    for nd, ends in enumerate(inc):
        sel = np.where(owner == nd)[0]
        if len(ends) < 2 or len(sel) == 0:
            continue
        dirs = np.array([E[i] if k == 0 else -E[i] for i, k in ends])
        rr = np.array([radii[i] for i, _ in ends])
        reach = np.array([toff[i, k] for i, k in ends]) + free_len * rr + 2 * rr
        flank[sel] = flank_weights(V2[sel], X[nd], dirs, rr, reach)
    Lc = cotan_laplacian(V2, F2)
    area = 0.5 * np.linalg.norm(np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]]), axis=1)
    M = sp.diags(np.bincount(F2.ravel(), weights=np.repeat(area / 3, 3), minlength=len(V2)))
    K = (Lc.T @ M @ Lc).tocsr()
    wv = anchor * M.diagonal() / r_min ** 4 * flank
    A = (K + sp.diags(wv)).tocsr()
    fr, fx = np.where(~fixed)[0], np.where(fixed)[0]
    solve = spl.factorized(A[fr][:, fr].tocsc())
    Vn = V2.copy()
    for k in range(3):
        Vn[fr, k] = solve(wv[fr] * V2[fr, k] - A[fr][:, fx] @ V2[fx, k])
    man = m3.Manifold(m3.Mesh(np.ascontiguousarray(Vn, np.float32), np.ascontiguousarray(F2, np.uint32)))
    if man.status() != m3.Error.NoError:
        raise RuntimeError(f'faired lattice is not a manifold: {man.status()}')
    if z_range is not None:                                        # 7. plate planes
        man = man.trim_by_plane((0.0, 0.0, 1.0), float(z_range[0])).trim_by_plane((0.0, 0.0, -1.0), -float(z_range[1]))
    return man, dict(control_faces=len(ctrl.faces), free_vertices=len(fr), ring_offsets=toff)


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
