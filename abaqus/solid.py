"""Solid (continuum) lattice models for the level-B Abaqus benchmarks: strut cylinders + joint spheres
(manifold3d union), trimmed at the plate planes, tet-meshed with TetGen (meshpy) and written as C3D10M."""
import numpy as np


def lattice_solid(X, conn, r, z_lo, z_hi, segments=24):
    """Watertight triangle mesh (verts, tris) of the union of cylinders (radius r) along the struts and
    spheres (radius r) at the nodes, intersected with the slab z_lo <= z <= z_hi."""
    import manifold3d as m3
    parts = []
    for a, b in conn:
        p, q = X[a], X[b]
        d = q - p
        L = float(np.linalg.norm(d))
        cyl = m3.Manifold.cylinder(L, r, r, segments)          # along +z from the origin
        t = d / L
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, t)
        s, c = np.linalg.norm(v), float(np.dot(z, t))
        if s < 1e-12:
            Rm = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            k = v / s
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rm = np.eye(3) + s * K + (1 - c) * K @ K
        M = np.hstack((Rm, p.reshape(3, 1)))                     # 3x4 affine
        parts.append(cyl.transform(M.tolist()))
    for p in X:
        parts.append(m3.Manifold.sphere(r, segments).translate(tuple(map(float, p))))
    solid = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
    big = 1e3
    slab = m3.Manifold.cube((2 * big, 2 * big, z_hi - z_lo)).translate((-big, -big, z_lo))
    solid = solid ^ slab                                          # intersection
    mesh = solid.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(float), np.asarray(mesh.tri_verts).astype(int)


def lattice_solid_sdf(X, conn, r, z_lo, z_hi, h):
    """Same solid (union of capsules = strut cylinders + joint spheres, trimmed to z_lo..z_hi) from a signed
    distance field: uniform surface triangles of edge ~h (no boolean slivers, so no tiny tets)."""
    import manifold3d as m3
    P0, D = X[conn[:, 0]], X[conn[:, 1]] - X[conn[:, 0]]
    DD = (D * D).sum(1)

    def sdf(x, y, z):
        p = np.array((x, y, z))
        t = np.clip(((p - P0) * D).sum(1) / DD, 0.0, 1.0)
        d = np.sqrt((((P0 + t[:, None] * D) - p) ** 2).sum(1)).min()
        return min(r - d, z - z_lo, z_hi - z)                   # inside > 0; slab intersection
    lo, hi = X.min(0) - r - h, X.max(0) + r + h
    m = m3.Manifold.level_set(sdf, [lo[0], lo[1], z_lo - h, hi[0], hi[1], z_hi + h], h, 0.0, 0.02 * h)
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(float), np.asarray(mesh.tri_verts).astype(int)


def tet_mesh(verts, tris, h):
    """TetGen quality tets with max volume ~ regular tet of edge h. Returns (points, tets (n,4), 0-based)."""
    from meshpy.tet import MeshInfo, build, Options
    mi = MeshInfo()
    mi.set_points(verts.tolist())
    mi.set_facets(tris.tolist())
    vmax = h**3 / (6 * np.sqrt(2))
    m = build(mi, options=Options('pq1.4'), max_volume=vmax)
    P = np.array(m.points, float)
    T = np.array(m.elements, int)
    # positive orientation
    v = np.einsum('ij,ij->i', np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), P[T[:, 3]] - P[T[:, 0]])
    T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]
    return P, T


def quadratic(P, T):
    """C3D10 connectivity: corners 1-4 then mid-edge nodes (1-2, 2-3, 3-1, 1-4, 2-4, 3-4). Returns (P10, T10)."""
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    key = {}
    new = []
    T10 = np.zeros((len(T), 10), int)
    T10[:, :4] = T
    n0 = len(P)
    for e, tet in enumerate(T):
        for k, (i, j) in enumerate(edges):
            a, b = tet[i], tet[j]
            kk = (a, b) if a < b else (b, a)
            if kk not in key:
                key[kk] = n0 + len(new)
                new.append(0.5 * (P[a] + P[b]))
            T10[e, 4 + k] = key[kk]
    return np.vstack((P, np.array(new))), T10


def lumped_masses_c3d10m(P10, T10, rho):
    """C3D10M lumped mass: (approximately) uniform over the 10 nodes of each element (corner = mid-edge)."""
    m = np.zeros(len(P10))
    for e in T10:
        a, b, c, d = P10[e[:4]]
        V = abs(np.dot(np.cross(b - a, c - a), d - a)) / 6
        m[e] += rho * V / 10
    return m


def gmsh_lattice(X, conn, r, z_lo, z_hi, h, optimize=True, verbose=False):
    """Solid lattice with gmsh/OpenCASCADE: exact cylinders + joint spheres fused, intersected with the slab
    z_lo..z_hi, linear tets of size ~h with Netgen optimisation (removes slivers). Returns (P, T) 0-based."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
    occ = gmsh.model.occ
    rs = np.broadcast_to(np.asarray(r, float), (len(conn),))      # per-strut radii (scalar allowed)
    rn = np.zeros(len(X))
    for (a, b), rr in zip(conn, rs):                               # joint sphere: largest radius at the node
        rn[a], rn[b] = max(rn[a], rr), max(rn[b], rr)
    vols = []
    for (a, b), rr in zip(conn, rs):
        p, d = X[a], X[b] - X[a]
        vols.append((3, occ.addCylinder(*p, *d, float(rr))))
    for p, rr in zip(X, rn):
        vols.append((3, occ.addSphere(*p, float(rr))))
    fused, _ = occ.fuse(vols[:1], vols[1:])
    lo, hi = X.min(0) - 2 * rs.max(), X.max(0) + 2 * rs.max()
    box = occ.addBox(lo[0], lo[1], z_lo, hi[0] - lo[0], hi[1] - lo[1], z_hi - z_lo)
    out, _ = occ.intersect(fused, [(3, box)])
    occ.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.5 * h)
    gmsh.option.setNumber('Mesh.MeshSizeMax', h)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 12)        # ~12 elements around a circle
    gmsh.option.setNumber('Mesh.Algorithm3D', 10)                  # HXT
    gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.option.setNumber('Mesh.OptimizeNetgen', 1 if optimize else 0)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    P = coords.reshape(-1, 3)
    idx = {int(t): i for i, t in enumerate(tags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(3)
    T = np.array([idx[int(n)] for n in enodes[list(etypes).index(4)]]).reshape(-1, 4)
    gmsh.finalize()
    used = np.unique(T)                                            # drop surface-only nodes (none expected)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, T = P[used], remap[T]
    v = np.einsum('ij,ij->i', np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), P[T[:, 3]] - P[T[:, 0]])
    T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]
    return P, T


def tet_quality(P, T):
    """(volumes, shortest altitudes) of linear tets."""
    a, b, c, d = (P[T[:, k]] for k in range(4))
    V = np.abs(np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)) / 6
    A = np.stack([np.linalg.norm(np.cross(b - a, c - a), axis=1), np.linalg.norm(np.cross(b - a, d - a), axis=1),
                  np.linalg.norm(np.cross(c - a, d - a), axis=1), np.linalg.norm(np.cross(c - b, d - b), axis=1)], 1)
    return V, 3 * V / (A.max(1) / 2)


def gmsh_from_stl(stl, h, angle=40.0, curvature=12, optimize=True, verbose=False, algo3d=10, extra_opt=(), algo2d=6,
                  size_min=0.5, smooth2d=0, opt2d=()):
    """Solid tet mesh of a watertight STL (e.g. an FHS lattice, nodeJoint.lattice_fhs): gmsh splits the discrete
    surface at sharp edges (> angle deg: the plate rims; the FHS fillets are smooth, < 20 deg), reparametrises the
    patches, remeshes the surface at ~h (finer where curved: `curvature` elements per 2 pi) and fills it with linear
    tets (HXT + Netgen optimisation), so the solid mesh is independent of the STL's fine triangulation.
    Returns (P, T) 0-based."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
    gmsh.merge(str(stl))
    gmsh.model.mesh.classifySurfaces(np.radians(angle), True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    loop = gmsh.model.geo.addSurfaceLoop(surfs)
    gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMin', size_min * h)
    gmsh.option.setNumber('Mesh.MeshSizeMax', h)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', curvature)
    gmsh.option.setNumber('Mesh.Algorithm', algo2d)
    gmsh.option.setNumber('Mesh.Algorithm3D', algo3d)
    gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.option.setNumber('Mesh.OptimizeNetgen', 1 if optimize else 0)
    gmsh.option.setNumber('Mesh.Smoothing', smooth2d)
    gmsh.model.mesh.generate(2)                                    # surface first: smooth / optimise it, then fill
    for method in opt2d:                                           # e.g. 'Laplace2D'
        gmsh.model.mesh.optimize(method, niter=5)
    try:
        gmsh.model.mesh.generate(3)
    except Exception:
        if algo3d == 1:
            raise
        gmsh.finalize()                                            # HXT can fail on a valid surface: classic Delaunay
        return gmsh_from_stl(stl, h, angle, curvature, optimize, verbose, 1, extra_opt, algo2d, size_min, smooth2d, opt2d)
    for method in extra_opt:                                       # e.g. 'Netgen', 'Relocate3D'
        gmsh.model.mesh.optimize(method, niter=3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    P = coords.reshape(-1, 3)
    idx = {int(t): i for i, t in enumerate(tags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(3)
    T = np.array([idx[int(n)] for n in enodes[list(etypes).index(4)]]).reshape(-1, 4)
    gmsh.finalize()
    used = np.unique(T)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    P, T = P[used], remap[T]
    v = np.einsum('ij,ij->i', np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), P[T[:, 3]] - P[T[:, 0]])
    T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]
    return P, T


def boundary_defects(T):
    """Vertices where the tet mesh's boundary is not a closed 2-manifold: endpoints of boundary edges not shared by
    exactly two boundary faces, pinched vertices (the boundary faces around them do not form a single fan), and
    vertices of faces shared by > 2 tets. Vectorised: one sparse connected-components pass over all vertex links.
    Returns a set of vertex indices (empty = clean)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), 1)
    u, cnt = np.unique(F, axis=0, return_counts=True)
    bad = set(u[cnt > 2].ravel().tolist())
    B = u[cnt == 1]
    E = np.sort(np.vstack([B[:, [0, 1]], B[:, [1, 2]], B[:, [0, 2]]]), 1)
    ue, ec = np.unique(E, axis=0, return_counts=True)
    bad |= set(ue[ec != 2].ravel().tolist())
    # link graph of every boundary vertex v: nodes (v, w) for its boundary neighbours w, an edge per boundary face
    a, b, c = B[:, 0], B[:, 1], B[:, 2]
    V0 = np.concatenate((a, b, c))
    W1 = np.concatenate((b, c, a))
    W2 = np.concatenate((c, a, b))
    pairs = np.concatenate((np.column_stack((V0, W1)), np.column_stack((V0, W2))))
    up, inv = np.unique(pairs, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(V0)
    g = coo_matrix((np.ones(n), (inv[:n], inv[n:])), shape=(len(up), len(up)))
    _, comp = connected_components(g, directed=False)
    vc = np.unique(np.column_stack((up[:, 0], comp)), axis=0)       # (vertex, component) pairs
    vv, nc = np.unique(vc[:, 0], return_counts=True)
    bad |= set(vv[nc > 1].tolist())
    return bad


def boundary_ok(T):
    return not boundary_defects(T)


def remove_boundary_slivers(P, T, min_alt, max_iter=50, one_face_alt=None):
    """Delete flat boundary tets: shortest altitude < min_alt with >= 2 boundary faces (they bridge two surface
    triangles across a fold of a curved surface: removing one shaves a negligible wedge), and, if one_face_alt is
    given, tets with ONE boundary face below that altitude (removing one leaves a dent no deeper than its altitude).
    Batch removal, then every removed tet touching a boundary defect (non-manifold edge, pinched vertex) is put
    back, until the boundary is a closed 2-manifold. Returns (P, T, removed)."""
    removed_total = 0
    for _ in range(max_iter):
        F = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), 1)
        _, inv, cnt = np.unique(F, axis=0, return_inverse=True, return_counts=True)
        nbnd = (cnt[inv.ravel()] == 1).reshape(4, -1).sum(0)
        _, alt = tet_quality(P, T)
        kill = (alt < min_alt) & (nbnd >= 2)
        if one_face_alt is not None:
            kill |= (alt < one_face_alt) & (nbnd == 1)
        if not kill.any():
            break
        keep = ~kill
        for _ in range(20):                                 # put back removed tets at boundary defects
            bad = boundary_defects(T[keep])
            if not bad:
                break
            back = (~keep) & np.isin(T, list(bad)).any(1)
            if not back.any():
                break
            keep |= back
        if boundary_defects(T[keep]) or keep.all():
            break
        removed_total += int((~keep).sum())
        T = T[keep]
    removed = removed_total
    used = np.unique(T)
    remap = -np.ones(len(P), int)
    remap[used] = np.arange(len(used))
    return P[used], remap[T], removed


def relax_small_tets(P, T, min_alt, passes=10):
    """Lift the interior vertices of tets thinner than min_alt: each such vertex is moved toward the centroid of its
    edge neighbours (full step, then half, quarter), accepted only if every tet around it stays positive and the
    worst altitude around it improves. Boundary vertices never move, so the surface is untouched. Returns P."""
    P = P.copy()
    F = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), 1)
    u, cnt = np.unique(F, axis=0, return_counts=True)
    on_bnd = np.zeros(len(P), bool)
    on_bnd[u[cnt == 1].ravel()] = True
    vt = [[] for _ in range(len(P))]
    for e, t in enumerate(T):
        for v in t:
            vt[v].append(e)

    def worst(v, X):
        tt = T[vt[v]]
        a, b, c, d = (X[tt[:, k]] for k in range(4))
        vol = np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a) / 6
        if (vol <= 0).any():
            return -1.0
        return tet_quality(X, tt)[1].min()
    for _ in range(passes):
        _, alt = tet_quality(P, T)
        bad = np.where(alt < min_alt)[0]
        if len(bad) == 0:
            break
        moved = 0
        for e in bad:
            for v in T[e]:
                if on_bnd[v]:
                    continue
                nbr = np.unique(T[vt[v]].ravel())
                nbr = nbr[nbr != v]
                w0 = worst(v, P)
                target = P[nbr].mean(0)
                for step in (1.0, 0.5, 0.25):
                    old = P[v].copy()
                    P[v] = old + step * (target - old)
                    if worst(v, P) > w0:
                        moved += 1
                        break
                    P[v] = old
        if moved == 0:
            break
    return P
