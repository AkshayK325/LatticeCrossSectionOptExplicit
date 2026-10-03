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
    vols = []
    for a, b in conn:
        p, d = X[a], X[b] - X[a]
        vols.append((3, occ.addCylinder(*p, *d, r)))
    for p in X:
        vols.append((3, occ.addSphere(*p, r)))
    fused, _ = occ.fuse(vols[:1], vols[1:])
    lo, hi = X.min(0) - 2 * r, X.max(0) + 2 * r
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
