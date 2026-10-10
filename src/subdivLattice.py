"""Soap-film node (SF) ported from the authors' code, Qiang-Zou/Subdiv-Lattice (C++, release 2025.09; read, not run):
G. Luo, Q. Zou, "Soap Film-inspired Subdivisional Lattice Structure Construction", CAD 2025. Source file in brackets.

    cut circles [nodeManager.cpp nodepairCutting] -> spherical Voronoi + 3 points per Voronoi edge, projected on the
    strut cylinders [TopoMesh.cpp] -> 5-row strip per strut -> 2 x clamped-cotangent bi-Laplacian fairing, boundary
    + 2 rings locked [OptiMesh.cpp] -> 1 PN-Loop step, cut circles as creases [SubDivMesh.cpp, main lapIter = 2,
    subdivIter = 1]

Kept as in the C++ (they change the shape slightly): the Loop vertex update keeps each vertex's own normal (nlist
collects n_v, not n_j), and the Voronoi-area mass entry squares the cotangent ((e * cot).squaredNorm()).
scipy's SphericalVoronoi replaces svFortune (no rotation retries needed). Returns the soapFilm.soap_film_node dict.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import SphericalVoronoi

from soapFilm import _frame, _unit


def _ang(a, b):
    return np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))


def cut_lengths(E, r):
    """[nodepairCutting] d_i = 1.3 max_j sqrt(m_ij^2 + r1^2), m_ij = (r2 + r1 cos th) / sin th, r1 <= r2 (the C++
    swap persists over j). Circle i at d_i e_i; node sphere radius = min_i d_i."""
    d = np.zeros(len(E))
    for i in range(len(E)):
        r1 = r[i]
        for j in range(len(E)):
            if i != j:
                r2 = r[j]
                if r1 > r2:
                    r1, r2 = r2, r1
                th = _ang(E[i], E[j])
                m = (r2 + r1 * np.cos(th)) / np.sin(th)
                d[i] = max(d[i], np.sqrt(m * m + r1 * r1))
    return 1.3 * d


def _voronoi_cells(E, R):
    """[FortuneVoronoi + removeDuplicateVertices + reorderPointsCCW] points (on the sphere R), owner cells per point,
    per cell its corner loop."""
    sv = SphericalVoronoi(E)
    P, cells = [], []
    for reg in sv.regions:
        loop = []
        for a in reg:
            p = sv.vertices[a] * R
            k = next((k for k, q in enumerate(P) if np.abs(q - p).max() < 1e-4), None)
            if k is None:
                k = len(P)
                P.append(p)
            if not loop or loop[-1] != k:
                loop.append(k)
        if loop[0] == loop[-1]:
            loop.pop()
        cells.append(loop)
    P = np.array(P)
    belong = [[] for _ in P]
    for i, loop in enumerate(cells):
        for k in loop:
            belong[k].append(i)
    for i, loop in enumerate(cells):                                # sort by angle about the corner centroid, reversed
        z = _unit(P[loop].mean(0))
        x = P[loop[0]] - (P[loop[0]] @ z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ang = np.arctan2(P[loop] @ y, P[loop] @ x)
        cells[i] = [loop[k] for k in np.argsort(ang, kind='stable')][::-1]
    return list(P), belong, cells


def _valence3_cells(E, R):
    """[buildInitialMesh, 3 struts] two poles on +-(e0 + e1 + e2), per pair the bisector point and two half points."""
    v1, v2, v3 = E
    v4 = _unit(v1 + v2 + v3)
    p1, p2 = v4 * R, -v4 * R
    P, belong = [p1, p2], [[0, 1, 2], [0, 1, 2]]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pm = _unit(E[a] + E[b]) * R
        P += [pm, 0.5 * p1 + 0.5 * pm, 0.5 * p2 + 0.5 * pm]
        belong += [[a, b]] * 3
    # p1-0, p2-1, p01-2, p101-3, p201-4, p02-5, p102-6, p202-7, p12-8, p112-9, p212-10
    cells = [[0, 6, 5, 7, 1, 4, 2, 3], [0, 3, 2, 4, 1, 10, 8, 9], [0, 9, 8, 10, 1, 7, 5, 6]]
    return P, belong, cells


def _add_edge_points(P, belong, cells, E, R):
    """[addVtxInEdge_all] 3 points on each Voronoi edge spanning > 0.5 rad, at 0.3 / 0.7 between the ends and a
    middle point (the axis bisector if it lies on the edge, else the chord middle). Returns the flags -2 (middle) /
    -3 (others) for edges between struts < 0.5 rad apart."""
    flag = {}
    done = {}
    for f, loop in enumerate(cells):                                # the first cell of an edge creates its points
        n = len(loop)
        for k in range(n):
            a, b = loop[k], loop[(k + 1) % n]
            key = (min(a, b), max(a, b))
            if key in done:
                continue
            done[key] = None
            g = [c for c in belong[a] if c != f and c in belong[b]]
            p1, p2 = P[a], P[b]
            th = _ang(p1, p2)
            if not g or not (_ang(E[f], E[g[0]]) < 3 and th > 0.5):
                continue
            close = _ang(E[f], E[g[0]]) < 0.5
            bis = _unit(E[f] + E[g[0]])
            mid_dir = _unit(0.5 * (p1 + p2))
            m = mid_dir * R
            if _ang(bis, p1) < th and _ang(bis, p2) < th:
                m = bis * R
                if _ang(0.3 * p1 + 0.7 * m, bis) < 0.2 or _ang(0.3 * p2 + 0.7 * m, bis) < 0.2:
                    m = mid_dir * R
            ids = []
            for q, fl in ((0.3 * p1 + 0.7 * m, -3), (m, -2), (0.3 * p2 + 0.7 * m, -3)):
                if close:
                    flag[len(P)] = fl
                ids.append(len(P))
                P.append(_unit(q) * R)
                belong.append([f])
            done[key] = (a, ids)
    # the second cell of an edge walks it backwards: rebuild loops with the stored order
    cells2 = []
    for f, loop in enumerate(cells):
        out = []
        n = len(loop)
        for k in range(n):
            a, b = loop[k], loop[(k + 1) % n]
            out.append(a)
            v = done.get((min(a, b), max(a, b)))
            if v is not None:
                start, ids = v
                out += ids if start == a else ids[::-1]
        cells2.append(out)
    return P, belong, cells2, flag


def _closest_on_segment(p, a, b):
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
    return a + t * (b - a)


def initial_mesh(E, r, d):
    """[buildInitialMesh, proVronoiVtxToNode, meshFromVoronoiDiagram_irregularHandle] node at the origin."""
    R = d.min()
    if len(E) == 3:
        P, belong, cells = _valence3_cells(E, R)
        flag = {}
    else:
        P, belong, cells = _voronoi_cells(E, R)
        P, belong, cells, flag = _add_edge_points(P, belong, cells, E, R)
    P = np.array(P)
    for k in range(len(P)):                                         # along the ray onto the owner cell's cylinder
        u = _unit(P[k])
        e = E[belong[k][0]]
        P[k] = r[belong[k][0]] / np.sqrt(max(1 - (u @ e) ** 2, 1e-12)) * u
    V = list(P)
    nP = len(P)
    # vertex properties: belong strut (circle only), hasNormal, normal, isHandled, isBoundary
    strut = [-1] * nP
    for k, fl in flag.items():
        strut[k] = fl
    has_n, normal, handled, bnd = [False] * nP, [np.zeros(3)] * nP, [False] * nP, [False] * nP
    F = []
    for i, loop in enumerate(cells):
        e, c, ri = E[i], d[i] * E[i], r[i]
        m = len(loop)
        circle = []
        for j in loop:                                              # [cirDivideProjection]
            p = P[j]
            q = p + e * abs((p - c) @ e)
            nrm = _unit(q - c)
            circle.append(len(V))
            V.append(c + ri * nrm)
            strut.append(i); has_n.append(True); normal.append(nrm); handled.append(True); bnd.append(True)
        layers = [list(loop)]
        for k in range(3):
            alpha = 0.6 + 0.4 / 4 * (k + 1)
            lay = []
            for jj, j in enumerate(loop):
                q = alpha * V[circle[jj]] + (1 - alpha) * P[j]
                foot = _closest_on_segment(q, np.zeros(3), c)
                vec = _unit(q - foot)
                lay.append(len(V))
                V.append(foot + ri * vec)
                if k == 0:
                    strut.append(-2); has_n.append(False); normal.append(np.zeros(3)); handled.append(False)
                else:
                    strut.append(i); has_n.append(k == 2); normal.append(vec)
                    handled.append(not (k == 1 and strut[j] in (-2, -3)))
                bnd.append(False)
            layers.append(lay)
        layers.append(circle)
        for j in range(m):
            jn = (j + 1) % m
            for mk in range(4):
                p1, p2, c1, c2 = layers[mk][j], layers[mk][jn], layers[mk + 1][j], layers[mk + 1][jn]
                F += [(p1, c1, p2), (p2, c1, c2)]
    props = dict(strut=np.array(strut), has_n=np.array(has_n), normal=np.array(normal),
                 handled=np.array(handled), bnd=np.array(bnd))
    return np.array(V), np.array(F), props


# ---------------------------------------------------------------------------------------------- [OptiMesh.cpp]
def _cot_laplacian_clamped(V, F):
    """L_ij = sum of opposite-angle cotangents (no 1/2), L_ii = -sum_j L_ij; negative off-diagonals clamped to 0."""
    l2 = np.stack([((V[F[:, 1]] - V[F[:, 2]]) ** 2).sum(1), ((V[F[:, 0]] - V[F[:, 2]]) ** 2).sum(1),
                   ((V[F[:, 0]] - V[F[:, 1]]) ** 2).sum(1)], 1)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    ok = A > np.finfo(float).tiny
    cot = np.zeros_like(l2)
    cot[ok, 0] = 0.25 * (l2[ok, 1] + l2[ok, 2] - l2[ok, 0]) / A[ok]
    cot[ok, 1] = 0.25 * (l2[ok, 2] + l2[ok, 0] - l2[ok, 1]) / A[ok]
    cot[ok, 2] = 0.25 * (l2[ok, 0] + l2[ok, 1] - l2[ok, 2]) / A[ok]
    I = np.concatenate([F[:, 1], F[:, 0], F[:, 0]])
    J = np.concatenate([F[:, 2], F[:, 2], F[:, 1]])
    W = np.concatenate([cot[:, 0], cot[:, 1], cot[:, 2]])
    n = len(V)
    Wd = sp.coo_matrix((np.r_[W, W], (np.r_[I, J], np.r_[J, I])), shape=(n, n)).tocsr()
    Wd.data[Wd.data < 0] = 0.0                                      # clamp, keeping the row sums zero
    return Wd - sp.diags(np.asarray(Wd.sum(1)).ravel())


def _mass(V, F):
    """Mixed Voronoi areas, as the C++ (the cotangent enters squared)."""
    p = [V[F[:, k]] for k in range(3)]
    e = [p[(k + 1) % 3] - p[k] for k in range(3)]
    tri = np.linalg.norm(np.cross(e[0], e[1]), axis=1)               # twice the area
    dd = [-(e[k] * e[(k + 2) % 3]).sum(1) for k in range(3)]
    ok = tri > np.finfo(float).tiny
    cot = [np.where(ok, dd[k] / np.where(ok, tri, 1), 0) for k in range(3)]
    M = np.zeros(len(V))
    for k in range(3):
        obt_here = dd[k] < 0
        obt_other = (dd[(k + 1) % 3] < 0) | (dd[(k + 2) % 3] < 0)
        vor = 0.125 * ((e[k] * cot[(k + 2) % 3][:, None]) ** 2).sum(1) + \
              0.125 * ((e[(k + 2) % 3] * cot[(k + 1) % 3][:, None]) ** 2).sum(1)
        a = np.where(obt_here, 0.25 * tri, np.where(obt_other, 0.125 * tri, vor))
        np.add.at(M, F[:, k], np.where(ok, a, 0))
    return M


def _neighbours(F, n):
    A = sp.coo_matrix((np.ones(3 * len(F)), (F[:, [0, 1, 2]].ravel(), F[:, [1, 2, 0]].ravel())), shape=(n, n))
    A = ((A + A.T) > 0).tocsr()
    return [A.indices[A.indptr[k]:A.indptr[k + 1]] for k in range(n)]


def fairing(V, F, props, iterations=2):
    """[laplacianFairing, k = 3] lock the topological boundary, its 1-ring and 2-ring (2-ring vertices that are not
    'handled' stay free); solve L M^-1 L x = 0 on the free vertices; repeat with the new positions."""
    n = len(V)
    nb = _neighbours(F, n)
    he = set(map(tuple, np.vstack((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))))
    boundary = np.zeros(n, bool)
    for a, b in he:
        if (b, a) not in he:
            boundary[a] = boundary[b] = True
    locked = np.zeros(n, bool)
    for _ in range(iterations):
        for v in np.where(boundary)[0]:
            locked[v] = True
            for vv in nb[v]:
                locked[vv] = True
                for vvv in nb[vv]:
                    locked[vvv] = props['handled'][vvv]
        L = _cot_laplacian_clamped(V, F)
        M = _mass(V, F)
        A = (L @ sp.diags(1.0 / M) @ L).tocsr()
        fr, fx = np.where(~locked)[0], np.where(locked)[0]
        solve = spl.factorized(A[fr][:, fr].tocsc())
        X = V.copy()
        for c in range(3):
            X[fr, c] = solve(-A[fr][:, fx] @ V[fx, c])
        V = X
    return V


# -------------------------------------------------------------------------------------------- [SubDivMesh.cpp]
def vertex_normals(V, F, props):
    """[calculate_vertex_normals, type 0] vertices without a preset normal: mean of the unit face normals."""
    fn = _unit(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]))
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N = _unit(N)
    out = props['normal'].copy()
    out[~props['has_n']] = N[~props['has_n']]
    return out


def _H(ni, nj, pi, pj):
    s = ni + nj
    den = (ni * s).sum(-1)
    ok = (np.linalg.norm(s, axis=-1) > 0) & (np.linalg.norm(ni, axis=-1) > 0)
    return np.where(ok, (s * (pj - pi)).sum(-1) / np.where(ok, den, 1), 0.0)


def pn_loop(V, N, F, props, E, r, d):
    """[PNLoopSubDiv] one PN-Loop step; circle (boundary) vertices fixed, circle edges split on the circle."""
    n = len(V)
    nb = _neighbours(F, n)
    bnd, strut = props['bnd'], props['strut']
    Vn, Nn = V.copy(), N.copy()
    for v in np.where(~bnd)[0]:
        js = nb[v]
        k = len(js)
        al = (0.625 - (0.375 + 0.25 * np.cos(2 * np.pi / k)) ** 2) / k if k > 3 else 0.1875
        p = V[v] * (1 - k * al) + V[js].sum(0) * al
        nv = N[v]                                                   # the C++ averages n_v with itself
        h = _H(nv, N[v], p, V[v]) * (1 - k * al) + al * _H(nv[None], N[js], p[None], V[js]).sum()
        Vn[v] = p + nv * h
    # edges, with the opposite vertex of each half-edge
    opp = {}
    for a, b, c in F:
        opp[(a, b)], opp[(b, c)], opp[(c, a)] = c, a, b
    edges = sorted({(min(a, b), max(a, b)) for a, b in opp})
    eid = {e: n + k for k, e in enumerate(edges)}
    Ve, Ne, ps, bs = [], [], [], []
    for a, b in edges:
        if bnd[a] and bnd[b]:
            i = strut[a]
            c = d[i] * E[i]
            vec = _unit(_unit(V[a] - c) + _unit(V[b] - c))
            Ve.append(c + vec * r[i]); Ne.append(vec); ps.append(max(strut[a], strut[b])); bs.append(True)
        else:
            p, q = opp[(a, b)], opp[(b, a)]
            t = 0.375 * (V[a] + V[b]) + 0.125 * (V[p] + V[q])
            ne = _unit(0.375 * (N[a] + N[b]) + 0.125 * (N[p] + N[q]))
            h = 0.375 * (_H(ne, N[a], t, V[a]) + _H(ne, N[b], t, V[b])) + \
                0.125 * (_H(ne, N[p], t, V[p]) + _H(ne, N[q], t, V[q]))
            Ve.append(t + ne * h); Ne.append(ne); ps.append(max(strut[a], strut[b])); bs.append(False)
    F2 = []
    for a, b, c in F:
        mab, mbc, mca = eid[(min(a, b), max(a, b))], eid[(min(b, c), max(b, c))], eid[(min(c, a), max(c, a))]
        F2 += [(mab, mbc, mca), (mab, b, mbc), (mbc, c, mca), (mca, a, mab)]
    props2 = dict(strut=np.r_[strut, ps], bnd=np.r_[bnd, bs])
    return np.vstack((Vn, Ve)), np.vstack((Nn, Ne)), np.array(F2), props2


def subdiv_lattice_node(node, E, r, lap_iter=2, subdiv_iter=3):
    """SF node patch at `node` (struts along unit E, radii r) as the authors' code builds it. Same dict as
    soapFilm.soap_film_node (subdiv_iter = 3 as the paper, Fig. 15; the C++ main uses 1, same shape within 1 %): V, F, N, circ (strut of each cut-circle vertex, else -1), upar (circle angle),
    d (circle distances), E, r, ctrl = the faired initial mesh."""
    E = _unit(np.asarray(E, float))
    r = np.broadcast_to(np.asarray(r, float), (len(E),)).copy()
    c = E.mean(0)
    if np.linalg.eigvalsh((E - c).T @ (E - c))[0] < 1e-10 and len(E) > 3:
        raise ValueError('coplanar struts (checkCoplanar)')
    d = cut_lengths(E, r)
    V, F, props = initial_mesh(E, r, d)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if (fn * V[F].mean(1)).sum() < 0:                               # outward, so face normals match the radial ones
        F = F[:, ::-1].copy()
    V = fairing(V, F, props, lap_iter)
    ctrl = (V + node, F.copy())
    N = vertex_normals(V, F, props)
    for _ in range(subdiv_iter):
        V, N, F, props = pn_loop(V, N, F, props, E, r, d)
    circ = np.where(props['bnd'], props['strut'], -1)
    upar = np.zeros(len(V))
    for i, e in enumerate(E):
        s = circ == i
        u, v = _frame(e)
        upar[s] = np.arctan2(V[s] @ v, V[s] @ u)
    return dict(V=V + node, F=F, N=N, circ=circ, upar=upar, d=d, E=E, r=r, ctrl=ctrl)
