"""FHS vs the soap-film method (Luo & Zou 2025) and the other node methods (notes/16_SoapFilm_vs_FHS.md).
Run: PYTHONPATH=src python tests/study_soap_vs_fhs.py [step]
"""
import sys
import time

import numpy as np
import trimesh

import nodeJoint as nj
import soapFilm as sfm

OUT = 'STL/soap_vs_fhs'


# ------------------------------------------------------------------------------------------------- node cases
def regular(n):
    """Regular nodes of SF Table 1: 6 = octahedron axes, 12 = icosahedron vertices, 20 = dodecahedron vertices."""
    if n == 6:
        return np.vstack((np.eye(3), -np.eye(3)))
    g = (1 + 5 ** 0.5) / 2
    if n == 'cuboct':                                             # FCC directions (12, nearest 60 deg)
        P = [v for v in __import__('itertools').product((-1, 0, 1), repeat=3) if sum(map(abs, v)) == 2]
    elif n == 12:
        P = [(0, a, b * g) for a in (-1, 1) for b in (-1, 1)]
        P = P + [(a, b * g, 0) for a in (-1, 1) for b in (-1, 1)] + [(b * g, 0, a) for a in (-1, 1) for b in (-1, 1)]
    else:
        P = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
        for a in (-1, 1):
            for b in (-1, 1):
                P += [(0, a / g, b * g), (a / g, b * g, 0), (b * g, 0, a / g)]
    P = np.array(P, float)
    return P / np.linalg.norm(P, axis=1)[:, None]


def irregular(n, seed=0, min_deg=None):
    """Random directions with a minimum pairwise angle (the paper's irregular nodes are not given; ours are
    seeded): min angle 0.8 x the regular node's nearest-neighbour angle."""
    reg = regular(n)
    a_reg = np.degrees(np.arccos(np.sort(reg @ reg.T, 1)[:, -2].max()))
    min_deg = min_deg or 0.8 * a_reg
    rng = np.random.default_rng(seed)
    P = []
    while len(P) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if all(np.degrees(np.arccos(np.clip(v @ p, -1, 1))) > min_deg for p in P):
            P.append(v)
    return np.array(P)


# --------------------------------------------------------------------------------------- reference and metrics
def boolean_node(E, r, L, seg=256, h=0.05):
    """Original nodal shape: union of the strut cylinders + the node sphere (max r), finely remeshed."""
    import manifold3d as m3
    parts = [nj.cylinder(np.zeros(3), e, ri, l, seg=seg) for e, ri, l in zip(E, r, L)]
    parts.append(m3.Manifold.sphere(float(r.max()), seg))
    m = nj.to_trimesh(m3.Manifold.batch_boolean(parts, m3.OpType.Add))
    V, F = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=h)
    return trimesh.Trimesh(V, F, process=False)


def sample(V, F, n, seed=0, mask=None):
    """Area-uniform samples on the faces F (optionally only faces with mask)."""
    if mask is not None:
        F = F[mask]
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    rng = np.random.default_rng(seed)
    fi = rng.choice(len(F), n, p=A / A.sum())
    u, v = rng.random(n), rng.random(n)
    fl = u + v > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    T = V[F[fi]]
    return T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])


def deviation_stats(P, ref):
    d = nj.surface_distance(P, ref, ring=6)
    return dict(max=float(d.max()), avg=float(d.mean()), std=float(d.std()))


def g1_jump(sf, axial=False):
    """Max angle (deg) between the patch normal at the boundary and the cylinder normal; axial=True: only its
    tilt along the strut axis, asin(n.e), the tangency (G1) part (the rest is the chord angle of the facets)."""
    V, F, circ = sf['V'], sf['F'], sf['circ']
    m = trimesh.Trimesh(V, F, process=False)
    out = 0.0
    for i, e in enumerate(sf['E']):
        b = np.where(circ == i)[0]
        fb = np.unique(np.concatenate([m.vertex_faces[k][m.vertex_faces[k] >= 0] for k in b]))
        fn = m.face_normals[fb]
        c = V[F[fb]].mean(1)
        rad = c - np.outer(c @ e, e)
        rad /= np.linalg.norm(rad, axis=1)[:, None]
        if axial:
            out = max(out, np.degrees(np.arcsin(np.clip(np.abs(fn @ e), 0, 1))).max())
        else:
            out = max(out, np.degrees(np.arccos(np.clip((fn * rad).sum(1), -1, 1))).max())
    return out


def step_s1s2():
    """S1 gates + S2: SF Table 1 (r = 1, lambda = 0.3)."""
    paper = {6: (0.2846, 0.0647, 0.0715), 12: (0.3021, 0.0645, 0.0751), 20: (0.6653, 0.1009, 0.1394)}
    print('| node | d (cut) | faces | watertight | boundary err | G1 jump deg | max / avg / std dev (ours) | paper | time s |')
    print('|---|---|---|---|---|---|---|---|---|')
    paper['cuboct'] = paper[12]
    for n in (6, 12, 'cuboct', 20):
        E = regular(n)
        r = np.ones(len(E))
        t = time.time()
        sf = sfm.soap_film_node(np.zeros(3), E, r)
        dt = time.time() - t
        L = np.full(len(E), sf['d'].max() + 3.0)
        m = sfm.node_solid(sf, L)
        m.export(f'{OUT}/SF_regular{n}.stl')
        b = sf['circ'] >= 0
        Vb = sf['V'][b]
        ax = np.array([Vb[k] @ sf['E'][c] for k, c in enumerate(sf['circ'][b])])
        rad = np.linalg.norm(Vb - ax[:, None] * sf['E'][sf['circ'][b]], axis=1)
        berr = max(np.abs(rad - 1).max(), np.abs(ax - sf['d'][sf['circ'][b]]).max())
        ref = boolean_node(sf['E'], r, L)
        P = sample(sf['V'], sf['F'], 200_000)
        s = deviation_stats(P, ref)
        print(f"| regular-{n} | {sf['d'][0]:.3f} | {len(sf['F'])} | {m.is_watertight and m.is_volume} | {berr:.1e} | "
              f"{g1_jump(sf):.2f} | {s['max']:.4f} / {s['avg']:.4f} / {s['std']:.4f} | "
              f"{paper[n][0]} / {paper[n][1]} / {paper[n][2]} | {dt:.2f} |", flush=True)



def hull_node(E, r, L, d):
    """Published convex hull joint (VAE paper, Fig. 25): convex hull of the strut end circles (here at the SF cut
    d_i, the minimum non-intersecting cut), Boolean-united with the struts from d_i to L_i."""
    import manifold3d as m3
    pts = []
    for e, ri, di in zip(E, r, d):
        u, v = nj._frame(e)
        t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        pts.append(di * e + ri * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)))
    parts = [m3.Manifold.hull_points(np.vstack(pts).tolist())]
    parts += [nj.cylinder(np.zeros(3), e, ri, l, seg=128, start=di - 0.05 * ri)       # overlap: no coplanar caps
              for e, ri, l, di in zip(E, r, L, d)]
    return nj.to_trimesh(m3.Manifold.batch_boolean(parts, m3.OpType.Add))


def ball_volume(m, R):
    import manifold3d as m3
    man = m3.Manifold(m3.Mesh(np.ascontiguousarray(m.vertices, np.float32), np.ascontiguousarray(m.faces, np.uint32)))
    return float((man ^ m3.Manifold.sphere(float(R), 256)).volume())


def min_section(m, E, r, R):
    """Smallest equivalent radius / r_i along each strut axis into the node (0.05 r .. R), notes/12."""
    import study_three_beam_joint as tb
    tb.NODE = np.zeros(3)
    out = np.inf
    for e, ri in zip(E, r):
        S, Rr = tb.section_profile(m, e, s_max=R, ds=0.05)
        out = min(out, np.nanmin(Rr) / ri)
    return out


def metrics(m, E, r, R, ref, n_samp=150_000):
    V = np.asarray(m.vertices)
    inb = np.linalg.norm(V, axis=1) < R
    fin = np.linalg.norm(V[m.faces].mean(1), axis=1) < R
    P = sample(V, np.asarray(m.faces), n_samp, mask=fin)
    dv = deviation_stats(P, ref)
    ang = np.degrees(m.face_adjacency_angles)
    ina = inb[m.face_adjacency_edges].all(1)
    H = mean_curvature(m)
    return dict(ok=bool(m.is_watertight and m.is_volume), dev_avg=dv['avg'], dev_max=dv['max'],
                conc=concave_radius(m, inb) / r.min(), dih=float(ang[ina].max()),
                H99=float(np.quantile(np.abs(H[inb]) * r.min(), 0.99)), vol=ball_volume(m, R),
                sec=min_section(m, E, r, 0.9 * R))


def run_case(name, E, r, methods=('Boolean', 'Hull', 'SF', 'FHS'), save=True):
    E = E / np.linalg.norm(E, axis=1)[:, None]
    r = np.asarray(r, float)
    t = time.time()
    sf = sfm.soap_film_node(np.zeros(3), E, r)
    t_sf = time.time() - t
    L = np.full(len(E), max(sf['d'].max(), 2.5 * r.max()) + 4 * r.max())
    out, meshes = {}, {}
    if 'FHS' in methods:
        t = time.time()
        try:
            fm, R_s = fhs_node(E, r, L)
            meshes['FHS'] = fm
            out['FHS'] = dict(time=time.time() - t)
            t_fhs = nj.ring_offsets(E, r, R_s, core=True)
        except Exception as ex:                                    # recorded, not hidden
            out['FHS'] = dict(fail=f'{type(ex).__name__}: {ex}'[:80])
            t_fhs = np.zeros(len(E))
    if 'FHS' not in methods:                                       # same ball as with FHS
        t_fhs = nj.ring_offsets(E, r, nj.trim_radius(E, r, L, dt=0.005)[0], core=True)
    R = max(sf['d'].max(), (t_fhs + r).max()) + 0.5 * r.max()
    meshes['SF'] = sfm.node_solid(sf, L)
    out['SF'] = dict(time=t_sf)
    ref = boolean_node(E, r, L)
    if 'Boolean' in methods:
        import manifold3d as m3
        parts = [nj.cylinder(np.zeros(3), e, ri, l, seg=128) for e, ri, l in zip(E, r, L)]
        parts.append(m3.Manifold.sphere(float(r.max()), 128))
        meshes['Boolean'] = nj.to_trimesh(m3.Manifold.batch_boolean(parts, m3.OpType.Add))
        out['Boolean'] = dict(time=0.0)
    if 'Hull' in methods:
        t = time.time()
        meshes['Hull'] = hull_node(E, r, L, sf['d'])
        out['Hull'] = dict(time=time.time() - t)
    for k, m in meshes.items():
        if save:
            m.export(f'{OUT}/{name}_{k}.stl')
            m = trimesh.load(f'{OUT}/{name}_{k}.stl', process=True)      # metrics on the written STL
        if k in ('Boolean', 'Hull'):                                      # long flat faces: refine for sampling
            m = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=0.1 * r.min()))
        out[k].update(metrics(m, E, r, R, ref))
    vb = out['Boolean']['vol'] if 'Boolean' in out else None
    for k in out:
        if vb and 'vol' in out[k]:
            out[k]['dvol'] = 100 * (out[k]['vol'] / vb - 1)
    return out, R


HEAD = ('| case | method | valid | mean / max dev. from Boolean (mm) | tightest concave radius / r | max dihedral deg | '
        '99 % abs(H) r | volume in ball vs Boolean | min section / r | time s |\n|---|---|---|---|---|---|---|---|---|---|')


def row(name, k, o):
    if 'fail' in o:
        return f'| {name} | {k} | FAIL: {o["fail"]} | | | | | | | |'
    return (f"| {name} | {k} | {o['ok']} | {o['dev_avg']:.4f} / {o['dev_max']:.3f} | {o['conc']:.3f} | {o['dih']:.0f} | "
            f"{o['H99']:.2f} | {o.get('dvol', 0):+.1f} % | {o['sec']:.3f} | {o['time']:.2f} |")


def step_hull():
    """Re-run the hull method only (after the coplanar-cap fix)."""
    print(HEAD)
    for nm, E in [('regular-6', regular(6)), ('regular-12', regular(12)), ('regular-20', regular(20)),
                  ('irregular-6', irregular(6)), ('irregular-12', irregular(12)), ('irregular-20', irregular(20))]:
        out, R = run_case(nm, E, np.ones(len(E)), methods=('Boolean', 'Hull'))
        print(row(nm, 'Hull', out['Hull']), flush=True)


def step_s4():
    """S4: the three-beam sweep of notes/12 (4 angle layouts x 5 radius sets, graded radii, a 50 deg acute pair)
    plus a 30 deg acute pair; SF with its unequal-radius cut generalisation."""
    import study_three_beam_sweep as sw
    angles = dict(sw.ANGLES, acute_30_120_110=(30, 120, 110))
    print(HEAD)
    for an, ang in angles.items():
        try:
            E = sw.from_angles(*ang)
        except ValueError as ex:
            print(f'| {an} | - | not realisable: {ex} | | | | | | | |')
            continue
        for rn, rad in sw.RADII.items():
            nm = f'{an}_{rn}'
            try:
                out, R = run_case(nm, E, np.array(rad))
            except Exception as ex:
                print(f'| {nm} | all | FAIL: {type(ex).__name__}: {str(ex)[:80]} | | | | | | | |', flush=True)
                continue
            for k in ('Boolean', 'Hull', 'SF', 'FHS'):
                print(row(nm, k, out[k]), flush=True)


def tetra():
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / 3 ** 0.5


def build_all(E, r, L):
    """Node solids of every method for struts E, radii r, lengths L (STL-ready trimeshes) + SF cuts + FHS rings."""
    import manifold3d as m3
    sf = sfm.soap_film_node(np.zeros(3), E, r)
    fm, R_s = fhs_node(E, r, L)
    parts = [nj.cylinder(np.zeros(3), e, ri, l, seg=128) for e, ri, l in zip(E, r, L)]
    parts.append(m3.Manifold.sphere(float(r.max()), 128))
    M = dict(Boolean=nj.to_trimesh(m3.Manifold.batch_boolean(parts, m3.OpType.Add)), Hull=hull_node(E, r, L, sf['d']),
             SF=sfm.node_solid(sf, L), FHS=fm)
    return M, sf['d'], nj.ring_offsets(E, r, R_s, core=True)


def fe_peak(stl, E, r, L, R, h_in, fixed=0, nu=0.3, verbose=False):
    """Linear-static P2 tet FE (scikit-fem) of one node solid: strut `fixed` clamped at its far end, every other
    strut pulled along its axis at its far end with the same force F = pi r_min^2 (unit nominal stress in the
    thinnest strut); for a tetrahedral node the forces balance the clamp reaction exactly (sum e_i = 0), so the
    clamped strut is in pure tension. Returns peak von Mises in the node ball |x| < R (= K vs the thinnest strut's
    nominal stress), its location, and the element count."""
    sys.path.insert(0, 'abaqus')
    import solid
    from skfem import MeshTet, Basis, FacetBasis, ElementTetP2, ElementVector, asm, condense, solve, LinearForm
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    P, T = solid.gmsh_from_stl(stl, h=max(r.max() / 1.5, 2 * h_in), curvature=0, algo3d=1,
                               ball_refine=((0, 0, 0), R, h_in, 0.5 * R))
    mesh = MeshTet(P.T.copy(), T.T.copy())
    el = ElementVector(ElementTetP2())
    ib = Basis(mesh, el, intorder=4)
    lam, mu = lame_parameters(1.0, nu)
    K = asm(linear_elasticity(lam, mu), ib)
    f = np.zeros(K.shape[0])
    F = np.pi * r.min() ** 2
    D = None
    for k, (e, ri, l) in enumerate(zip(E, r, L)):
        fac = mesh.facets_satisfying(lambda x, e=e, l=l: (x.T @ e) > l - 1e-3)
        if k == fixed:
            D = ib.get_dofs(fac).all()
            continue
        t = e * F / (np.pi * ri ** 2)

        @LinearForm
        def trac(v, w, t=t):
            return v[0] * t[0] + v[1] * t[1] + v[2] * t[2]
        f += asm(trac, FacetBasis(mesh, el, facets=fac, intorder=4))
    import pyamg                                                    # AMG-CG: the graded meshes exceed SuperLU memory
    A, b, x0, I = condense(K, f, D=D)
    Xd = ib.doflocs[:, I].T                                         # rigid-body modes of the free dofs
    B = np.zeros((len(I), 6))
    comp = I % 3                                                    # ElementVector dofs interleave x, y, z
    for c in range(3):
        B[comp == c, c] = 1.0
    for k, (a_, b_) in enumerate(((0, 1), (1, 2), (0, 2))):
        B[comp == a_, 3 + k] = -Xd[comp == a_, b_]
        B[comp == b_, 3 + k] = Xd[comp == b_, a_]
    ml = pyamg.smoothed_aggregation_solver(A.tocsr(), B=B)
    res = []
    xI = ml.solve(b, tol=1e-9, accel='cg', maxiter=500, residuals=res)
    if res[-1] > 1e-7 * res[0]:
        raise RuntimeError(f'AMG-CG did not converge: {res[-1] / res[0]:.1e}')
    u = np.zeros(K.shape[0])
    u[I] = xI
    G = ib.interpolate(u).grad                                      # (3, 3, nel, nqp)
    eps = 0.5 * (G + G.transpose(1, 0, 2, 3))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    S = 2 * mu * eps
    for i in range(3):
        S[i, i] += lam * tr
    vm = np.sqrt(0.5 * ((S[0, 0] - S[1, 1]) ** 2 + (S[1, 1] - S[2, 2]) ** 2 + (S[2, 2] - S[0, 0]) ** 2)
                 + 3 * (S[0, 1] ** 2 + S[1, 2] ** 2 + S[0, 2] ** 2))
    X = ib.global_coordinates().value                               # (3, nel, nqp)
    inb = np.linalg.norm(X, axis=0) < R
    k = np.argmax(np.where(inb, vm, -1))
    i, j = np.unravel_index(k, vm.shape)
    return dict(K=float(vm[i, j]), at=X[:, i, j], nel=len(T), ndof=K.shape[0])


def step_s5():
    """S5: stress concentration at a tetrahedral node (equal and graded radii), two mesh sizes."""
    print('| node | method | K (h = r_min/4) | K (h = r_min/6) | change | peak at |x| / R | tets (fine) |')
    print('|---|---|---|---|---|---|---|')
    cases = (('tetra_r080', np.full(4, 0.8)), ('tetra_graded_100_080_060_040', np.array([1.0, 0.8, 0.6, 0.4])))
    for nm, r in cases[1:] if 'graded' in sys.argv else cases:
        E = tetra()
        L0 = np.full(4, 8.0)
        _, d, tf = build_all(E, r, L0)
        R = max(d.max(), (tf + r).max()) + 0.5 * r.max()
        L = R + 3 * r
        M, _, _ = build_all(E, r, L)
        for k, m in M.items():
            stl = f'{OUT}/{nm}_{k}_fe.stl'
            m.export(stl)
            res = []
            for h in (r.min() / 4, r.min() / 6):
                try:
                    res.append(fe_peak(stl, E, r, L, R, h))
                except Exception as ex:
                    res.append(dict(fail=f'{type(ex).__name__}: {str(ex)[:60]}'))
            if any('fail' in q for q in res):
                print(f"| {nm} | {k} | {res[0].get('K', res[0].get('fail'))} | {res[1].get('K', res[1].get('fail'))} | | | |",
                      flush=True)
                continue
            ch = 100 * (res[1]['K'] / res[0]['K'] - 1)
            print(f"| {nm} | {k} | {res[0]['K']:.3f} | {res[1]['K']:.3f} | {ch:+.1f} % | "
                  f"{np.linalg.norm(res[1]['at']) / R:.2f} | {res[1]['nel']} |", flush=True)


def step_s5b():
    """S5 bracket for SF: the most-rounded SF variant (L1 fairing, 1 fixed ring) and the paper-text one, three
    mesh sizes, equal-radius tetrahedral node (same L, R as S5)."""
    E, r = tetra(), np.full(4, 0.8)
    _, d, tf = build_all(E, r, np.full(4, 8.0))
    R = max(d.max(), (tf + r).max()) + 0.5 * r.max()
    L = R + 3 * r
    print('| SF variant | K h = r/4 | r/6 | r/8 | tets (r/8) |')
    print('|---|---|---|---|---|')
    for nm, kw in (('paper text', {}), ('L1 + 1 ring (most rounded)', dict(fair_order=1, fixed_rows=1))):
        stl = f"{OUT}/tetra_r080_SF_{'round' if kw else 'paper'}_fe.stl"
        sfm.node_solid(sfm.soap_film_node(np.zeros(3), E, r, **kw), L).export(stl)
        res = [fe_peak(stl, E, r, L, R, r.min() / k) for k in (4, 6, 8)]
        print(f"| {nm} | " + ' | '.join(f"{q['K']:.3f}" for q in res) + f" | {res[-1]['nel']} |", flush=True)


def step_sfcal():
    """Re-run SF (paper-calibrated fixed band, soapFilm default) on the S3 and S4 nodes and the S5 tetra FE."""
    import study_three_beam_sweep as sw
    cases = [('regular-6', regular(6), np.ones(6)), ('regular-12', regular(12), np.ones(12)),
             ('regular-20', regular(20), np.ones(20)), ('irregular-6', irregular(6), np.ones(6)),
             ('irregular-12', irregular(12), np.ones(12)), ('irregular-20', irregular(20), np.ones(20))]
    for an, ang in dict(sw.ANGLES, acute_30_120_110=(30, 120, 110)).items():
        for rn, rad in sw.RADII.items():
            cases.append((f'{an}_{rn}', sw.from_angles(*ang), np.array(rad)))
    print(HEAD)
    for nm, E, r in cases:
        out, R = run_case(nm + '_cal', E, r, methods=('Boolean', 'SF'))
        print(row(nm, 'SF cal', out['SF']), flush=True)
    E, r = tetra(), np.full(4, 0.8)
    _, d, tf = build_all(E, r, np.full(4, 8.0))
    R = max(d.max(), (tf + r).max()) + 0.5 * r.max()
    L = R + 3 * r
    stl = f'{OUT}/tetra_r080_SF_cal_fe.stl'
    sfm.node_solid(sfm.soap_film_node(np.zeros(3), E, r), L).export(stl)
    res = [fe_peak(stl, E, r, L, R, r.min() / k) for k in (4, 6, 8)]
    print('| tetra_r080 FE | SF cal | K = ' + ' / '.join(f"{q['K']:.3f}" for q in res) + ' (h = r/4, r/6, r/8) |', flush=True)


PAPER_T1 = {'regular-6': ((0.2846, 0.0647, 0.0715), (0.2832, 0.1732, 0.0737)),
            'regular-12': ((0.3021, 0.0645, 0.0751), (0.3021, 0.1808, 0.0793)),
            'regular-20': ((0.6653, 0.1009, 0.1394), (0.6958, 0.3279, 0.1666)),
            'irregular-6': ((1.1082, 0.1643, 0.2205), (0.8457, 0.2254, 0.1659)),
            'irregular-12': ((0.9762, 0.1442, 0.1690), (0.8029, 0.1972, 0.1481)),
            'irregular-20': ((1.3798, 0.1681, 0.2480), (0.9722, 0.2059, 0.1601))}   # (SF, Savio): max, avg, std


def paper_nodes():
    return [('regular-6', regular(6)), ('regular-12', regular(12)), ('regular-20', regular(20)),
            ('irregular-6', irregular(6)), ('irregular-12', irregular(12)), ('irregular-20', irregular(20))]


def step_paperfigs():
    """Recreate the SF paper's Table 1 and Figs. 12, 14, 15 (SF and the Savio-like baseline, r = 1, lambda = 0.3).
    Figures: ResultsFig/explicit/70_SF_paper_fig12.png, 71_SF_paper_fig14.png, 72_SF_paper_fig15.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    nodes = paper_nodes()
    res = {}
    print('| node | method | max / avg / std deviation (ours) | paper | ours vs paper (avg) |')
    print('|---|---|---|---|---|')
    for nm, E in nodes:
        r = np.ones(len(E))
        out = {}
        for meth, fn in (('Savio', sfm.savio_node), ('SF', sfm.soap_film_node)):
            sf = fn(np.zeros(3), E, r)
            ref = boolean_node(sf['E'], r, np.full(len(E), sf['d'].max() + 3.0))
            P = sample(sf['V'], sf['F'], 400_000)
            dv = nj.surface_distance(P, ref, ring=6)
            m = sfm.node_solid(sf, sf['d'] + 0.5)
            m.export(f'{OUT}/paper_{nm}_{meth}.stl')
            m = trimesh.load(f'{OUT}/paper_{nm}_{meth}.stl', process=True)
            out[meth] = dict(sf=sf, dev=dv, mesh=m, H=mean_curvature(m))
            pk = PAPER_T1[nm][0 if meth == 'SF' else 1]
            print(f"| {nm} | {meth} | {dv.max():.4f} / {dv.mean():.4f} / {dv.std():.4f} | {pk[0]} / {pk[1]} / {pk[2]} | "
                  f"{dv.mean() / pk[1]:.2f} |", flush=True)
        res[nm] = out
    # Fig. 12: graph, Savio control mesh + surface, SF control mesh + surface
    fig = plt.figure(figsize=(18, 15))
    for k, (nm, E) in enumerate(nodes):
        sf = res[nm]['SF']['sf']
        ax = fig.add_subplot(5, 6, 1 + k, projection='3d')
        for e, dd in zip(sf['E'], sf['d']):
            ax.plot(*np.c_[np.zeros(3), dd * e], 'k', lw=0.6)
            u, v = nj._frame(e)
            t = np.linspace(0, 2 * np.pi, 60)
            ax.plot(*(dd * e[:, None] + np.outer(u, np.cos(t)) + np.outer(v, np.sin(t))), 'r', lw=0.6)
        lim = sf['d'].max() + 1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim); ax.set_box_aspect((1, 1, 1))
        ax.view_init(25, 35); ax.set_axis_off(); ax.set_title(nm, fontsize=9)
        for row, meth in ((1, 'Savio'), (3, 'SF')):
            o = res[nm][meth]
            Vc, Fc = o['sf']['ctrl']
            mc = trimesh.Trimesh(Vc, Fc, process=False)
            lim = 1.1 * (o['sf']['d'].max() + 0.6)
            render_z(fig.add_subplot(5, 6, 1 + k + 6 * row), mc, np.zeros(len(Vc)), 1.0,
                     f'{meth}{"-like" if meth == "Savio" else ""} control mesh', lim, n_px=360, n_pts=600_000,
                     grey=True, edges=True)
            pm = trimesh.Trimesh(o['sf']['V'], o['sf']['F'], process=False)       # the nodal patch (open circles)
            render_z(fig.add_subplot(5, 6, 1 + k + 6 * (row + 1)), pm, np.zeros(len(pm.vertices)), 1.0,
                     f'{meth}{"-like" if meth == "Savio" else ""} surface', lim, n_px=360, n_pts=1_500_000, grey=True)
    fig.savefig('ResultsFig/explicit/70_SF_paper_fig12.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    # Fig. 14: mean curvature, red (+) -> green (0) -> blue (-), one symmetric range per node for both methods
    fig = plt.figure(figsize=(18, 6.5))
    for k, (nm, _) in enumerate(nodes):
        hl = max(np.quantile(np.abs(res[nm][m]['H']), 0.995) for m in ('Savio', 'SF'))
        for row, meth in enumerate(('Savio', 'SF')):
            o = res[nm][meth]
            render_z(fig.add_subplot(2, 6, 1 + k + 6 * row), o['mesh'], o['H'], 1.0,
                     f'{meth}{"-like" if meth == "Savio" else ""}: {nm} (H r, +-{hl:.1f})', 1.1 * (o['sf']['d'].max() + 0.6),
                     n_px=360, n_pts=1_500_000, cmap='jet', hlim=hl)
    fig.savefig('ResultsFig/explicit/71_SF_paper_fig14.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    # Fig. 15: deviation histograms
    fig, axs = plt.subplots(2, 6, figsize=(20, 6))
    for k, (nm, _) in enumerate(nodes):
        for row, meth in enumerate(('Savio', 'SF')):
            dv = res[nm][meth]['dev']
            ax = axs[row, k]
            cnt, edges_ = np.histogram(dv, bins=120, range=(0, dv.max()))
            ax.bar(edges_[:-1], cnt, width=np.diff(edges_), align='edge', color=cm.jet(edges_[:-1] / edges_[-1]))
            ax.set_title(f'{meth}{"-like" if meth == "Savio" else ""}: {nm}', fontsize=9)
            ax.set_xlabel('Deviations (mm)', fontsize=8); ax.set_ylabel('Counts', fontsize=8); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig('ResultsFig/explicit/72_SF_paper_fig15.png', dpi=100)
    plt.close(fig)


def poly_edges(faces):
    return np.array(sorted({(min(a, b), max(a, b)) for f in faces for a, b in zip(f, list(f[1:]) + [f[0]])}))


def poly_tris(faces):
    return np.array([[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)])


def build3(E, r, L):
    """Savio (2018), SF (Luo & Zou 2025, R3) and FHS solids of one node with flat far ends at L, the control meshes
    (V, polygon faces) and build times; plus the ball radius that contains every method's node region."""
    out = {}
    t = time.time()
    m, ctrl, info = sfm.savio_node(np.zeros(3), E, r, L, levels=4)    # 4 CC levels: face count as SF / FHS
    out['Savio'] = dict(mesh=m, ctrl=ctrl, time=time.time() - t, extent=(info['t'] + r).max())
    t = time.time()
    sf = sfm.soap_film_node(np.zeros(3), E, r)
    t_sf = time.time() - t                                           # the node patch (struts are just tubes)
    out['SF'] = dict(mesh=sfm.node_solid(sf, L), ctrl=(sf['ctrl'][0], [list(f) for f in sf['ctrl'][1]]),
                     time=t_sf, extent=sf['d'].max(), sf=sf)
    t = time.time()
    try:
        fm, R_s = fhs_node(E, r, L)
        c = fhs_node.ctrl
        out['FHS'] = dict(mesh=fm, ctrl=(np.asarray(c.vertices), [list(f) for f in np.asarray(c.faces)]),
                          time=time.time() - t, extent=(nj.ring_offsets(E, r, R_s, core=True) + 2 * r).max())
    except Exception as ex:
        out['FHS'] = dict(fail=f'{type(ex).__name__}: {str(ex)[:60]}')
    R = max(o['extent'] for o in out.values() if 'extent' in o) + 0.5 * r.max()
    return out, R


METHODS3 = ('Savio', 'SF', 'FHS')


def metrics3(name, E, r):
    E = E / np.linalg.norm(E, axis=1)[:, None]
    r = np.asarray(r, float)
    sf0 = sfm.soap_film_node(np.zeros(3), E, r)
    L = np.full(len(E), max(sf0['d'].max(), 2.5 * r.max()) + 4 * r.max())
    out, R = build3(E, r, L)
    L = np.full(len(E), R + 3 * r.max())                            # struts well past the common ball
    out, R = build3(E, r, L)
    ref = boolean_node(E, r, L)
    vb = ball_volume(ref, R)
    for k in METHODS3:
        o = out[k]
        if 'fail' in o:
            continue
        f = f'{OUT}/m3_{name}_{k}.stl'
        o['mesh'].export(f)
        m = trimesh.load(f, process=True)
        o['stl'] = f
        o['m'] = m
        o.update(metrics(m, E, r, R, ref))
        o['dvol'] = 100 * (o['vol'] / vb - 1)
        V = np.asarray(m.vertices)
        P = sample(V, np.asarray(m.faces), 300_000, mask=np.linalg.norm(V[m.faces].mean(1), axis=1) < R)
        o['dev'] = nj.surface_distance(P, ref, ring=6)
    return out, R, E, r


HEAD3 = ('| case | method | valid | mean / max / std dev. from Boolean (mm) | tightest concave radius / r | max dihedral deg | '
         '99 % abs(H) r | volume in ball vs Boolean | min section / r | build time s |\n|---|---|---|---|---|---|---|---|---|---|')


def row3(name, k, o):
    if 'fail' in o:
        return f'| {name} | {k} | FAIL: {o["fail"]} | | | | | | | |'
    return (f"| {name} | {k} | {o['ok']} | {o['dev'].mean():.4f} / {o['dev'].max():.3f} / {o['dev'].std():.3f} | "
            f"{o['conc']:.3f} | {o['dih']:.0f} | {o['H99']:.2f} | {o['dvol']:+.1f} % | {o['sec']:.3f} | {o['time']:.2f} |")


def step_all3():
    """Savio vs SF vs FHS on the SF paper's six nodes (+ Figs. 12, 14, 15 with all three) and the 25 valence-3
    nodes of notes/12 (+ 30 deg). Figures: ResultsFig/explicit/70_three_methods_fig12.png, 71_..._fig14.png,
    72_..._fig15.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import study_three_beam_sweep as sw
    print(HEAD3)
    res = {}
    for nm, E in paper_nodes():
        out, R, E, r = metrics3(nm, E, np.ones(len(E)))
        res[nm] = (out, R, E, r)
        for k in METHODS3:
            print(row3(nm, k, out[k]), flush=True)
    nodes = [nm for nm, _ in paper_nodes()]
    # Fig. 12 (7 rows): graph, then control mesh + surface for each method
    fig = plt.figure(figsize=(18, 21))
    for c, nm in enumerate(nodes):
        out, R, E, r = res[nm]
        ax = fig.add_subplot(7, 6, 1 + c, projection='3d')
        for e in E:
            ax.plot(*np.c_[np.zeros(3), R * e], 'k', lw=0.6)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R); ax.set_box_aspect((1, 1, 1))
        ax.view_init(25, 35); ax.set_axis_off(); ax.set_title(nm, fontsize=10)
        for j, k in enumerate(METHODS3):
            o = out[k]
            if 'fail' in o:
                continue
            Vc, Fc = o['ctrl']
            Vc = np.asarray(Vc)
            mc = trimesh.Trimesh(Vc, poly_tris(Fc), process=False)
            keep = np.linalg.norm(Vc[poly_tris(Fc)].mean(1), axis=1) < 1.15 * R
            mcc = trimesh.Trimesh(Vc, poly_tris(Fc)[keep], process=False)
            ed = poly_edges(Fc)
            ed = ed[(np.linalg.norm(Vc[ed].mean(1), axis=1) < 1.15 * R)]
            render_z(fig.add_subplot(7, 6, 1 + c + 6 * (1 + 2 * j)), mcc, np.zeros(len(Vc)), 1.0, f'{k} control mesh',
                     1.1 * R, n_px=340, n_pts=500_000, grey=True, edges=ed)
            m = o['m']
            keep = np.linalg.norm(m.vertices[m.faces].mean(1), axis=1) < R
            render_z(fig.add_subplot(7, 6, 1 + c + 6 * (2 + 2 * j)), trimesh.Trimesh(m.vertices, m.faces[keep], process=False),
                     np.zeros(len(m.vertices)), 1.0, f'{k} surface', 1.1 * R, n_px=340, n_pts=1_400_000, grey=True)
    fig.savefig('ResultsFig/explicit/70_three_methods_fig12.png', dpi=95, bbox_inches='tight')
    plt.close(fig)
    # Fig. 14: mean curvature, red (+) -> green (0) -> blue (-); one symmetric range per node for all three
    fig = plt.figure(figsize=(18, 10))
    for c, nm in enumerate(nodes):
        out, R, E, r = res[nm]
        Hs = {}
        for k in METHODS3:
            if 'fail' in out[k]:
                continue
            m = out[k]['m']
            H = mean_curvature(m)
            inb = np.linalg.norm(m.vertices, axis=1) < R
            Hs[k] = (m, H, inb)
        hl = max(np.quantile(np.abs(H[inb]), 0.99) for m, H, inb in Hs.values())
        for j, k in enumerate(METHODS3):
            if k not in Hs:
                continue
            m, H, inb = Hs[k]
            keep = np.linalg.norm(m.vertices[m.faces].mean(1), axis=1) < R
            render_z(fig.add_subplot(3, 6, 1 + c + 6 * j), trimesh.Trimesh(m.vertices, m.faces[keep], process=False), H, 1.0,
                     f'{k}: {nm} (H r, +-{hl:.1f})', 1.1 * R, n_px=340, n_pts=1_400_000, cmap='jet', hlim=hl)
    fig.savefig('ResultsFig/explicit/71_three_methods_fig14.png', dpi=95, bbox_inches='tight')
    plt.close(fig)
    # Fig. 15: deviation histograms in the common node ball
    fig, axs = plt.subplots(3, 6, figsize=(20, 9))
    for c, nm in enumerate(nodes):
        out, R, E, r = res[nm]
        xmax = max(out[k]['dev'].max() for k in METHODS3 if 'dev' in out[k])
        for j, k in enumerate(METHODS3):
            ax = axs[j, c]
            if 'dev' not in out[k]:
                continue
            dv = out[k]['dev']
            cnt, e_ = np.histogram(dv, bins=120, range=(0, xmax))
            ax.bar(e_[:-1], cnt, width=np.diff(e_), align='edge', color=cm.jet(e_[:-1] / xmax))
            ax.set_title(f'{k}: {nm} (mean {dv.mean():.3f})', fontsize=9)
            ax.set_xlabel('Deviation from Boolean node (mm)', fontsize=8); ax.set_ylabel('Counts', fontsize=8)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig('ResultsFig/explicit/72_three_methods_fig15.png', dpi=95)
    plt.close(fig)
    # valence-3 sweep
    for an, ang in dict(sw.ANGLES, acute_30_120_110=(30, 120, 110)).items():
        for rn, rad in sw.RADII.items():
            nm = f'{an}_{rn}'
            try:
                out, R, E, r = metrics3(nm, sw.from_angles(*ang), np.array(rad))
            except Exception as ex:
                print(f'| {nm} | all | FAIL: {type(ex).__name__}: {str(ex)[:80]} | | | | | | | |', flush=True)
                continue
            for k in METHODS3:
                print(row3(nm, k, out[k]), flush=True)


def step_fe3():
    """Tetrahedral node FE (as S5) for Savio and SF (R3), equal and graded radii; FHS from S5."""
    print('| node | method | K (h = r_min/4) | K (r_min/6) | K (r_min/8) | tets (finest) |')
    print('|---|---|---|---|---|---|')
    for nm, r in (('tetra_r080', np.full(4, 0.8)), ('tetra_graded_100_080_060_040', np.array([1.0, 0.8, 0.6, 0.4]))):
        E = tetra()
        out, R = build3(E, r, np.full(4, 8.0))
        L = R + 3 * r
        out, R = build3(E, r, L)
        for k in ('Savio', 'SF', 'FHS'):
            stl = f'{OUT}/fe3_{nm}_{k}.stl'
            out[k]['mesh'].export(stl)
            res = []
            for h in (4, 6, 8):
                try:
                    q = fe_peak(stl, E, r, L, R, r.min() / h)
                    res.append(f"{q['K']:.3f}")
                    nel = q['nel']
                except Exception as ex:
                    res.append(f'fail ({type(ex).__name__}: {str(ex)[:40]})')
                    nel = '-'
            print(f'| {nm} | {k} | ' + ' | '.join(res) + f' | {nel} |', flush=True)


def step_s3():
    """S3: regular and irregular nodes, r = 1, all methods."""
    print(HEAD)
    for nm, E in [('regular-6', regular(6)), ('regular-12', regular(12)), ('regular-20', regular(20)),
                  ('irregular-6', irregular(6)), ('irregular-12', irregular(12)), ('irregular-20', irregular(20))]:
        out, R = run_case(nm, E, np.ones(len(E)))
        for k in ('Boolean', 'Hull', 'SF', 'FHS'):
            print(row(nm, k, out[k]), flush=True)



# --------------------------------------------------------------------------------------------- the other methods
def fhs_node(E, r, L, dt=0.005):
    """FHS single node (notes/13): trim sphere, hull of node + core rings, Loop x3, anchored thin-plate fairing."""
    node = np.zeros(3)
    R_s, _ = nj.trim_radius(E, r, L, dt=dt)
    sm, ctrl = nj.hull_subdivision_node(node, E, r, L, R_s, lead=1.0, core=True)
    fm, _ = nj.fair_node_anchored(nj.to_trimesh(sm), node, E, r, nj.ring_offsets(E, r, R_s, core=True))
    fhs_node.ctrl = ctrl
    return fm, R_s


def mean_curvature(m):
    """Vertex mean curvature H (cotangent Laplacian, + convex) and outward vertex normals."""
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    Lc = nj.cotan_laplacian(V, F)
    return 0.5 * ((Lc @ V) * m.vertex_normals).sum(1)


def node_region(m, E, r, reach):
    """Vertex mask of the node region: closer to the node than reach_i along every strut it is near."""
    V = np.asarray(m.vertices)
    keep = np.ones(len(V), bool)
    for e, ri, R in zip(E, r, reach):
        ax = V @ e
        on = (ax > 0) & (np.linalg.norm(V - np.outer(ax, e), axis=1) < 1.5 * ri)
        keep &= ~(on & (ax > R))
    return keep


def concave_radius(m, mask):
    """Tightest concave radius (edge-based normal curvature, as notes/12) over the vertices in mask."""
    n = m.vertex_normals
    Ed = m.edges_unique
    d = m.vertices[Ed[:, 1]] - m.vertices[Ed[:, 0]]
    L2 = (d * d).sum(1)
    kmin = np.full(len(m.vertices), np.inf)
    np.minimum.at(kmin, Ed[:, 0], -2 * (n[Ed[:, 0]] * d).sum(1) / L2)
    np.minimum.at(kmin, Ed[:, 1], 2 * (n[Ed[:, 1]] * d).sum(1) / L2)
    c = -kmin[mask]
    return 1.0 / c.max() if (c > 0).any() else np.inf


def render_z(ax, m, H, r, title, lim, n_px=420, n_pts=2_500_000, view=(25, 35), grey=False, edges=False, cmap=None,
             hlim=1.5):
    """Opaque render with a z-buffer (dense surface points splatted to pixels, nearest wins): colour = mean
    curvature H r, Lambert shading. matplotlib's 3D polygons have no depth buffer."""
    import matplotlib.cm as cm
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    el, az = np.radians(view)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])        # towards the viewer
    up = np.array([0, 0, 1.0]) - d[2] * d
    up /= np.linalg.norm(up)
    rt = np.cross(up, d)
    A = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    rng = np.random.default_rng(0)
    fi = rng.choice(len(F), n_pts, p=A / A.sum())
    u, v = rng.random(n_pts), rng.random(n_pts)
    fl = u + v > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    T = V[F[fi]]
    P = T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])
    x, y, z = P @ rt, P @ up, P @ d
    ix = ((x / lim + 1) / 2 * n_px).astype(int)
    iy = ((1 - (y / lim + 1) / 2) * n_px).astype(int)
    ok = (ix >= 0) & (ix < n_px) & (iy >= 0) & (iy < n_px)
    pix = iy[ok] * n_px + ix[ok]
    order = np.lexsort((-z[ok], pix))                              # per pixel, nearest (largest z) first
    pix_s = pix[order]
    first = np.r_[True, pix_s[1:] != pix_s[:-1]]
    sel = np.where(ok)[0][order[first]]
    Hf = np.clip(H[F[fi[sel]]].mean(1) * r, -hlim, hlim)
    cmf = getattr(cm, cmap) if cmap else cm.jet_r
    col = np.full((len(sel), 3), 0.82) if grey else cmf((Hf + hlim) / (2 * hlim))[:, :3]
    fn = m.face_normals[fi[sel]]
    light = _unit_vec(d + 0.6 * up + 0.3 * rt)
    lam = np.abs(fn @ light)
    col *= (0.35 + 0.65 * lam)[:, None]
    if grey:                                                        # specular highlight shows curvature like Fig. 12
        h = _unit_vec(light + d)
        col += 0.35 * (np.abs(fn @ h) ** 40)[:, None]
        col = np.clip(col, 0, 1)
    img = np.ones((n_px * n_px, 3))
    img[pix_s[first]] = col
    if edges:                                                       # visible mesh edges (control meshes)
        zb = np.full(n_px * n_px, -np.inf)
        zb[pix_s[first]] = z[sel]
        Ed = np.asarray(edges) if not isinstance(edges, bool) else m.edges_unique
        k = 40
        s_ = np.linspace(0, 1, k)
        Q = (V[Ed[:, 0]][:, None, :] * (1 - s_)[None, :, None] + V[Ed[:, 1]][:, None, :] * s_[None, :, None]).reshape(-1, 3)
        qx, qy, qz = Q @ rt, Q @ up, Q @ d
        jx = ((qx / lim + 1) / 2 * n_px).astype(int)
        jy = ((1 - (qy / lim + 1) / 2) * n_px).astype(int)
        okq = (jx >= 0) & (jx < n_px) & (jy >= 0) & (jy < n_px)
        pq = jy[okq] * n_px + jx[okq]
        vis = qz[okq] >= zb[pq] - 0.015 * lim
        img[pq[vis]] = 0.15
    ax.imshow(img.reshape(n_px, n_px, 3))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def _unit_vec(v):
    return v / np.linalg.norm(v)


def render(ax, m, H, r, title, lim=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    Hf = np.clip(H[F].mean(1) * r, -1.5, 1.5)
    col = cm.jet_r((Hf + 1.5) / 3.0)
    fn = m.face_normals
    shade = 0.55 + 0.45 * np.clip(fn @ np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75]), 0, 1)
    col[:, :3] *= shade[:, None]
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolor='none'))
    lim = lim or np.abs(V).max()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.set_title(title, fontsize=8)


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    {'s1s2': step_s1s2, 's3': step_s3, 'hull': step_hull, 's4': step_s4, 's5': step_s5, 's5b': step_s5b, 'sfcal': step_sfcal, 'paperfigs': step_paperfigs, 'all3': step_all3, 'fe3': step_fe3}[sys.argv[1] if len(sys.argv) > 1 else 's1s2']()
