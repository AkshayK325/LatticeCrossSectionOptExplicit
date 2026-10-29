"""Abaqus/Explicit reference cases for the helmet-patch development (shell + beam-lattice liner), notes/19, notes/0_1.

    PYTHONPATH=src python abaqus/helmet_cases.py [HS1a HS1b HS1c HS2a]    # writes abaqus/inp/<name>.inp, runs geometry checks

Units mm, N, s, tonne, MPa. Stack, from the fixed anvil upward (the standards drop head + helmet onto a fixed anvil; in the
anvil frame this is the same as the head hitting the stack):
    rigid anvil sphere R 48  |  SHELL (S4R, reference surface = HEAD-side face, thickness extends toward the anvil: OFFSET 0.5)
    |  lattice liner  |  rigid HEAD (free mass, initial speed v0)
Normals of the shell elements point toward the head; the *SHELL SECTION OFFSET=0.5 puts the reference (meshed) surface on the
positive-normal face, so the lattice end nodes (z = 0) lie ON the reference surface and the thickness hangs below it.

Cases (what each one decides is in notes/0_1_GlobalRunPolicy.md):
  HS1a/b/c  shell bowl (spherical cap, R 80) alone between a rigid free spherical head (R 80) and a rigid fixed anvil sphere
            (R 48): shell element + shell mesh size (4 / 2 / 1 mm) + sphere contacts + free head mass. Elastic, 1 m/s. Cheap.
  HS2a      the F2a solid (ONE C4 unit cell, FHS joints, r 0.8, h 0.4, linear E 40 MPa; its Abaqus result F2a exists) on a flat
            shell plate 30 x 30 x 2 mm, tied; rigid flat head as F2a; hemispherical anvil beneath the plate centre. F2a minus
            this = effect of the compliant shell + tie + anvil. Same energy / speed / mesh as F2a.
HS3 (curved lattice between two spheres) is NOT generated: it needs the sphere-trimmed FHS lattice (see notes/19), only
if HS1/HS2 pass and curvature is judged to matter.
Assumptions to flag in any report: head radius 80 mm, shell E 2300 MPa / nu 0.38 / rho 1.2e-9 t/mm^3 / t 2 mm (generic PC/ABS-class
shell, not a measured material), frictionless contacts (isolates the mechanics; friction is a later, separate question).
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases as C                                              # noqa: E402

PC = dict(E=2300.0, nu=0.38, rho=1.2e-9)                       # assumed shell material
T_SH, R_HEAD, R_ANV, GAP = 2.0, 80.0, 48.0, 0.05               # shell thickness, head radius, anvil radius, initial gaps [mm]
MU = 0.0


def _grid(half, h, node0, project, flip=False):
    """Square grid of (2n+1)^2 points over [-half, half]^2 (even number of cells, centre node present), quads CCW seen from
    +z (flip: reversed). project(x, y) -> (x, y, z). Returns node rows [(id, x, y, z)], quad rows [(i1..i4)], centre id."""
    n = int(round(2 * half / h))
    n += n % 2
    xs = np.linspace(-half, half, n + 1)
    nid = lambda i, j: node0 + i * (n + 1) + j
    nodes = [(nid(i, j), *project(xs[i], xs[j])) for i in range(n + 1) for j in range(n + 1)]
    quads = []
    for i in range(n):
        for j in range(n):
            q = (nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1))
            quads.append(q[::-1] if flip else q)
    return nodes, quads, nid(n // 2, n // 2)


def _sphere_proj(R, cz, sign):
    """Gnomonic projection of the tangent plane at the pole onto the sphere of radius R centred (0,0,cz); sign=-1: bowl (pole
    below the centre), +1: dome. Distances on the plane are ~ distances on the sphere near the pole."""
    def p(x, y):
        a, b = x / R, y / R
        d = np.array([a, b, sign]) / np.sqrt(a * a + b * b + 1.0)
        return tuple(float(v) for v in (R * d + np.array([0.0, 0.0, cz])))
    return p


def _rows(rows):
    return '\n'.join(', '.join(f'{v:.10g}' if isinstance(v, (float, np.floating)) else str(v) for v in r) for r in rows)


def _shell_block(nodes, quads, e0, elset='SHELL', offset=0.5):
    return (f"*NODE\n{_rows(nodes)}\n*ELEMENT, TYPE=S4R, ELSET={elset}\n{_rows([(e0 + k, *q) for k, q in enumerate(quads)])}\n"
            f"*SHELL SECTION, ELSET={elset}, MATERIAL=PCSHELL, OFFSET={offset:g}\n{T_SH:g},\n"
            f"*MATERIAL, NAME=PCSHELL\n*DENSITY\n{PC['rho']:.10g},\n*ELASTIC\n{PC['E']:.10g}, {PC['nu']:.10g}\n")


def _rigid_block(nodes, quads, e0, elset, ref, ref_xyz, mass=None):
    s = (f"*NODE\n{_rows([(ref, *ref_xyz)])}\n*NODE\n{_rows(nodes)}\n"
         f"*ELEMENT, TYPE=R3D4, ELSET={elset}\n{_rows([(e0 + k, *q) for k, q in enumerate(quads)])}\n"
         f"*NSET, NSET={elset}REF\n{ref}\n*RIGID BODY, REF NODE={ref}, ELSET={elset}\n")
    if mass is not None:
        s += f"*ELEMENT, TYPE=MASS, ELSET={elset}M\n{e0 + len(quads)}, {ref}\n*MASS, ELSET={elset}M\n{mass:.10g},\n"
    return s


def _geometry_hs1(h, mid=False):
    """HS1 geometry; returns the model text parts and a dict of clearances for the sanity check."""
    Rs = R_HEAD + GAP + (T_SH / 2 if mid else 0.0)   # shell reference surface radius (mid: the mid-surface); bowl pole at origin
    cz = Rs                                    # sphere centre above the pole
    half = 34.0
    sh_n, sh_q, sh_c = _grid(half, h, 1, _sphere_proj(Rs, cz, -1))
    ne0 = 1
    anv_z0 = -(T_SH / 2 if mid else T_SH) - GAP   # anvil pole
    an_n, an_q, _ = _grid(36.0, 1.5, 100000, _sphere_proj(R_ANV, anv_z0 - R_ANV, +1))
    hd_n, hd_q, _ = _grid(36.0, 1.5, 200000, _sphere_proj(R_HEAD, cz, -1), flip=True)
    return dict(shell=(sh_n, sh_q, sh_c), anvil=(an_n, an_q), head=(hd_n, hd_q), cz=cz, anv_z0=anv_z0, half=half)


def hs1_inp(c):
    g = _geometry_hs1(c['h'], c.get('mid', False))
    sh_n, sh_q, sh_c = g['shell']
    an_n, an_q = g['anvil']
    hd_n, hd_q = g['head']
    M = c['mass']
    t_end = c['T']
    txt = f"""*HEADING
{c['title']}
** Generated by abaqus/helmet_cases.py (notes/19). Units mm, N, s, tonne, MPa. Shell bowl: spherical cap, reference surface radius
** {R_HEAD + GAP:g} (head side), t {T_SH:g} mm OFFSET 0.5 (thickness toward the anvil), S4R h {c['h']:g} mm, {len(sh_q)} elements, centre node {sh_c}.
** Rigid head sphere R {R_HEAD:g} (free in z, mass {M:.4g} t at v0 {c['v0']:g} mm/s), rigid fixed anvil sphere R {R_ANV:g}, gaps {GAP:g} mm.
{_shell_block(sh_n, sh_q, 1, offset=0.0 if c.get('mid') else 0.5)}{_rigid_block(an_n, an_q, 100000, 'ANVIL', 199990, (0.0, 0.0, g['anv_z0'] - R_ANV))}{_rigid_block(hd_n, hd_q, 200000, 'HEAD', 299990, (0.0, 0.0, g['cz']), mass=M)}*NSET, NSET=SHC
{sh_c}
*SURFACE INTERACTION, NAME=SI
*FRICTION
{MU:g},
*BOUNDARY
ANVILREF, 1, 6
HEADREF, 1, 2
HEADREF, 4, 6
*INITIAL CONDITIONS, TYPE=VELOCITY
HEADREF, 3, {-c['v0']:.10g}
*STEP, NAME=DROP, NLGEOM=YES
*DYNAMIC, EXPLICIT
, {t_end:.10g}
*CONTACT
*CONTACT INCLUSIONS, ALL EXTERIOR
*CONTACT PROPERTY ASSIGNMENT
, , SI
*OUTPUT, FIELD, NUMBER INTERVAL=10
*NODE OUTPUT
U
*OUTPUT, HISTORY, TIME INTERVAL={c['out_dt']:.10g}
*NODE OUTPUT, NSET=HEADREF
U3, V3, RF3
*NODE OUTPUT, NSET=ANVILREF
RF3
*NODE OUTPUT, NSET=SHC
U3
*ENERGY OUTPUT
ALLKE, ALLIE, ALLSE, ALLWK, ETOTAL, ALLAE, ALLVD, ALLCD
*END STEP
"""
    return txt


def hs2_inp():
    """HS2a: F2a (one C4 FHS cell, linear E 40) on a tied flat shell plate, hemispherical anvil, rigid free flat head."""
    spec = C.F2_CASES['F2a']
    g = C.b2_mesh(spec)
    P = g['P']
    txt = C.b2_inp(spec)
    cx, cy = (float(v) for v in g['center'])
    half, h = 15.0, 2.0
    nodes0 = len(P) + 10
    sh_n, sh_q, sh_c = _grid(half, h, nodes0, lambda x, y: (cx + x, cy + y, 0.0))
    e0 = len(g['T']) + 10
    anv_z0 = -T_SH - GAP
    an_n, an_q, _ = _grid(30.0, 1.5, 500000, lambda x, y: (lambda p: (cx + p[0], cy + p[1], p[2]))(_sphere_proj(R_ANV, anv_z0 - R_ANV, +1)(x, y)))
    shell = _shell_block(sh_n, sh_q, e0)
    anv = _rigid_block(an_n, an_q, 600000, 'ANVIL', 999990, (cx, cy, anv_z0 - R_ANV))
    extra_model = f"""** HS2a: F2a + flat shell plate {2 * half:g} x {2 * half:g} x {T_SH:g} mm (S4R h {h:g}), reference surface z = 0 = the lattice end
** faces, normal +z (toward the lattice), OFFSET 0.5 -> thickness below; node-based tie of ALL bottom-face nodes to the shell surface;
** rigid hemispherical anvil R {R_ANV:g} (pole {GAP:g} mm below the shell), fixed. Head (REF + TOP faces) as F2a: only v0 and rotations as F2a, x-y fixed.
{shell}{anv}*NSET, NSET=SHC
{sh_c}
"""
    tie = """*SURFACE, TYPE=NODE, NAME=LATBOT_S
BOT,
*SURFACE, TYPE=ELEMENT, NAME=SHELLTOP
SHELL, SPOS
*TIE, NAME=LATTIE, ADJUST=NO, POSITION TOLERANCE=0.05
LATBOT_S, SHELLTOP
"""
    def once(s, old, new):
        assert s.count(old) == 1, (old, s.count(old))
        return s.replace(old, new)
    txt = once(txt, '*NSET, NSET=BOT\n', extra_model + '*NSET, NSET=BOT\n')
    txt = once(txt, '*SURFACE INTERACTION, NAME=CELLINT\n', tie + '*SURFACE INTERACTION, NAME=CELLINT\n')
    txt = once(txt, '*BOUNDARY\nBOT, 1, 3\nREF, 4, 6\n', '*BOUNDARY\nREF, 1, 2\nREF, 4, 6\nANVILREF, 1, 6\n')
    txt = once(txt, '*CONTACT INCLUSIONS, ALL EXTERIOR\n', '*CONTACT INCLUSIONS, ALL EXTERIOR\n*CONTACT EXCLUSIONS\nLATBOT_S, SHELLTOP\n')
    txt = once(txt, '*NODE OUTPUT, NSET=BOT\nRF3\n', '*NODE OUTPUT, NSET=ANVILREF\nRF3\n*NODE OUTPUT, NSET=SHC\nU3\n')
    txt = once(txt, '*HEADING\n', '*HEADING\nHS2a: F2a (one C4 FHS cell, r 0.8, linear E 40 MPa, 0.083 J at 6 m/s) tied to a flat PC-class shell plate on a hemispherical anvil. ')
    return txt


HS1 = dict(name='HS1a_shell_bowl_h4', h=4.0, mass=4.0e-5, v0=1000.0, T=1.5e-3, out_dt=5e-6,
           title='HS1a: spherical shell bowl (R 80, t 2, S4R h 4 mm) between a free rigid head sphere (40 g, 1 m/s) and a fixed rigid anvil sphere R 48; elastic, frictionless')
HS1d = dict(HS1, name='HS1d_shell_bowl_h2_midplane', h=2.0, T=2.5e-3, mid=True,
            title='HS1d: as HS1b but shell MID-surface geometry (OFFSET 0, faces at +-t/2, gaps 0.05 to the faces), T 2.5 ms: removes the section-offset contact ambiguity found in HS1a-c')
HS_CASES = {'HS1a': HS1,
            'HS1b': dict(HS1, name='HS1b_shell_bowl_h2', h=2.0, title=HS1['title'].replace('HS1a', 'HS1b').replace('h 4 mm', 'h 2 mm')),
            'HS1c': dict(HS1, name='HS1c_shell_bowl_h1', h=1.0, title=HS1['title'].replace('HS1a', 'HS1c').replace('h 4 mm', 'h 1 mm')),
            'HS1d': HS1d,
            'HS2a': dict(name='HS2a_F2a_cell_on_shell_hemisphere')}


def check(case):
    """Geometry sanity of the generated model (no Abaqus here): normals, clearances, tie coverage."""
    if case.startswith('HS1'):
        c = HS_CASES[case]
        g = _geometry_hs1(c['h'], c.get('mid', False))
        nodes = {i: np.array(p) for i, *p in g['shell'][0]}
        q = g['shell'][1]
        n = np.cross(nodes[q[0][1]] - nodes[q[0][0]], nodes[q[0][3]] - nodes[q[0][0]])
        centre = np.array([0.0, 0.0, g['cz']])
        mid = np.mean([nodes[i] for i in q[len(q) // 2]], axis=0)
        tt = T_SH / 2 if c.get('mid') else T_SH
        print(f'{case}: {len(q)} shell elements; normal at centre quad {np.round(n / np.linalg.norm(n), 3)} (must be +z, toward the head); '
              f'apex z {min(p[2] for p in nodes.values()):.3f}; node radius from sphere centre: '
              f'{np.ptp([np.linalg.norm(p - centre) for p in nodes.values()]):.2e} spread; anvil pole z {g["anv_z0"]:.3f} '
              f'(shell outer apex {-tt:.3f}, gap {-tt - g["anv_z0"]:.3f})')
    else:
        spec = C.F2_CASES['F2a']
        g = C.b2_mesh(spec)
        P = g['P']
        bot = P[g['bot']]
        cx, cy = g['center']
        print(f'{case}: lattice bottom-face nodes {len(bot)}; z range [{bot[:, 2].min():.2e}, {bot[:, 2].max():.2e}] (must be 0); '
              f'max |x-cx|,|y-cy| = {np.abs(bot[:, 0] - cx).max():.2f}, {np.abs(bot[:, 1] - cy).max():.2f} (shell half-width 15)')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT.parent / 'tests'))
    import runStatus as rs                                      # repo rule: every job reports to run_status.log
    names = sys.argv[1:] or list(HS_CASES)
    rs.start('write helmet Abaqus inputs ' + ' '.join(names))
    out = ROOT / 'inp'
    out.mkdir(exist_ok=True)
    for ci, nm in enumerate(names):
        rs.update(nm, ci, len(names))
        c = HS_CASES[nm]
        p = out / f"{c['name']}.inp"
        p.write_text(hs2_inp() if nm == 'HS2a' else hs1_inp(c))
        check(nm)
        print('wrote', p)
