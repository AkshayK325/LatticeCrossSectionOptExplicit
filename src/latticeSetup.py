"""optBeam.ipynb exampleCase 3 lattice (cell 19) + its symmetry groups, and the drop-test rig built on it.

Design flow (as in the notebook / OptimizeFrame, varType 'Element'):
    x (n_rows, ncol) --apply_symmetry--> per-beam values (n_beams, ncol) --[beam_of_elem]--> per-element values
"""
import math

import numpy as np
import torch
import torch.nn.functional as F_

from mesher import MeshFrame

SYM_TYPES = ('Independent', 'Unitcell', 'OneValue', 'Midplane')


def example3(elemSize=3.0):
    """exampleCase '3fz' lattice with the notebook cell-19 params. Returns a dict with base and refined mesh."""
    from examples import getExample
    H = 0.5 * 25.4                                    # 0.5 inch per row
    params = {'base': H, 'height': H, 'phi': 0.0, 'delta': 0.0, 'nx': 3, 'ny': 3, 'nz': 1,
              'Name': 'C4', 'Shape': 'Square'}
    _, X, conn, _, radiiNodIndex, radiiElemIndex = getExample('3fz', params)
    Xr, connr = MeshFrame().refineFrameMeshElemSize(X, conn, elemSize)
    L = np.linalg.norm(X[conn[:, 1]] - X[conn[:, 0]], axis=1)
    n_per_beam = np.rint(L / elemSize).astype(int)    # same rule as refineFrameMeshElemSize
    return {'X': X, 'conn': conn, 'Xr': Xr, 'connr': connr, 'elemSize': elemSize,
            'beam_of_elem': np.repeat(np.arange(len(conn)), n_per_beam),
            'radiiElemIndex': np.asarray(radiiElemIndex), 'radiiNodIndex': np.asarray(radiiNodIndex),
            'numUnitLatElem': int(max(radiiElemIndex)) + 1, 'params': params}


def symmetry_groups(symType, lat):
    """symArray exactly as notebook cell 19 builds it for exampleCase 3."""
    X, conn, nU = lat['X'], lat['conn'], lat['numUnitLatElem']
    if symType == 'Unitcell':
        return np.arange(nU).tolist()
    if symType == 'OneValue':
        return [np.arange(nU)]
    if symType == 'Independent':
        return np.arange(conn.shape[0]).tolist()
    if symType == 'Midplane':
        mid_lat = np.mean(X[0:nU, :], axis=0)
        mids = (X[conn[0:nU, 0], :] + X[conn[0:nU, 1], :]) / 2
        dist = np.round((mids[:, -1] - mid_lat[-1]) / np.linalg.norm(mids[:, -1] - mid_lat[-1]), decimals=5)
        dist = (dist > 0.0) * 1.0 + (dist < 0.0) * 2.0 + (dist == 0.0) * 3.0
        uniq, inv = np.unique(dist, return_inverse=True)
        return [np.where(inv == i)[0] for i in range(len(uniq))]
    raise ValueError(f'symType must be one of {SYM_TYPES}')


def n_design_rows(symType, lat):
    return lat['conn'].shape[0] if symType == 'Independent' else lat['numUnitLatElem']


def apply_symmetry(x, symArray, lat):
    """Group means, then unit-cell -> all beams (OptimizeFrame.applySymm, varType 'Element')."""
    var = x.clone()
    for g in symArray:
        var[g] = x[g].reshape(-1, x.shape[1]).mean(0, keepdim=True)
    if var.shape[0] < len(lat['radiiElemIndex']):
        var = var[torch.as_tensor(lat['radiiElemIndex'])]
    return var


def element_values(x, symType, lat):
    """Design rows -> per-element values (n_elements_refined, ncol)."""
    return apply_symmetry(x, symmetry_groups(symType, lat), lat)[torch.as_tensor(lat['beam_of_elem'])]


def circle_sections(r, nu=None):
    """Per-element radius (ne,) -> frameFE3-layout A (ne,3) = [A, kappa A, kappa A], I (ne,3) = [J, Iy, Iz].
    nu given: Timoshenko shear coefficient of a solid circle, kappa = 6(1+nu)/(7+6nu) (Cowper 1966; 0.899 for
    nu = 0.49). nu None: kappa = 1 (legacy frameFE3 convention)."""
    kappa = 1.0 if nu is None else 6 * (1 + nu) / (7 + 6 * nu)
    A = (math.pi * r**2)[:, None] * torch.tensor([1.0, kappa, kappa], dtype=r.dtype)
    I = torch.stack((math.pi * r**4 / 2, math.pi * r**4 / 4, math.pi * r**4 / 4), 1)
    return A, I


# ------------------------------------------------------------------------------------------------
# Drop rig: lattice bonded to a fixed rigid bottom plate, rigid top plate bonded to the top nodes,
# separate rigid impactor (1 translational DOF) falling onto the top plate. Contact acts only between
# impactor and top plate (scalar gap law from contact.py).

def drop_rig(lat, r_elem, law, mat, energy_J=0.75, v0_ms=6.0, plate_mass=7.8e-6, gap0=0.3, courant=0.7,
             start='gap', lattice_contact=True, contact_model='line', fibre=True, mu=0.5, freeze=None,
             crush=False, contact_ref_radius=None, contact_exclusion='smooth'):
    """law: a gap law, or a dict of contact.calibrate() kwargs (lets 'crit_damped' get the reduced mass).
    r_elem (ne,) radii [mm]. plate_mass [t] (default 2 mm aluminium 38x38: 7.8 g). gap0 [mm] initial
    impactor-plate gap.
    start='gap':       impactor (M = 2E/v0^2) at v0 hits the resting plate -> rigid-rigid collision: a force
                       spike and plate rattling are physical here, whatever the contact law.
    start='co_moving': impactor and plate start together at v0 (the impactor-plate collision is taken as
                       already completed, plastic; i.e. the plate is the striker face), small gap, zero
                       relative speed. Total KE = E: M = 2E/v0^2 - plate mass. The contact force then
                       builds continuously as the lattice decelerates the plate, and releases continuously
                       on rebound: no collision event, no jump.
    start='merged':    impactor rigidly bonded to the plate (one rigid body, no impactor contact; lattice contact still active if requested): zero jump by
                       construction, but the impactor can never separate on rebound.
    lattice_contact: beam-to-beam contact and node-to-plate Hertz contact of every non-bonded node with both
                       plates (True), or only one of them ('self' / 'plates', for diagnostics). contact_model 'line' (default): contact.LineContact, line-contact density
                       integrated along the beams; 'capsule': contact.SelfContact (one Hertz point per beam pair).
    mu:                Coulomb friction coefficient for beam-beam and beam-plate contact (line model; 0 = off).
    fibre:             large-strain (fibre-integrated Yeoh) bending and torsion in the section law.
    contact_ref_radius: LineContact exclusion_radius: joint/overlap exclusions computed with this radius for all
                       elements (design-independent excluded set; use the upper radius bound in optimisation).
    contact_exclusion: 'smooth' (default; joint/overlap exclusion as a C1 weight of the radii, so J(x) is
                       continuous and differentiable) or 'hard' (excluded set; J jumps when a design change moves
                       quadrature points across the threshold).
    crush:             constant-velocity crush (standard SEA test): impactor mass 1 t, so the top plate keeps
                       ~v0 whatever the lattice does (energy_J ignored); stop at the target stroke via T.
    freeze:            a nominal rig: reuse its design-dependent constants (impactor mass and contact law,
                       contact smoothing eps, joint-exclusion set, plate-contact node sets) so that a perturbed
                       design differs from the nominal ONLY through r_elem (needed for AD vs FD and adjoints)."""
    from explicitFE import Frame, lumped_mass, adaptive_dt
    from contact import law_tangent, contact_dt_cap, potential_force
    Xr, connr = lat['Xr'], lat['connr']
    zt = Xr[:, 2].max()
    top = np.nonzero(np.isclose(Xr[:, 2], zt))[0]
    bot = np.nonzero(np.isclose(Xr[:, 2], Xr[:, 2].min()))[0]
    imp = len(Xr)                                                # impactor = extra node, no elements
    X = np.vstack((Xr, [[Xr[:, 0].mean(), Xr[:, 1].mean(), zt + gap0]]))
    fr = Frame(X, connr)
    A, I = circle_sections(r_elem, nu=mat['nu'])
    if 'yeoh' in mat:                                            # visco-hyperelastic (EPU 45)
        from explicitFE import ViscoHyperLaw
        beam_law = ViscoHyperLaw(fr, A, I, mat, fibre=fibre)   # fibre: large-strain bending + torsion
    else:
        E, G = mat['E'], mat['G']
        beam_law = lambda d: fr.elastic_law(d, A, I, E, G)
    m, j = lumped_mass(fr, A, I, mat['rho'])
    v0 = v0_ms * 1e3
    M = 2 * energy_J * 1e3 / v0**2
    m = m.clone()
    if start in ('co_moving', 'merged'):
        M -= (m[top].sum().item() + plate_mass)          # plate + tied top nodes move with the impactor
    if crush:
        M = 1.0                                          # 1 t at 6 m/s: velocity loss over the crush is negligible
    if freeze is not None:
        M = freeze['M']
    m[imp], j[imp] = M, 1.0                                      # impactor rotations are fixed below
    m[top] += plate_mass / len(top)
    fix_u = torch.zeros(fr.N, 3, dtype=torch.bool)
    fix_u[bot] = True
    fix_u[imp, :2] = True
    fix_r = torch.zeros(fr.N, 3, dtype=torch.bool)
    fix_r[bot] = fix_r[top] = fix_r[imp] = True
    top_t, t0 = torch.as_tensor(top), int(top[0])
    m_plate = m[top].sum().item()
    mu_red = 1 / (1 / M + 1 / m_plate)                           # impactor-plate reduced mass (not friction mu)
    if freeze is not None:
        law = freeze['law']
    if isinstance(law, dict):
        from contact import calibrate
        law = calibrate(mu=mu_red, v_ref=v0, **law)

    def gap(u, v):
        return gap0 + u[imp, 2] - u[t0, 2], v[imp, 2] - v[t0, 2]

    # ---- lattice contact: beam-beam capsules + nodes against both plates (one potential)
    phi_lat, sc = None, None
    if lattice_contact:
        from contact import SelfContact, node_radius, hertz_potential, hertz_tangent
        if contact_model == 'line':
            from contact import LineContact
            sc = LineContact(fr, r_elem, lat['beam_of_elem'], mat['E0'], mat['nu'], beam_conn=lat['conn'],
                             eps=None if freeze is None else freeze['self_contact'].eps,
                             exclusion_radius=contact_ref_radius, exclusion=contact_exclusion)
            if freeze is not None and contact_exclusion == 'hard':      # smooth weights are functions of r_elem
                sc.excluded, sc.n_excluded_points = freeze['self_contact'].excluded, freeze['self_contact'].n_excluded_points
        else:
            sc = SelfContact(fr, r_elem, mat['E0'], mat['nu'], beam_of_elem=lat['beam_of_elem'])
        rn = node_radius(fr, A)
        kp = (4.0 / 3.0) * mat['E0'] / (1 - mat['nu']**2) * torch.sqrt(rn)   # beam on rigid flat (sphere R = r)
        zb = float(Xr[:, 2].min())
        lat_nodes = np.arange(len(Xr))
        bc = torch.as_tensor(np.setdiff1d(lat_nodes, bot))
        tc = torch.as_tensor(np.setdiff1d(lat_nodes, top))
        with torch.no_grad():
            bc = bc[(fr.X[bc, 2] - zb - rn[bc]) > 0]            # drop nodes already touching (none expected)
            tc = tc[(zt - fr.X[tc, 2] - rn[tc]) > 0]
        if freeze is not None:
            bc, tc = freeze['bc'], freeze['tc']
        eps_c = sc.eps

        use_self, use_plates = lattice_contact in (True, 'self'), lattice_contact in (True, 'plates')

        def phi_lat(x):
            Phi_s, g_s = sc.energy(x) if use_self else (x.sum() * 0.0, x[:0, 0])
            gb = x[bc, 2] - zb - rn[bc]
            gt = x[t0, 2] - x[tc, 2] - rn[tc]                   # top-plate face = tied top-node plane
            Phi = Phi_s
            if use_plates:
                Phi = Phi + hertz_potential(gb, kp[bc], eps_c).sum() + hertz_potential(gt, kp[tc], eps_c).sum()
            return Phi, (g_s, gb, gt)

    dt_now = {'h': 1e-6}
    last = {}

    def friction(u, v):
        """Coulomb friction (mu) at beam-beam line-contact points and node-plate contacts: (forces, power>=0)."""
        f = torch.zeros(fr.N, 3, dtype=u.dtype)
        if phi_lat is None or mu == 0 or contact_model != 'line':
            return f, 0.0
        from contact import _line_friction, coulomb
        x = fr.X + u
        f, P = _line_friction(sc, x, v, mu, m, dt_now['h'])
        for nodes, gp, vt in ((bc, x[bc, 2] - zb - rn[bc], v[bc].clone()),
                              (tc, x[t0, 2] - x[tc, 2] - rn[tc], v[tc] - v[t0])):
            dl = eps_c * F_.softplus(-gp / eps_c)
            Fn = kp[nodes] * dl**1.5 * torch.sigmoid(-gp / eps_c)
            on = Fn > 1e-12
            if not on.any():
                continue
            vt[:, 2] = 0.0                                       # in-plane slip on the plate
            Ft = coulomb(Fn[on], vt[on], mu, m[nodes[on]], dt_now['h'])
            f = f.index_add(0, nodes[on], Ft)
            if nodes is tc:
                f[t0] = f[t0] - Ft.sum(0)                        # reaction on the (tied) top plate
            P += -(Ft * vt[on]).sum().item()
        return f, P

    def contact(u, v):
        g, gd = gap(u, v)
        Fc = law(g, gd)
        f = torch.zeros(fr.N, 3, dtype=u.dtype)
        f[imp, 2] = Fc
        f[t0, 2] = -Fc                                           # tie spreads it over the plate
        if phi_lat is not None:
            fp, Phi_l, ex = potential_force(phi_lat, fr.X + u)
            ff, Pf = friction(u, v)
            f = f + fp + ff
            last.update(Phi_lat=Phi_l.item(), ex=ex, Pf=Pf, ff=ff.detach().clone())  # for run_drop bookkeeping
        return f

    def contact_dt(u, v):
        g, gd = gap(u, v)
        kg, kv = law_tangent(law, g, gd)
        w = np.sqrt(max(kg, 0.0) / mu_red)
        zeta = kv / (2 * np.sqrt(max(kg, 1e-30) * mu_red))
        w_eff = w / (np.sqrt(1 + zeta**2) - zeta) if w > 0 else 0.0  # damped central-difference limit
        cap = contact_dt_cap(law, g.item(), gd.item())
        if phi_lat is not None:
            x = fr.X + u
            scale = 2 * eps_c
            w_s, cap_s = sc.dt_terms(x, v, m, scale)
            with torch.no_grad():
                gb = x[bc, 2] - zb - rn[bc]
                gt = x[t0, 2] - x[tc, 2] - rn[tc]
                w_p = max(torch.sqrt(hertz_tangent(gb, kp[bc], eps_c) / m[bc]).max().item(),
                          torch.sqrt(hertz_tangent(gt, kp[tc], eps_c) / m[tc]).max().item())
                vb = v[bc, 2].abs()[gb < 3 * scale]
                vt = (v[t0, 2] - v[tc, 2]).abs()[gt < 3 * scale]
                vmax = max(vb.max().item() if len(vb) else 0.0, vt.max().item() if len(vt) else 0.0)
            cap = min(cap, cap_s, 0.1 * scale / vmax if vmax > 0 else np.inf)
            w_eff = np.sqrt(w_eff**2 + w_s**2 + w_p**2)
        return w_eff, cap

    # beam tangent by power iteration (every 20 steps); contact stiffness by a per-step row-sum bound
    dt_base = adaptive_dt(fr, beam_law, m, j, courant=courant, contact_fn=contact_dt)

    def dt_fn(n, u, R, v=None):
        dt_now['h'] = dt_base(n, u, R, v)                       # friction regularisation needs the step size
        return dt_now['h']
    v_init = torch.zeros(fr.N, 3, dtype=torch.float64)
    v_init[imp, 2] = -v0
    if start in ('co_moving', 'merged'):
        v_init[top_t, 2] = -v0
    tie = top_t
    if start == 'merged':
        tie = torch.cat((top_t, torch.tensor([imp])))
        if phi_lat is None:
            contact = None
        else:                                                    # bonded impactor: lattice contact only

            def contact(u, v):
                fp, Phi_l, ex = potential_force(phi_lat, fr.X + u)
                ff, Pf = friction(u, v)
                last.update(Phi_lat=Phi_l.item(), ex=ex, Pf=Pf, ff=ff.detach().clone())
                return fp + ff
    return dict(fr=fr, A=A, I=I, beam_law=beam_law, law=law, m=m, j=j, M=M, imp=imp, top=top_t, bot=bot,
                fix_u=fix_u, fix_r=fix_r, contact=contact, tie=tie, gap=gap, dt_fn=dt_fn, v_init=v_init, gap0=gap0,
                v0=v0, plate_mass=plate_mass, phi_lat=phi_lat, self_contact=sc, friction=friction, mu=mu,
                last=last, dt_now=dt_now, bc=bc if lattice_contact else None, tc=tc if lattice_contact else None,
                r_elem=r_elem)


def run_drop(rig, T=8e-3, record=True):
    """Simulate; record impactor force, bottom reaction, gap, energies. Returns numpy histories."""
    from explicitSolve import simulate, strain_energy
    from contact import law_potential
    fr, law, imp, bot = rig['fr'], rig['law'], rig['imp'], rig['bot']
    W = {'Wd': 0.0, 'prev': None, 'Wint': 0.0, 'dprev': None}

    def cb(n, t, state, fint):
        u, v, R, w = state
        fu, fth, d, f_loc, fc = fint
        g, gd = rig['gap'](u, v)
        Fc = fc[imp, 2].item() if rig['contact'] is not None else -fu[rig['top'], 2].sum().item() * rig['M'] / (rig['M'] + rig['m'][rig['top']].sum().item())
        Fel = law(g, torch.zeros_like(gd)).item()
        Pd = (Fc - Fel) * -gd.item()                             # power dissipated by contact damping (>= 0)
        if W['prev'] is not None:
            W['Wd'] += 0.5 * (Pd + W['prev'][0]) * (t - W['prev'][1])
        W['prev'] = (Pd, t)
        ke = 0.5 * (rig['m'] * (v**2).sum(1)).sum().item() + 0.5 * (rig['j'] * (w**2).sum(1)).sum().item()
        # internal work W_int = int f_loc . dd (trapezoid) = stored + dissipated for any section law
        if W['dprev'] is not None:
            W['Wint'] += (0.5 * (f_loc + W['dprev'][1]) * (d - W['dprev'][0])).sum().item()
        W['dprev'] = (d.detach().clone(), f_loc.detach().clone())
        bl = rig['beam_law']
        U = bl.stored_long_term(d).item() if hasattr(bl, 'stored_long_term') else strain_energy(d, f_loc).item()
        Phi = law_potential(law, g.item(), rig['gap0'])
        # friction dissipation = -(work of the friction forces the integrator applied), trapezoid in
        # displacement like the internal work (power from half-step velocities does not match Verlet's work)
        ff = rig['last'].get('ff')
        if ff is not None:
            if W.get('fu') is not None:
                W['Wf'] = W.get('Wf', 0.0) - (0.5 * (ff + W['fu'][0]) * (u - W['fu'][1])).sum().item()
            W['fu'] = (ff, u.detach().clone())
        Phi_lat, gs_min, gp_min = 0.0, np.inf, np.inf
        if rig['phi_lat'] is not None and 'ex' in rig['last']:
            gs, gb, gt = rig['last']['ex']
            Phi_lat = rig['last']['Phi_lat']
            gs_min = gs.min().item() if len(gs) else np.inf
            gp_min = min(gb.min().item(), gt.min().item())
        return {'F_imp': Fc, 'a_imp': Fc / rig['M'], 'F_bottom': fu[bot, 2].sum().item(), 'gap': g.item(),
                'z_imp': u[imp, 2].item(), 'v_imp': v[imp, 2].item(), 'z_plate': u[rig['top'][0], 2].item(),
                'KE': ke, 'U': U, 'W_int': W['Wint'], 'Phi': Phi, 'Wd': W['Wd'], 'Phi_lat': Phi_lat,
                'g_self_min': gs_min, 'g_plate_min': gp_min, 'W_fric': W.get('Wf', 0.0),
                'rot_max': d[:, 1:].abs().max().item()}

    with torch.no_grad():
        h = simulate(fr, rig['beam_law'], rig['m'], rig['j'], rig['dt_fn'], None, t_end=T, v0=rig['v_init'],
                     fix_u=rig['fix_u'], fix_r=rig['fix_r'], contact=rig['contact'], tie=rig['tie'], callback=cb)
    h = {k: (np.array(v) if isinstance(v, list) else v) for k, v in h.items()}
    h['E_tot'] = h['KE'] + h['W_int'] + h['Phi'] + h['Wd'] + h['Phi_lat'] + h['W_fric']
    h['E_diss_material'] = h['W_int'] - h['U']                  # viscous loss (+ small branch storage)
    return h


def jump_metrics(F):
    """Force-history smoothness: max per-step jump / peak, chatter = total variation / (2 peak)
    (1.0 for a single clean pulse), number of separate contact episodes (F > 2% of peak)."""
    F = np.asarray(F)
    pk = np.abs(F).max() + 1e-30
    on = F > 0.02 * pk
    return {'peak': pk, 'max_step_jump': np.abs(np.diff(F)).max() / pk,
            'chatter': np.abs(np.diff(F)).sum() / (2 * pk), 'episodes': int(np.sum(on[1:] & ~on[:-1]) + on[0])}
