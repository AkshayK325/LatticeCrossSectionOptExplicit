"""Our solver vs Abaqus/Explicit for one benchmark case.

    PYTHONPATH=src python abaqus/compare.py A1 [A2 ...]

1. Runs our solver on the case (abaqus/cases.py) and saves abaqus/ours/<name>.csv (always: the 'prediction').
2. If abaqus/csv/<name>_history.csv exists (from extract_odb.py), overlays both and computes the metrics;
   otherwise falls back to the .sta energy table if present.
3. Writes a numbered figure in ResultsFig/abaqus/ and abaqus/results/<name>.md.
"""
import csv
import datetime
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases                                                     # noqa: E402

FIGDIR = ROOT.parent / 'ResultsFig' / 'abaqus'


def read_abaqus(name):
    """{(region, variable): (t, y)} from <name>_history.csv, or None."""
    # whole file, or split parts (<name>_history_*.csv / <name>_history_h1.csv ...) for large runs
    parts = sorted((ROOT / 'csv').glob(f'{name}_history.csv')) + sorted((ROOT / 'csv').glob(f'{name}_history_*.csv'))
    if not parts:
        return None
    data = {}
    for p in parts:
        with open(p) as f:
            for row in csv.DictReader(f):
                data.setdefault((row['region'], row['variable']), []).append((float(row['time']), float(row['value'])))
    for k in data:                                                    # parts may interleave: sort by time
        data[k].sort()
    return {k: tuple(np.array(v).T) for k, v in data.items()} or None     # header only (failed job) -> None


def read_sta(name):
    """Fallback without the odb/CSV: the Abaqus/Explicit .sta table (time, kinetic energy, total energy at the
    field-output increments, 4 significant digits). Looks in abaqus/logs/ (then csv/, inp/)."""
    for d in ('logs', 'csv', 'inp'):
        p = ROOT / d / f'{name}.sta'
        if p.exists():
            break
    else:
        return None
    t, ke, et = [], [], []
    for line in p.read_text().splitlines():
        f = line.split()
        # rows: increment, total time, step time, wall hh:mm:ss, stable inc, critical element, KE, total energy
        if len(f) == 8 and f[0].isdigit() and ':' in f[3]:
            t.append(float(f[1]))
            ke.append(float(f[6]))
            et.append(float(f[7]))
    if not t:
        return None
    return {('Assembly ASSEMBLY (from .sta)', 'ALLKE'): (np.array(t), np.array(ke)),
            ('Assembly ASSEMBLY (from .sta)', 'ETOTAL'): (np.array(t), np.array(et))}


def pick(ab, var, where=None):
    """Abaqus history of `var`. where: None (assembly, energies), ('node', label), ('sum', labels) or
    ('wsum', labels, weights) (weighted sum over nodes, e.g. momentum = sum m v)."""
    if where is None:
        for (reg, v), ty in ab.items():
            if v == var and reg.startswith('Assembly'):
                return ty
        return None
    kind, labels = where[0], where[1]
    labels = [labels] if kind == 'node' else list(labels)
    w = where[2] if kind == 'wsum' else np.ones(len(labels))
    found = {}
    for (reg, v), ty in ab.items():
        if v == var and reg.startswith('Node'):
            lab = int(reg.rsplit('.', 1)[-1])
            if lab in labels:
                found[lab] = ty
    if len(found) != len(labels):
        return None
    t = found[labels[0]][0]
    return t, sum(wi * np.interp(t, *found[lab]) for wi, lab in zip(w, labels))


def nrmse(t_ref, y_ref, t, y):
    """RMS of (ours - Abaqus) on Abaqus's time points, normalised by max |Abaqus|."""
    yi = np.interp(t_ref, t, y)
    return float(np.sqrt(np.mean((yi - y_ref) ** 2)) / max(np.abs(y_ref).max(), 1e-300))


def corr(t_ref, y_ref, t, y, window=None):
    """Correlation on Abaqus's time points; window [s]: moving-average low-pass of both first (removes
    high-frequency axial ringing whose phase is not reproducible between codes)."""
    yi = np.interp(t_ref, t, y)
    if window:
        k = max(1, int(round(window / np.median(np.diff(t_ref)))))
        w = np.ones(k) / k
        yi, y_ref = np.convolve(yi, w, 'valid'), np.convolve(y_ref, w, 'valid')
    return float(np.corrcoef(yi, y_ref)[0, 1])


def period(t, y):
    """Mean period from upward zero crossings of y - mean(y)."""
    z = y - y.mean()
    k = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    if len(k) < 2:
        return float('nan')
    tc = t[k] - z[k] * (t[k + 1] - t[k]) / (z[k + 1] - z[k])
    return float(np.mean(np.diff(tc)))


def first_time(t, y, level):
    """First time |y| exceeds level (linear interpolation)."""
    k = int(np.argmax(np.abs(y) > level))
    if k == 0:
        return float('nan')
    a, b = abs(y[k - 1]), abs(y[k])
    return float(t[k - 1] + (level - a) / (b - a) * (t[k] - t[k - 1]))


# ------------------------------------------------------------------------------------------------ per case
def _rig_labels_from_inp(name):
    """Reference node and BOT node set of a B2 / F rig, read from the .inp that was run (re-meshing need not give
    the same labels once the mesh code has changed)."""
    import re
    txt = (cases.ROOT / 'inp' / f'{name}.inp').read_text()
    ref = int(re.search(r'\*NSET, NSET=REF\n(\d+)', txt).group(1))
    blk = re.search(r'\*NSET, NSET=BOT\n(.*?)\n\*', txt, re.S).group(1)
    bot = [int(v) for v in re.split(r'[,\s]+', blk.strip()) if v]
    return {'ref': ref, 'bot': bot}


def _b1_labels_from_inp(name):
    """Driven node and BOT node set of a B1 strut, read from the .inp that was run."""
    import re
    txt = (cases.ROOT / 'inp' / f'{name}.inp').read_text()
    drv = int(re.search(r'\*NSET, NSET=DRV\n(\d+)', txt).group(1))
    blk = re.search(r'\*NSET, NSET=BOT\n(.*?)\n\*', txt, re.S).group(1)
    return {'drv': drv, 'bot': [int(v) for v in re.split(r'[,\s]+', blk.strip()) if v]}


def labels(case):
    if case.startswith('B1'):
        return _b1_labels_from_inp(cases.CASES[case]['spec']['name'])
    if case == 'A1':
        return {'tip': cases.A1['n_el'] + 1, 'root': 1}
    if case.startswith('A2'):
        n = cases.CASES[case]['spec']['n_el']
        return {'top': n + 1, 'mid': n // 2 + 1, 'root': 1}
    if case in ('A3', 'A3b', 'A3c'):
        n = cases.CASES[case]['spec']['n_el']
        return {'amid': n // 2 + 1, 'bmid': n + 1 + n // 2 + 1, 'b': list(range(n + 2, 2 * n + 3))}
    if case in ('A3s', 'A3s1'):
        g = cases.a3s_mesh(cases.CASES[case]['spec'])
        return {'amid': g['midA'], 'bmid': g['midB'], 'b': g['labelsB']}
    if case.startswith(('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'F', 'H')):   # F = B2 rig with FHS joints, H = F with EPU 45, B3/4 driven
        return _rig_labels_from_inp(cases.CASES[case]['spec']['name'])
    if case in ('A4', 'A5a', 'A5b', 'A5as', 'A5aq', 'A5bs'):
        lat, rig = cases._a4_rig(cases.CASES[case]['spec'])
        return {'ref': len(lat['Xr']) + 1, 'bot': [int(k) + 1 for k in rig['bot']]}
    raise KeyError(case)


def signals(case, ours):
    """(label, our key, Abaqus variable, where, unit, nRMSE limit or None)."""
    L = labels(case)
    E = [('kinetic energy', 'ALLKE', 'ALLKE', None, 'N mm', None),
         ('strain energy', 'ALLSE', 'ALLSE', None, 'N mm', None)]
    if case.startswith('H'):                                     # EPU 45: viscous (Prony) dissipation
        E.append(('viscous dissipation', 'ALLCD', 'ALLCD', None, 'N mm', None))
    if case.startswith(('B3', 'B4', 'B5', 'B6', 'B7', 'B8')):                     # joint specimen / cell, prescribed plate motion
        return [('plate U3 (driven)', 'U3ref', 'U3', ('node', L['ref']), 'mm', 0.01),
                ('bottom reaction (sum RF3)', 'RF3bot', 'RF3', ('sum', L['bot']), 'N', 0.05),
                ('kinetic energy', 'ALLKE', 'ALLKE', None, 'N mm', None)]
    if case.startswith('B1'):                                   # single EPU 45 strut, prescribed ramp-and-hold
        comp = '1' if cases.CASES[case]['spec']['mode'] == 'bend' else '3'
        return [('driven displacement', 'Udrv', 'U' + comp, ('node', L['drv']), 'mm', 0.01),
                ('bottom reaction (sum RF)', 'RFbot', 'RF' + comp, ('sum', L['bot']), 'N', 0.05),
                ('kinetic energy', 'ALLKE', 'ALLKE', None, 'N mm', None)]
    if case == 'A1':
        return [('tip U3', 'U3', 'U3', ('node', L['tip']), 'mm', 0.01), ('tip U1', 'U1', 'U1', ('node', L['tip']), 'mm', 0.02),
                ('tip UR2', 'UR2', 'UR2', ('node', L['tip']), 'rad', 0.01),
                ('root RF3', 'RF3', 'RF3', ('node', L['root']), 'N', None)] + E
    if case.startswith('A2'):
        return [('top U3 (stroke)', 'U3top', 'U3', ('node', L['top']), 'mm', 0.02),
                ('top V3', 'V3top', 'V3', ('node', L['top']), 'mm/s', None),
                ('mid U1 (buckling)', 'U1mid', 'U1', ('node', L['mid']), 'mm', None),
                ('root RF3', 'RF3', 'RF3', ('node', L['root']), 'N', None)] + E
    if case in ('A3', 'A3b', 'A3c', 'A3s', 'A3s1'):
        mB = ours['massB_solid'] if case.startswith('A3s') else ours['massB']
        return [('B mid U3', 'U3Bmid', 'U3', ('node', L['bmid']), 'mm', None),
                ('A mid U3', 'U3Amid', 'U3', ('node', L['amid']), 'mm', None),
                ('B momentum z', 'PB3', 'V3', ('wsum', L['b'], mB), 't mm/s', None)] + E
    if case in ('A5a', 'A5b', 'A5as', 'A5aq', 'A5bs') or case.startswith(('B2', 'F', 'H')):
        return [('plate U3 (stroke)', 'U3ref', 'U3', ('node', L['ref']), 'mm', None),
                ('plate V3', 'V3ref', 'V3', ('node', L['ref']), 'mm/s', None),
                ('bottom reaction (sum RF3)', 'RF3bot', 'RF3', ('sum', L['bot']), 'N', None),
                ('friction work (ours) / ALLFD', 'WFRIC', 'ALLFD', None, 'N mm', None)] + E
    if case == 'A4':
        return [('plate U3 (stroke)', 'U3ref', 'U3', ('node', L['ref']), 'mm', 0.02),
                ('plate V3', 'V3ref', 'V3', ('node', L['ref']), 'mm/s', None),
                ('bottom reaction (sum RF3)', 'RF3bot', 'RF3', ('sum', L['bot']), 'N', 0.05)] + E
    raise KeyError(case)


def scalar_metrics(case, ours, ab):
    """(name, ours, Abaqus, relative difference or None, limit) for the plan's pass criteria."""
    L = labels(case)
    out = []

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-300)
    if case.startswith('B1'):
        sp = cases.CASES[case]['spec']
        comp = '1' if sp['mode'] == 'bend' else '3'
        rf, ud = pick(ab, 'RF' + comp, ('sum', L['bot'])), pick(ab, 'U' + comp, ('node', L['drv']))
        if rf is not None and ud is not None:
            to, ro = ours['time'], ours['RFbot']
            tr, te = sp['t_ramp'], sp['t_ramp'] + sp['t_hold']
            pk_o, pk_a = np.abs(ro).max(), np.abs(rf[1]).max()
            out.append(('peak |reaction| [N]', pk_o, pk_a, rel(pk_o, pk_a), 0.05))
            r_o, r_a = np.interp(tr, to, ro), np.interp(tr, *rf)
            out.append(('reaction at the end of the ramp [N]', r_o, r_a, rel(r_o, r_a), 0.05))
            h_o, h_a = np.interp(te, to, ro), np.interp(te, *rf)
            out.append(('reaction at the end of the hold (relaxed) [N]', h_o, h_a, rel(h_o, h_a), 0.05))
            k_o, k_a = to <= tr, rf[0] <= tr                     # work during the ramp: int RF dU
            w_o = np.trapezoid(ro[k_o], ours['Udrv'][k_o])
            w_a = np.trapezoid(rf[1][k_a], np.interp(rf[0][k_a], *ud))
            out.append(('work during the ramp [N mm]', w_o, w_a, rel(w_o, w_a), 0.05))
    if case == 'A1':
        r = pick(ab, 'U3', ('node', L['tip']))
        if r is not None:
            Tp, Ta = period(ours['time'], ours['U3']), period(*r)
            out.append(('U3 period [ms]', Tp * 1e3, Ta * 1e3, rel(Tp, Ta), 0.005))
        elif pick(ab, 'ALLKE') is not None:                      # .sta only: KE period = U3 period / 2
            Tk, Tko = period(*pick(ab, 'ALLKE')), period(ours['time'], ours['ALLKE'])
            out.append(('kinetic-energy period [ms]', Tko * 1e3, Tk * 1e3, rel(Tko, Tk), 0.005))
    if case.startswith('A2'):
        # graded on the COMPRESSION phase (t <= Abaqus time of max stroke): the rebound then stretches the strut far
        # into tension, where 'linear elastic' differs between codes (engineering vs log strain), see the plan
        u = pick(ab, 'U3', ('node', L['top']))
        rf = pick(ab, 'RF3', ('node', L['root']))
        mu = pick(ab, 'U1', ('node', L['mid']))
        if u is not None:
            tcut = u[0][np.argmin(u[1])]
            k = u[0] <= tcut
            out.append(('max stroke [mm]', -ours['U3top'].min(), -u[1].min(), rel(ours['U3top'].min(), u[1].min()), 0.02))
            e = nrmse(u[0][k], u[1][k], ours['time'], ours['U3top'])
            out.append(('stroke nRMSE, compression phase', e, 0.0, e, 0.02))
            if mu is not None:
                lev = 0.2 * np.abs(mu[1]).max()                  # 20% of the Abaqus max lateral deflection
                tb_o, tb_a = first_time(ours['time'], ours['U1mid'], lev), first_time(*mu, lev)
                out.append((f'buckling time (|mid U1| > {lev:.3g} mm) [ms]', tb_o * 1e3, tb_a * 1e3, rel(tb_o, tb_a), 0.03))
            if rf is not None and mu is not None:
                w = (rf[0] >= tb_a) & (rf[0] <= tcut)
                if w.sum() > 3:
                    po, pa = np.interp(rf[0][w], ours['time'], ours['RF3']).mean(), rf[1][w].mean()
                    out.append(('post-buckling plateau, mean root RF3 [N]', po, pa, rel(po, pa), 0.03))
                kr = rf[0] <= tcut
                out.append(('root RF3 correlation, compression phase, 0.1 ms low-pass',
                            corr(rf[0][kr], rf[1][kr], ours['time'], ours['RF3'], window=1e-4), 1.0, None, 0.98))
    if case in ('A3', 'A3b', 'A3c', 'A3s', 'A3s1'):
        mB = ours['massB_solid'] if case.startswith('A3s') else ours['massB']
        if case.startswith('A3s'):
            ae, ie = pick(ab, 'ALLAE'), pick(ab, 'ALLIE')
            if ae is not None and ie is not None:
                fr = ae[1][-1] / max(ie[1][-1], 1e-300)
                out.append(('Abaqus hourglass energy / internal energy (end)', fr, 0.0, fr, 0.10))
            cf = [ty for (reg, v), ty in ab.items() if v == 'CFN3']
            if cf:
                tc, fc = cf[0]
                Ic = abs(np.trapezoid(fc, tc)) / -ours['PB3'][0] * (ours['massB_solid'].sum() / ours['massB'].sum())
                Io = (ours['PB3'][-1] - ours['PB3'][0]) / -ours['PB3'][0]
                out.append(('impulse from contact force CFN3 / initial momentum', Io, Ic, rel(Io, Ic), 0.05))
        r = pick(ab, 'V3', ('wsum', L['b'], mB))
        if r is not None:
            Io, Ia = ours['PB3'][-1] - ours['PB3'][0], r[1][-1] - r[1][0]
            if case.startswith('A3s'):                                    # solid B has 1.1% less mass (24-gon section)
                Io, Ia = Io / -ours['PB3'][0], Ia / -r[1][0]
                out.append(('impulse on B / initial momentum', Io, Ia, rel(Io, Ia), 0.05))
            else:
                out.append(('impulse on B [t mm/s]', Io, Ia, rel(Io, Ia), 0.05))
            ro, ra = -ours['PB3'][-1] / ours['PB3'][0], -r[1][-1] / r[1][0]
            out.append(('B rebound momentum ratio', ro, ra, rel(ro, ra), 0.05))
        r = pick(ab, 'U3', ('node', L['bmid']))
        if r is not None:
            out.append(('B mid max downward U3 [mm]', -ours['U3Bmid'].min(), -r[1].min(),
                        rel(ours['U3Bmid'].min(), r[1].min()), None))
    if case in ('A5a', 'A5b', 'A5as', 'A5aq', 'A5bs') or case.startswith(('B2', 'F', 'H')):
        r = pick(ab, 'U3', ('node', L['ref']))
        if r is not None:
            ia = np.argmin(r[1])
            out.append(('max stroke [mm]', -ours['U3ref'].min(), -r[1].min(), rel(ours['U3ref'].min(), r[1].min()), 0.03))
            to = ours['time'][np.argmin(ours['U3ref'])]
            out.append(('time of max stroke [ms]', to * 1e3, r[0][ia] * 1e3, rel(to, r[0][ia]), 0.05))
            k = r[0] <= r[0][ia]
            e = nrmse(r[0][k], r[1][k], ours['time'], ours['U3ref'])
            out.append(('stroke nRMSE, loading', e, 0.0, e, 0.03))
        rf = pick(ab, 'RF3', ('sum', L['bot']))
        if rf is not None:
            pa, po = np.abs(rf[1]).max(), np.abs(ours['RF3bot']).max()
            out.append(('peak |bottom reaction| [N]', po, pa, rel(po, pa), 0.10))
            if r is not None:
                k = rf[0] <= r[0][np.argmin(r[1])]
                out.append(('bottom reaction correlation, loading, 0.1 ms low-pass',
                            corr(rf[0][k], rf[1][k], ours['time'], ours['RF3bot'], window=1e-4), 1.0, None, 0.95))
        fd = pick(ab, 'ALLFD')
        if case in ('A5b', 'A5bs') and fd is not None:
            out.append(('friction dissipation at end [N mm]', ours['WFRIC'][-1], fd[1][-1], rel(ours['WFRIC'][-1], fd[1][-1]), 0.15))
    if case == 'A4':
        r = pick(ab, 'U3', ('node', L['ref']))
        if r is not None:
            out.append(('max stroke [mm]', -ours['U3ref'].min(), -r[1].min(), rel(ours['U3ref'].min(), r[1].min()), 0.02))
            k = r[0] <= r[0][np.argmin(r[1])]
            e = nrmse(r[0][k], r[1][k], ours['time'], ours['U3ref'])
            out.append(('stroke nRMSE, first compression', e, 0.0, e, 0.02))
            Tp, Ta = period(ours['time'], ours['U3ref']), period(*r)
            out.append(('plate U3 period [ms]', Tp * 1e3, Ta * 1e3, rel(Tp, Ta), 0.01))
        r = pick(ab, 'RF3', ('sum', L['bot']))
        if r is not None:
            pa, po = np.abs(r[1]).max(), np.abs(ours['RF3bot']).max()
            out.append(('peak |bottom reaction| [N]', po, pa, rel(po, pa), 0.05))
    return out


def main(case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT.parent / 'src'))
    from plotFrame import numbered
    c = cases.CASES[case]
    name = c['spec']['name']
    ours = c['ours']()
    (ROOT / 'ours').mkdir(exist_ok=True)
    keys = [k for k in ours if isinstance(ours[k], np.ndarray) and ours[k].shape == ours['time'].shape]
    with open(ROOT / 'ours' / f'{name}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(zip(*[ours[k] for k in keys]))
    ab = read_abaqus(name)
    source = f'abaqus/csv/{name}_history.csv'
    if ab is None:
        ab = read_sta(name)
        source = f'{name}.sta only (kinetic and total energy at the field frames, 4 digits): no odb/CSV yet'
    if case.startswith(('B2', 'F', 'H')) and ab is not None and pick(ab, 'ALLKE') is not None:
        ke0, ke_nom = pick(ab, 'ALLKE')[1][0], c['spec']['energy_J'] * 1e3
        if abs(ke0 / ke_nom - 1) > 0.01:                       # solid started with a different moving mass
            jm = cases.CASES[case].get('joint', c['spec'].get('joint'))
            ours = cases.a4_ours(dict(c['spec'], energy_J=ke0 * 1e-3, joint=jm))
            source += f' (ours re-run energy-matched to the solid: {ke0:.3f} vs nominal {ke_nom:.3f} N mm)'
    sig = signals(case, ours)
    ncol = (len(sig) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(5 * ncol, 7))
    rows = []
    for ax, (label, ok, av, where, unit, lim) in zip(axs.flat, sig):
        t = ours['time']
        ax.plot(t * 1e3, ours[ok], 'C3-', lw=1.5, label='ours')
        ref = None if ab is None else pick(ab, av, where)
        if ref is not None:
            ax.plot(ref[0] * 1e3, ref[1], 'k--', lw=1.2, label='Abaqus/Explicit')
            e = nrmse(ref[0], ref[1], t, ours[ok])
            rows.append((label, f'{e:.2e}', '—' if lim is None else ('✔' if e <= lim else f'✘ (limit {lim})')))
        ax.set(xlabel='time [ms]', ylabel=f'{label} [{unit}]')
        ax.legend(fontsize=8)
    for ax in list(axs.flat)[len(sig):]:
        ax.axis('off')
    fig.suptitle(c['spec']['title'], fontsize=9)
    fig.tight_layout()
    fp = numbered(FIGDIR, f'{name}.png')
    fig.savefig(fp, dpi=120)
    plt.close(fig)
    (ROOT / 'results').mkdir(exist_ok=True)
    md = [f'# {name}: ours vs Abaqus/Explicit', '',
          f'Generated by `abaqus/compare.py {case}` ({datetime.datetime.now():%Y-%m-%d %H:%M}).', '',
          c['spec']['title'] + '.', '',
          f'- Ours: {ours["n_steps"]} steps, dt = {ours["dt"]:.3e} s.',
          f'- Abaqus: {source if ab is not None else "**no CSV yet** (prediction only)"}.',
          f'- Figure: `ResultsFig/abaqus/{fp.name}`', '']
    sm = [] if ab is None else scalar_metrics(case, ours, ab)
    if sm:
        md += ['| metric | ours | Abaqus | rel. diff | pass |', '|---|---|---|---|---|']
        for nm, o, a, rd, lim in sm:
            if rd is None:                                       # correlation-type: compare value to limit
                md.append(f'| {nm} | {o:.4f} | — | — | {"✔" if o >= lim else f"✘ (≥ {lim})"} |')
            else:
                md.append(f'| {nm} | {o:.5g} | {a:.5g} | {rd:.2%} | '
                          f'{"—" if lim is None else ("✔" if rd <= lim else f"✘ (limit {lim:.1%})")} |')
        md.append('')
    if ab is not None and pick(ab, 'ETOTAL') is not None and pick(ab, 'ALLKE') is not None:
        ta, ea = pick(ab, 'ETOTAL')                              # constant (= initial energy) when conserved
        md.append(f'- Energy balance: Abaqus ETOTAL drift (max − min) {np.ptp(ea):.3e} N mm '
                  f'({np.ptp(ea) / pick(ab, "ALLKE")[1].max():.2%} of max KE).')
    if rows:
        md += ['', '| signal | nRMSE (ours − Abaqus, / max abs Abaqus) | pass |', '|---|---|---|']
        md += [f'| {a} | {b} | {p} |' for a, b, p in rows]
    if ab is None:
        md += ['', '### Our prediction (before Abaqus)']
        md += [f'- {k}: min {ours[k].min():.5g}, max {ours[k].max():.5g}' for k in keys if k != 'time']
    (ROOT / 'results' / f'{name}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    for case in (sys.argv[1:] or ['A1']):
        main(case)
