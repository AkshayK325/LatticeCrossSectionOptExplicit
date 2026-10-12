"""Our beam model on the H cells with the Prony branches relaxing nominal (current) vs Kirchhoff-type axial force:
max stroke, time of max, energy absorbed at 50 / 75 % and peak reaction vs the Abaqus H solids (energy-matched).
    PYTHONPATH=src:tests python abaqus/h_relax_check.py H1e H1f H1d ...   -> abaqus/results/H_relax_check.md"""
import pathlib
import resource
import sys

import numpy as np

lim = 24 * 1024 ** 3
resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases                                                     # noqa: E402
import compare                                                   # noqa: E402
import h_series_analysis as ha                                   # noqa: E402


def run(case, relax):
    c = cases.CASES[case]
    ab = compare.read_abaqus(c['spec']['name'])
    ke0 = compare.pick(ab, 'ALLKE')[1][0]
    jm = c.get('joint', c['spec'].get('joint'))
    return cases.a4_ours(dict(c['spec'], energy_J=ke0 * 1e-3, joint=jm, relax=relax)), ab


def main(cases_):
    import runStatus as rs
    rs.start('H relax check ' + ' '.join(cases_))
    lines = ['# H cells: Prony relaxation of the nominal vs the Kirchhoff-type axial force (beam model vs Abaqus)', '',
             '| case | relax | max stroke err | time of max err | energy err at 50 / 75 % | peak reaction err |',
             '|---|---|---|---|---|---|']
    for i, case in enumerate(cases_):
        for relax in ('nominal', 'kirchhoff'):
            rs.update(f'{case} {relax}', 2 * i + (relax == 'kirchhoff'), 2 * len(cases_))
            ours, ab = run(case, relax)
            L = compare.labels(case)
            ta, ua = compare.pick(ab, 'U3', ('node', L['ref']))
            ka = compare.pick(ab, 'ALLKE')
            ra = compare.pick(ab, 'RF3', ('sum', L['bot']))
            sa, ea, tma, sma = ha.loading_curve(ta, ua, np.interp(ta, ka[0], ka[1]))
            so, eo, tmo, smo = ha.loading_curve(ours['time'], ours['U3ref'], ours['ALLKE'])
            q = np.array([0.5, 0.75]) * sma
            dE = ha.absorbed_at(so, eo, q) / ha.absorbed_at(sa, ea, q) - 1
            pk = np.abs(ours['RF3bot']).max() / np.abs(ra[1]).max() - 1
            lines.append(f'| {case} | {relax} | {100 * (smo / sma - 1):+.1f} % | {100 * (tmo / tma - 1):+.1f} % | '
                         f'{100 * dE[0]:+.1f} / {100 * dE[1]:+.1f} % | {100 * pk:+.1f} % |')
            print(lines[-1], flush=True)
    (ROOT / 'results' / 'H_relax_check.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
