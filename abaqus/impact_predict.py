"""Blind predictions for the impact solids regenerated with the CURRENT FHS geometry: F2a/d/e/f/g/h (linear E = 40 MPa) and H2a/d/e (EPU 45).
Ours energy as in the input (the solid's own initial KE can differ by ~1 %). Models: bare beams, JOINT_FHS, rule R5. For each: max stroke [mm],
time of max stroke [ms], peak force [N], energy absorbed at 50 / 75 % of the max stroke [N mm]. Abaqus rows are added when the CSVs exist.
    PYTHONPATH=src:tests python abaqus/impact_predict.py [cases...]   -> abaqus/results/F2H2_summary.md"""
import json
import pathlib
import resource
import sys

import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (24 * 1024 ** 3,) * 2)
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases                                                     # noqa: E402
import compare                                                   # noqa: E402
import fit_rule2 as fr                                           # noqa: E402
import h_series_analysis as ha                                   # noqa: E402

NAMES = ['F2e', 'F2f', 'F2g', 'F2h', 'H2e', 'F2d', 'F2a', 'H2d', 'H2a']


def curve(t, u, ke, rf):
    s, e, tm, sm = ha.loading_curve(t, u, ke)
    q = np.array([0.5, 0.75]) * sm
    return sm, tm * 1e3, float(np.abs(rf).max()), ha.absorbed_at(s, e, q)


def main(names):
    r5 = json.loads((ROOT / 'results' / 'rule_fit3' / 'R5.json').read_text())['params']
    import runStatus as rs                                       # repo rule: every long job reports to run_status.log
    rs.start('F2/H2 impact predictions ' + ' '.join(names))
    L = ['# F2 / H2 impact solids (current FHS geometry): blind predictions', '',
         '| case | model | max stroke [mm] | time [ms] | peak force [N] | E(50 %) [N mm] | E(75 %) [N mm] |', '|---|---|---|---|---|---|---|']
    for ci, case in enumerate(names):
        rs.update(case, ci, len(names))
        spec = cases.CASES[case]['spec']
        res = {}
        for lab, joint in (('bare beams', None), ('JOINT_FHS', cases.JOINT_FHS), ('rule R5', fr.joint_of(r5))):
            o = cases.a4_ours(dict(spec, joint=joint))
            res[lab] = curve(o['time'], o['U3ref'], o['ALLKE'], o['RF3bot'])
        try:
            ab = compare.read_abaqus(spec['name'])
            lb = compare.labels(case)
            ta, ua = compare.pick(ab, 'U3', ('node', lb['ref']))
            ka = compare.pick(ab, 'ALLKE')
            ra = compare.pick(ab, 'RF3', ('sum', lb['bot']))
            res['**Abaqus solid**'] = curve(ta, ua, np.interp(ta, ka[0], ka[1]), ra[1])
        except Exception:
            pass
        for lab, (sm, tm, pk, e) in res.items():
            L.append(f"| {case} | {lab} | {sm:.3f} | {tm:.3f} | {pk:.4g} | {e[0]:.4g} | {e[1]:.4g} |")
        print(chr(10).join(L[-len(res):]), flush=True)
    (ROOT / 'results' / 'F2H2_summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main(sys.argv[1:] or NAMES)
