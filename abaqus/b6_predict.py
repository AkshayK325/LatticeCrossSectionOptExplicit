"""B6 = B4 with the current FHS geometry: our beams bare, JOINT_FHS, JOINT_FHS + plate_layer and the simple valence rule
k_end = 1 + a * (neighbours), a = 1 (notes/18 section 6), against the Abaqus solids B6a / B6b, and against B4 (old geometry).
    PYTHONPATH=src:tests python abaqus/b6_predict.py   -> abaqus/results/B6_summary.md"""
import pathlib
import resource
import sys

import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 ** 3,) * 2)
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cases                                                     # noqa: E402
import compare                                                   # noqa: E402
import jointSection                                              # noqa: E402
from b3_predict import stats                                     # noqa: E402

A, FAM = [1.0], ['valence']


def end_data(lat, rb):
    X, C = lat['X'], lat['conn']; nb = len(C)
    L = np.linalg.norm(X[C[:, 1]] - X[C[:, 0]], axis=1)
    out = {}
    for b, (i, j) in enumerate(C):
        for side, (n, o) in enumerate(((i, j), (j, i))):
            ta = (X[o] - X[n]) / L[b]; ell = 0.0; vals = []
            for c in np.nonzero((C[:, 0] == n) | (C[:, 1] == n))[0]:
                if c == b: continue
                o2 = C[c, 1] if C[c, 0] == n else C[c, 0]
                tb = (X[o2] - X[n]) / np.linalg.norm(X[o2] - X[n])
                ct = float(np.clip(ta @ tb, -1, 1)); st = float(np.sqrt(max(1 - ct * ct, 0)))
                if st < 1e-6: continue
                ell = max(ell, (rb[c] + rb[b] * ct) / st)
                vals.append((rb[c] / rb[b], ct, st))
            out[b, side] = (min(ell, L[b] / 2), vals)
    return L, out

def kend(vals):
    a = A[0]
    if FAM[0] == 'valence': return 1 + a * len(vals)
    if FAM[0] == 'area': return 1 + a * sum(q * q for q, ct, st in vals)
    if FAM[0] == 'angle': return 1 + a * sum(q * q * st for q, ct, st in vals)
    if FAM[0] == 'align': return 1 + a * sum(q * q * ct * ct for q, ct, st in vals)

def fake(lat, r_beam, **kw):
    rb = np.asarray(r_beam); L, ed = end_data(lat, rb); boe = np.asarray(lat['beam_of_elem'])
    se = np.ones(len(boe))
    for b in range(len(lat['conn'])):
        idx = np.nonzero(boe == b)[0]; ne = len(idx); le = L[b] / ne
        (l0, v0), (l1, v1) = ed[b, 0], ed[b, 1]; k0, k1 = kend(v0), kend(v1)
        for m, e in enumerate(idx):
            s0, s1 = m * le, (m + 1) * le
            comp = (min(s1, l0) - min(s0, l0)) / k0 + (max(s1, L[b] - l1) - max(s0, L[b] - l1)) / k1
            comp += (s1 - s0) - (min(s1, l0) - min(s0, l0)) - (max(s1, L[b] - l1) - max(s0, L[b] - l1))
            se[e] = le / comp
    return se, None


def simple_rule(sp):
    orig = jointSection.element_factors_aniso
    jointSection.element_factors_aniso = fake
    try:
        return cases.b4_ours(sp, {'x': 1})
    finally:
        jointSection.element_factors_aniso = orig


def solid(case):
    sp = cases.CASES[case]['spec']
    ab = compare.read_abaqus(sp['name'])
    lb = compare.labels(case)
    rf, ud = compare.pick(ab, 'RF3', ('sum', lb['bot'])), compare.pick(ab, 'U3', ('node', lb['ref']))
    return stats(rf[0], np.interp(rf[0], *ud), rf[1], sp)


def main(names):
    L = ['# B6: full BCC r 1.0 cell with the current FHS geometry vs B4 (old geometry), prescribed plate motion', '',
         'Reaction = sum RF3 at the bottom plate; stiffening = end of ramp / bare beams (ours).', '',
         '| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | vs bare |', '|---|---|---|---|---|---|---|']
    for case in names:
        sp = cases.CASES[case]['spec']
        res = {}
        for lab, fn in (('ours, bare beams', lambda: cases.b4_ours(sp, None)),
                        ('ours, JOINT_FHS', lambda: cases.b4_ours(sp, cases.JOINT_FHS)),
                        ('ours, JOINT_FHS + plate_layer', lambda: cases.b4_ours(sp, dict(cases.JOINT_FHS, plate_layer=True))),
                        ('ours, simple rule a = 1', lambda: simple_rule(sp))):
            o = fn()
            res[lab] = stats(o['time'], o['U3ref'], o['RF3bot'], sp)
        res['**Abaqus solid (current geometry)**'] = solid(case)
        old = 'B4' + case[2]
        if old in cases.CASES:
            res['**Abaqus solid (B4, old geometry)**'] = solid(old)
        b = res['ours, bare beams']['ramp_end']
        for lab, s in res.items():
            L.append(f"| {case} | {lab} | {s['peak']:.4g} | {s['ramp_end']:.4g} | {s['hold_end']:.4g} | {s['work']:.4g} | "
                     f"{s['ramp_end'] / b:.3f} |")
        print(chr(10).join(L[-len(res):]), flush=True)
    (ROOT / 'results' / 'B6_summary.md').write_text(chr(10).join(L) + chr(10))


if __name__ == '__main__':
    main(sys.argv[1:] or list(cases.B6_CASES))
