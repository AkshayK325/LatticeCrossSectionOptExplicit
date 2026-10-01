"""Beam-on-beam contact uses the beam radii: two crossed beams with DIFFERENT radii (r_A = 0.8, r_B = 0.5).

Static: force is zero while the centreline distance d > r_A + r_B and starts at d = r_A + r_B.
Dynamic: beam A clamped at both ends, beam B (starting with d = 2.0 > 1.3) driven down onto it.
Run as a script for tube (thickness) stills, a GIF and force vs centreline distance.
"""
import math
import pathlib

import numpy as np
import pytest
import torch

from contact import LineContact, SelfContact, potential_force, segment_closest_params
from explicitFE import Frame, lumped_mass, stable_dt
from material import project_material

torch.set_default_dtype(torch.float64)
MAT = project_material()
RA, RB = 0.8, 0.5
OUT = pathlib.Path(__file__).resolve().parents[1] / 'ResultsFig' / 'explicit'


def crossed(d0=2.0, L=10.0, n=10):
    xa = np.stack([np.linspace(-L / 2, L / 2, n + 1), np.zeros(n + 1), np.zeros(n + 1)], 1)
    xb = np.stack([np.full(n + 1, 0.3), np.linspace(-L / 2, L / 2, n + 1), np.full(n + 1, d0)], 1)  # off-node crossing
    c = np.stack([np.arange(n), np.arange(1, n + 1)], 1)
    fr = Frame(np.vstack((xa, xb)), np.vstack((c, c + n + 1)))
    r = torch.cat((torch.full((n,), RA), torch.full((n,), RB)))
    beam = torch.cat((torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)))
    return fr, r, beam


def centreline_distance(fr, x):
    nA = fr.ne // 2
    i, j = torch.meshgrid(torch.arange(nA), torch.arange(nA, fr.ne), indexing='ij')
    i, j = i.flatten(), j.flatten()
    p0, p1, q0, q1 = x[fr.n1[i]], x[fr.n2[i]], x[fr.n1[j]], x[fr.n2[j]]
    s, t = segment_closest_params(p0, p1, q0, q1)
    return ((p0 + s[:, None] * (p1 - p0)) - (q0 + t[:, None] * (q1 - q0))).norm(dim=1).min().item()


@pytest.mark.parametrize('model', ['line', 'capsule'])
def test_onset_at_sum_of_radii(model):
    fr, r, beam = crossed()
    eps = 0.002
    sc = (LineContact(fr, r, beam, MAT['E0'], MAT['nu'], h_q=0.02, eps=eps) if model == 'line' else
          SelfContact(fr, r, MAT['E0'], MAT['nu'], eps=eps, beam_of_elem=beam))
    for d, expect in ((RA + RB + 0.3, 'zero'), (RA + RB + 10 * eps, 'zero'), (RA + RB - 0.02, 'pos'), (RA + RB - 0.05, 'pos')):
        X = fr.X.clone()
        X[fr.N // 2:, 2] = d
        f, _, _ = potential_force(sc.energy, X)
        Fz = f[fr.N // 2:, 2].sum().item()
        if expect == 'zero':
            assert abs(Fz) < 1e-6, (model, d, Fz)
        else:
            assert Fz > 0.05, (model, d, Fz)
    # a centreline-only model (r = 0) would show nothing at d = 1.25: the radii are what make contact here
    X = fr.X.clone()
    X[fr.N // 2:, 2] = RA + RB - 0.05
    sc0 = LineContact(fr, torch.full_like(r, 1e-3), beam, MAT['E0'], MAT['nu'], h_q=0.02, eps=eps)
    assert abs(potential_force(sc0.energy, X)[0].sum().item()) < 1e-9


def run_press(T=8e-3, v_drive=300.0, record_every=10):
    """Beam A clamped at both ends; beam B's end nodes are heavy drivers moving down at v_drive."""
    from explicitSolve import simulate
    fr, r, beam = crossed()
    A = (math.pi * r**2)[:, None].expand(-1, 3).clone()
    I = torch.stack((math.pi * r**4 / 2, math.pi * r**4 / 4, math.pi * r**4 / 4), 1)
    law = lambda d: fr.elastic_law(d, A, I, MAT['E'], MAT['G'])
    m, j = lumped_mass(fr, A, I, MAT['rho'])
    nA = fr.N // 2
    endsB = torch.tensor([nA, fr.N - 1])
    m[endsB] = 1.0                                                  # drivers (1 t): essentially kinematic
    fix = torch.zeros(fr.N, 3, dtype=torch.bool)
    fix[[0, nA - 1]] = True                                         # A clamped
    fix_r = fix.clone()
    sc = LineContact(fr, r, beam, MAT['E0'], MAT['nu'])
    v0 = torch.zeros(fr.N, 3)
    v0[nA:, 2] = -v_drive
    _, wb = stable_dt(fr, law, m, j)
    dt = 0.5 * 2 / math.sqrt(wb**2 + 4 * sc.k_line * 1.0 / m.min().item())
    rec = {'d': [], 'F': []}

    def cb(n, t, state, fint):
        u, v, R_, w = state
        x = fr.X + u
        rec['d'].append(centreline_distance(fr, x))
        rec['F'].append(fint[4][nA:, 2].sum().item())
        return {}

    alpha = 2000.0                                                  # mass-proportional damping (quasi-static)
    Fd = torch.zeros(fr.N, 3)
    Fd[endsB, 2] = m[endsB] * alpha * -v_drive                      # cancels damping on the drivers: steady v
    zero = torch.zeros(fr.N, 3)
    with torch.no_grad():
        h = simulate(fr, law, m, j, dt, int(T / dt), v0=v0, fix_u=fix, fix_r=fix_r, damping=alpha,
                     f_ext=lambda t: (Fd, zero),
                     contact=lambda u, v: potential_force(sc.energy, fr.X + u)[0], callback=cb,
                     record_every=record_every)
    return fr, r, h, np.array(rec['d']), np.array(rec['F'])


def test_press_onset_and_bending():
    fr, r, h, d, F = run_press()
    first = np.argmax(F > 0.01)                                     # 0.01 N: clearly in contact
    eps = 0.05 * RB                                                 # LineContact default smoothing width
    assert d[0] > RA + RB + 0.5                                     # started well apart
    # smooth law: under light load contact acts within ~2 eps of touching (regularisation width, ~5% of r_B)
    assert abs(d[first] - (RA + RB)) < 3 * eps, d[first]            # force appears at d = r_A + r_B
    assert (F[d > RA + RB + 4 * eps] < 1e-3).all()                  # only the smooth tail before touching
    assert F.max() > 0.1 and d.min() > RA + RB - 0.3                 # pressed (soft beams: ~0.3 N), bounded
    nA = fr.N // 2
    assert h['u'][-1][nA // 2, 2] < -0.05                            # beam A bent down by the contact


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotFrame import animate_frame, draw_tubes, numbered

    fr, r, h, d, F = run_press()
    colors = ['tab:blue'] * (fr.ne // 2) + ['tab:orange'] * (fr.ne // 2)
    animate_frame(fr, h['u'], numbered(OUT, 'beam_on_beam_tubes.gif'), times=h['t'], elev=8, azim=-35,
                  radii=r.numpy(), colors=colors, title=f'r_A = {RA}, r_B = {RB}', max_frames=50)
    first = int(np.argmax(F > 0.01))
    picks = [0, first, len(d) - 1]
    labels = ['start: gap = d − (r_A + r_B) = %.2f mm' % (d[0] - RA - RB), 'first contact: d = %.3f mm' % d[first],
              'pressed: d = %.3f mm' % d[-1]]
    fig = plt.figure(figsize=(17, 5))
    X = fr.X.numpy()
    for k, (p, lab) in enumerate(zip(picks, labels)):
        ax = fig.add_subplot(1, 4, k + 1, projection='3d')
        draw_tubes(ax, X + h['u'][p].numpy(), fr.conn.numpy(), r.numpy(), colors)
        ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-1.5, 2.5), title=lab)
        ax.title.set_fontsize(9)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=0, azim=-90)                                # look along y: the crossing in x-z
    ax = fig.add_subplot(1, 4, 4)
    ax.plot(d, F, 'k-')
    ax.axvline(RA + RB, color='r', ls='--', label='r_A + r_B = %.1f mm' % (RA + RB))
    ax.set(xlabel='centreline distance d [mm]', ylabel='contact force on beam B [N]', title='onset at r_A + r_B')
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(numbered(OUT, 'beam_on_beam_radius_check.png'), dpi=130)
    print('start d', d[0], 'first contact d', d[first], 'final d', d[-1], 'Fmax', F.max())
