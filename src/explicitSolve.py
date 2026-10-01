"""Explicit central-difference (velocity Verlet) time integration for explicitFE.Frame.

Translations: v_{n+1/2} = v_n + dt/2 a_n, u_{n+1} = u_n + dt v_{n+1/2}, v_{n+1} = v_{n+1/2} + dt/2 a_{n+1}.
Rotations: same with spatial angular velocity w and R_{n+1} = exp(dt w_{n+1/2}) R_n (exact on SO(3)).
Isotropic nodal rotary inertia j, so j w' = moment (no gyroscopic term).
"""
import torch

from explicitFE import rodrigues


def strain_energy(d, f_loc):
    """0.5 d.f for linear-elastic laws (resultants linear in d)."""
    return 0.5 * (d * f_loc).sum()


def simulate(frame, law, m, j, dt, n_steps, u0=None, v0=None, R0=None, w0=None,
             fix_u=None, fix_r=None, f_ext=None, damping=0.0, record_every=1, callback=None):
    """Run n_steps. fix_u / fix_r: bool masks (N,3) of fixed translation / rotation DOFs (held at u0/R0).
    f_ext(t) -> (F (N,3), Mom (N,3)) or None. damping: mass-proportional coefficient alpha (1/s).
    callback(step, t, state, fint) -> dict of scalars to record. Returns dict of recorded histories."""
    N, dtype = frame.N, frame.X.dtype
    z = torch.zeros(N, 3, dtype=dtype)
    u = z.clone() if u0 is None else u0.clone()
    v = z.clone() if v0 is None else v0.clone()
    w = z.clone() if w0 is None else w0.clone()
    R = torch.eye(3, dtype=dtype).expand(N, 3, 3).clone() if R0 is None else R0.clone()
    keep_u = torch.zeros(N, 3, dtype=torch.bool) if fix_u is None else fix_u
    keep_r = torch.zeros(N, 3, dtype=torch.bool) if fix_r is None else fix_r
    m3, j3 = m[:, None], j[:, None]

    def accel(t, u, R, v, w):
        fu, fth, d, f_loc = frame.internal_force(u, R, law)
        Fx, Mx = (z, z) if f_ext is None else f_ext(t)
        a = (Fx - fu) / m3 - damping * v
        al = (Mx - fth) / j3 - damping * w
        return a.masked_fill(keep_u, 0.0), al.masked_fill(keep_r, 0.0), (fu, fth, d, f_loc)

    hist = {'t': [], 'u': [], 'R': []}
    t = 0.0
    a, al, fint = accel(t, u, R, v, w)
    for n in range(n_steps + 1):
        if n % record_every == 0:
            hist['t'].append(t)
            hist['u'].append(u.detach().clone())
            hist['R'].append(R.detach().clone())
            if callback is not None:
                for k, val in callback(n, t, (u, v, R, w), fint).items():
                    hist.setdefault(k, []).append(val)
        if n == n_steps:
            break
        v = v + 0.5 * dt * a
        w = w + 0.5 * dt * al
        u = u + dt * v
        R = rodrigues(dt * w) @ R
        t += dt
        a, al, fint = accel(t, u, R, v, w)
        v = v + 0.5 * dt * a
        w = w + 0.5 * dt * al
    hist['u'] = torch.stack(hist['u'])
    hist['R'] = torch.stack(hist['R'])
    hist['state'] = (u, v, R, w)
    return hist
