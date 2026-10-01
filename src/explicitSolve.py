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
             fix_u=None, fix_r=None, f_ext=None, contact=None, tie=None, damping=0.0, record_every=1,
             callback=None, t_end=None):
    """Run n_steps. fix_u / fix_r: bool masks (N,3) of fixed translation / rotation DOFs (held at u0/R0).
    f_ext(t) -> (F (N,3), Mom (N,3)) or None. contact(u, v) -> F (N,3) or None.
    tie: node indices moving as one rigid translating body (e.g. a bonded impactor plate; put its mass into m
    on those nodes and fix their rotations). damping: mass-proportional coefficient alpha (1/s).
    dt: float, a per-step sequence (replay a recorded schedule), or a callable dt(step, u, R, v) (adaptive);
    with a callable, n_steps may be None and t_end sets the duration. hist['dt'] holds the steps used.
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
        fc = z if contact is None else contact(u, v)
        a = (Fx + fc - fu) / m3 - damping * v
        if tie is not None:
            a = a.index_copy(0, tie, ((m3[tie] * a[tie]).sum(0) / m3[tie].sum()).expand(len(tie), 3))
        al = (Mx - fth) / j3 - damping * w
        return a.masked_fill(keep_u, 0.0), al.masked_fill(keep_r, 0.0), (fu, fth, d, f_loc, fc)

    def step_dt(n):
        if callable(dt):
            return dt(n, u.detach(), R.detach(), v.detach())
        return float(dt[n]) if hasattr(dt, '__len__') else dt

    if n_steps is None:
        n_steps = len(dt) if hasattr(dt, '__len__') else 10**9
    hist = {'t': [], 'u': [], 'R': [], 'dt': []}
    t = 0.0
    a, al, fint = accel(t, u, R, v, w)
    for n in range(n_steps + 1):
        done = n == n_steps or (t_end is not None and t >= t_end)
        if n % record_every == 0 or done:
            hist['t'].append(t)
            hist['u'].append(u.detach().clone())
            hist['R'].append(R.detach().clone())
            if callback is not None:
                for k, val in callback(n, t, (u, v, R, w), fint).items():
                    hist.setdefault(k, []).append(val)
        if done:
            break
        h = step_dt(n)
        hist['dt'].append(h)
        v = v + 0.5 * h * a
        w = w + 0.5 * h * al
        u = u + h * v
        R = rodrigues(h * w) @ R
        t += h
        a, al, fint = accel(t, u, R, v, w)
        v = v + 0.5 * h * a
        w = w + 0.5 * h * al
    hist['u'] = torch.stack(hist['u'])
    hist['R'] = torch.stack(hist['R'])
    hist['state'] = (u, v, R, w)
    return hist
