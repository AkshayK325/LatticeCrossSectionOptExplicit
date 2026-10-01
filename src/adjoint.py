"""Memory-efficient exact gradients through the explicit drop: discrete adjoint with two-level checkpointing.

No unrolled time loop. The velocity-Verlet step is written as a pure map
    z_{n+1}, L_n = Phi_n(z_n; theta),  z = (u, v, R, w, a, al, q, f_prev, Fb)
(q, f_prev: Prony history of the section law; Fb: bottom reaction used by the placeholder objective),
J = sum_n L_n. Forward: no graph, checkpoints every K steps (+ the contact Verlet-list state).
Reverse: per segment, recompute its states without a graph, then step backwards with one local graph:
    lam_n = (dPhi_n/dz_n)^T lam_{n+1} + dL_n/dz_n,   g_theta += (dPhi_n/dtheta)^T lam_{n+1}.
Memory ~ (N/K + K) states + one step graph; the gradient equals unrolled reverse AD of the same scheme.
See notes/6_MemoryEfficientAdjointPlan.md.
"""
import math

import torch

from explicitFE import rodrigues

FIELDS = ('u', 'v', 'R', 'w', 'a', 'al', 'q', 'fp', 'Fb')


class DropStepper:
    """Pure step map for a drop rig (latticeSetup.drop_rig) with a replayed dt schedule.

    J_fn(Fb, h) -> per-step objective increment (placeholder until the objective is chosen)."""

    def __init__(self, rig, schedule, J_fn):
        self.rig, self.h = rig, [float(x) for x in schedule]
        self.J_fn = J_fn
        self.law = rig['beam_law']
        self.fr = rig['fr']
        self.m3, self.j3 = rig['m'][:, None], rig['j'][:, None]
        self.keep_u, self.keep_r, self.tie = rig['fix_u'], rig['fix_r'], rig['tie']
        self.sc = rig['self_contact']

    # -- contact Verlet-list bookkeeping (non-differentiable choice, replayed exactly)
    def aux_get(self):
        sc = self.sc
        return None if sc is None else (sc.active, getattr(sc, 'x_ref', None))

    def aux_set(self, aux):
        if aux is not None:
            self.sc.active, self.sc.x_ref = aux

    def _accel(self, u, R, v, w, q, fp):
        law = self.law
        law.q, law.f_prev = q, fp                               # history at the start of the step
        fu, fth, d, f_loc = self.fr.internal_force(u, R, law)
        q_new, f_inf = law.last_q, law.last_f
        fc = self.rig['contact'](u, v)
        a = (fc - fu) / self.m3
        tie = self.tie
        if tie is not None:
            a = a.index_copy(0, tie, ((self.m3[tie] * a[tie]).sum(0) / self.m3[tie].sum()).expand(len(tie), 3))
        al = -fth / self.j3
        Fb = fu[self.rig['bot'], 2].sum()
        return a.masked_fill(self.keep_u, 0.0), al.masked_fill(self.keep_r, 0.0), q_new, f_inf, Fb

    def initial_state(self):
        fr, rig = self.fr, self.rig
        N, dt = fr.N, fr.X.dtype
        u = torch.zeros(N, 3, dtype=dt)
        v = rig['v_init'].clone()
        R = torch.eye(3, dtype=dt).expand(N, 3, 3).clone()
        w = torch.zeros(N, 3, dtype=dt)
        q = torch.zeros_like(self.law.q)
        fp = torch.zeros_like(self.law.f_prev)
        self.law.step_dt = 0.0
        a, al, _, _, Fb = self._accel(u, R, v, w, q, fp)        # initial accel does not commit history
        return dict(u=u, v=v, R=R, w=w, a=a, al=al, q=q, fp=fp, Fb=Fb)

    def step(self, z, n):
        """(z_{n+1}, L_n). L_n = J_fn(Fb_n, h_n) uses the state at the start of the step (as dropGrad)."""
        h = self.h[n]
        self.law.step_dt = h
        self.rig['dt_now']['h'] = h
        L = self.J_fn(z['Fb'], h)
        v = z['v'] + 0.5 * h * z['a']
        w = z['w'] + 0.5 * h * z['al']
        u = z['u'] + h * v
        R = rodrigues(h * w) @ z['R']
        a, al, q, fp, Fb = self._accel(u, R, v, w, z['q'], z['fp'])
        v = v + 0.5 * h * a
        w = w + 0.5 * h * al
        return dict(u=u, v=v, R=R, w=w, a=a, al=al, q=q, fp=fp, Fb=Fb), L


def _detach(z):
    return {k: z[k].detach() for k in FIELDS}


def forward_J(stepper):
    """Plain forward (no graph) -> J. Used to check the stepper against simulate()."""
    with torch.no_grad():
        z = stepper.initial_state()
        J = 0.0
        for n in range(len(stepper.h)):
            z, L = stepper.step(z, n)
            J += float(L)
    return J


def checkpointed_adjoint(stepper, theta, K=None):
    """J and dJ/dtheta (list of tensors, same shapes as theta) via the checkpointed discrete adjoint.
    theta: tensors that the rig depends on (e.g. per-element radii); they must require grad and the
    rig must have been built from them (so the step graph reaches them)."""
    N = len(stepper.h)
    K = K or max(1, int(round(math.sqrt(N))))
    # ---- forward sweep: checkpoints only (state + contact bookkeeping), no graph
    ckpt = {}
    with torch.no_grad():
        z = stepper.initial_state()
        J = 0.0
        for n in range(N):
            if n % K == 0:
                ckpt[n] = (_detach(z), stepper.aux_get())
            z, L = stepper.step(z, n)
            J += float(L)
    # ---- reverse sweep
    lam = {k: torch.zeros_like(z[k]) for k in FIELDS}            # dJ/dz_N = 0 (J has no terminal term)
    g = [torch.zeros_like(t) for t in theta]
    starts = sorted(ckpt)
    for s in reversed(starts):
        e = min(s + K, N)
        zs, aux = ckpt[s]
        stepper.aux_set(aux)
        states, auxes = [zs], [aux]
        with torch.no_grad():                                   # recompute the segment's states
            zz = zs
            for n in range(s, e - 1):
                zz, _ = stepper.step(zz, n)
                states.append(_detach(zz))
                auxes.append(stepper.aux_get())
        for i in range(e - 1, s - 1, -1):                       # local VJP, one step at a time
            n = i
            stepper.aux_set(auxes[i - s])
            zin = {k: states[i - s][k].clone().requires_grad_(True) for k in FIELDS}
            with torch.enable_grad():
                zout, L = stepper.step(zin, n)
                pairs = [(zout[k], lam[k]) for k in FIELDS] + [(L, torch.ones_like(L))]
                pairs = [(o, go) for o, go in pairs if o.requires_grad]   # skip outputs independent of z, theta
                outs, gouts = [o for o, _ in pairs], [go for _, go in pairs]
                # retain_graph: the design -> section-property graph built once with the rig is shared by
                # every step; this step's own graph is freed when zin/zout go out of scope
                grads = torch.autograd.grad(outs, [zin[k] for k in FIELDS] + list(theta), grad_outputs=gouts,
                                            allow_unused=True, retain_graph=True)
            lam = {k: (torch.zeros_like(zin[k]) if gr is None else gr.detach()) for k, gr in zip(FIELDS, grads)}
            for t_i, gr in enumerate(grads[len(FIELDS):]):
                if gr is not None:
                    g[t_i] = g[t_i] + gr.detach()
    # ---- initial state z_0 = Z0(theta): chain lam_0 into theta
    stepper.aux_set(ckpt[0][1])
    with torch.enable_grad():
        z0 = stepper.initial_state()
        pairs = [(z0[k], lam[k]) for k in FIELDS if z0[k].requires_grad]
        grads = (torch.autograd.grad([o for o, _ in pairs], list(theta), grad_outputs=[go for _, go in pairs],
                                     allow_unused=True, retain_graph=True) if pairs else [None] * len(theta))
    for t_i, gr in enumerate(grads):
        if gr is not None:
            g[t_i] = g[t_i] + gr.detach()
    return J, g
