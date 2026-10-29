"""Our shell (shellFE) on the Abaqus HS1 case: spherical shell bowl, free rigid head sphere (40 g, 1 m/s), fixed anvil sphere.
The mesh is read from the Abaqus input (S4R quads split into 2 triangles). Compare with abaqus/csv/HS1*_history.csv.
Usage: python -u tests/study_hs1_ours.py HS1b_shell_bowl_h2 [T_ms]"""
import re, sys, time, json
import numpy as np, torch
import runStatus as rs
from shellFE import Shell
from explicitFE import rodrigues
from contact import potential_force
torch.set_default_dtype(torch.float64)
name = sys.argv[1]
T = (float(sys.argv[2]) if len(sys.argv) > 2 else 1.5) * 1e-3
rs.start(f'HS1 ours {name}')
E_, NU, RHO, t_sh, GAP, R_HEAD, R_ANV = 2300.0, 0.38, 1.2e-9, 2.0, 0.05, 80.0, 48.0
M, v0 = 4e-5, 1000.0
txt = open(f'abaqus/inp/{name}.inp').read().split('\n')
i0 = txt.index('*NODE') + 1
nodes = {}
i = i0
while not txt[i].startswith('*'):
    a = txt[i].split(','); nodes[int(a[0])] = [float(x) for x in a[1:4]]; i += 1
ie = next(k for k, l in enumerate(txt) if l.startswith('*ELEMENT, TYPE=S4R')) + 1
quads = []
while not txt[ie].startswith('*'):
    a = [int(x) for x in txt[ie].split(',')]; quads.append(a[1:5]); ie += 1
ids = sorted(nodes); idx = {n: k for k, n in enumerate(ids)}
X = np.array([nodes[n] for n in ids]); q = np.array([[idx[n] for n in qq] for qq in quads])
tri = np.vstack((q[:, [0, 1, 2]], q[:, [0, 2, 3]]))
MID = 'midplane' in name
Rs = R_HEAD + GAP + (t_sh / 2 if MID else 0.0)
HS = t_sh / 2 if MID else 0.0     # head-side face offset from the reference surface
C_h = np.array([0, 0, Rs])
D = (C_h - X); D /= np.linalg.norm(D, axis=1, keepdims=True)
n_f = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
flip = (n_f * D[tri].mean(1)).sum(1) < 0
tri[flip] = tri[flip][:, [0, 2, 1]]
sh = Shell(X, tri, D, t_sh, E_, NU, RHO, offset=0.0 if MID else -t_sh / 2)
N = len(X)
apex = int(np.argmin(np.linalg.norm(X[:, :2], axis=1)))
C_a = torch.tensor([0, 0, -(t_sh / 2 if MID else t_sh) - GAP - R_ANV])
k_n = E_ / t_sh * sh.A_node
eps = 0.01
F_ANV = 0.5 if MID else float(__import__("os").environ.get("F_ANV", 1.0))
def phi(x, zh):
    xs = x - F_ANV * t_sh * sh.D
    ga = (xs - C_a).norm(dim=1) - R_ANV
    gh = (x + HS * sh.D - (torch.tensor([0, 0, Rs]) + torch.stack((0 * zh, 0 * zh, zh)))).norm(dim=1) - R_HEAD
    da = eps * torch.nn.functional.softplus(-ga / eps)
    dh = eps * torch.nn.functional.softplus(-gh / eps)
    return 0.5 * (k_n * (da**2 + dh**2)).sum(), (ga, gh)
c_w = np.sqrt(E_ / (RHO * (1 - NU**2)))
h_el = float(torch.sqrt(2 * sh.area).mean())
dt = float(__import__("os").environ.get("DTF", 0.2)) * h_el / c_w
u = torch.zeros(N, 3); v = torch.zeros(N, 3); R = torch.eye(3).expand(N, 3, 3).clone(); w = torch.zeros(N, 3)
zh, vh = torch.zeros(()), torch.tensor(-v0)
m = sh.m[:, None]; j = sh.j[:, None]
def acc(u, R, zh):
    fu, fth, _ = sh.internal_force(u, R)
    xx = (sh.X + u).detach().requires_grad_(True); z = zh.detach().requires_grad_(True)
    with torch.enable_grad():
        P, (ga, gh) = phi(xx, z)
        gx, gz = torch.autograd.grad(P, (xx, z))
    return (-gx - fu) / m, -fth / j, -gz / M, ga, gh, -gx
a, al, ah, ga, gh, fc = acc(u, R, zh)
hist = []
t = 0.0; nsteps = int(T / dt)
rs.update('run', 0, nsteps)
t0 = time.time()
for n in range(nsteps):
    v = v + 0.5 * dt * a; w = w + 0.5 * dt * al; vh = vh + 0.5 * dt * ah
    u = u + dt * v; R = rodrigues(dt * w) @ R; zh = zh + dt * vh; t += dt
    a, al, ah, ga, gh, fc = acc(u, R, zh)
    v = v + 0.5 * dt * a; w = w + 0.5 * dt * al; vh = vh + 0.5 * dt * ah
    if n % 10 == 0:
        Fa = float(-(torch.tensor(0.) + 0).item()) if False else 0.0
        hist.append((t, float(zh), float(vh), float(u[apex, 2]), float(0.5 * (m[:, 0] * (v**2).sum(1)).sum() + 0.5 * M * vh**2),
                     float(ga.min()), float(gh.min())))
    if n % 500 == 0:
        rs.update('run', n, nsteps)
H = np.array(hist)
print('dt', dt, 'steps', nsteps, 'wall', time.time() - t0, 'nodes', N, 'tris', len(tri))
for x in (0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
    k = min(range(len(H)), key=lambda i: abs(H[i, 0] - x * 1e-3))
    print(f't={x:4.2f} ms head U3 {H[k,1]:8.4f} V3 {H[k,2]:8.1f} apex U3 {H[k,3]:8.4f} KE {H[k,4]:7.2f} g_anv_min {H[k,5]:.4f} g_head_min {H[k,6]:.4f}')
np.save(f'abaqus/results/{name}_ours_F{F_ANV:g}.npy', H)
