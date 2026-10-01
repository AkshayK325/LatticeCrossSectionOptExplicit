"""Run the unrolled-AD vs checkpointed-adjoint benchmark matrix, then (only when every case has finished)
write a numbered figure to ResultsFig/explicit/ and the results table to notes/7_AdjointBenchmark.md.
    PYTHONPATH=src:tests python tests/run_adjoint_benchmark.py
"""
import json
import pathlib
import subprocess
import sys
import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
CASES = [('unrolled', 'cpu', 3.0, 4e-4), ('adjoint', 'cpu', 3.0, 4e-4),
         ('unrolled', 'cpu', 3.0, 1e-3), ('adjoint', 'cpu', 3.0, 1e-3),
         ('adjoint', 'cpu', 3.0, 2.5e-3), ('adjoint', 'cpu', 3.0, 8e-3),
         ('unrolled', 'cuda', 3.0, 4e-4), ('adjoint', 'cuda', 3.0, 4e-4),
         ('unrolled', 'cuda', 3.0, 1e-3), ('adjoint', 'cuda', 3.0, 1e-3),
         ('adjoint', 'cuda', 3.0, 2.5e-3), ('adjoint', 'cuda', 3.0, 8e-3),
         ('adjoint', 'cuda', 1.0, 8e-3)]

rows = []
for m, dev, es, T in CASES:
    p = subprocess.run([PY, str(ROOT / 'tests' / 'study_adjoint_benchmark.py'), m, dev, str(es), str(T)],
                       capture_output=True, text=True, env={'PYTHONPATH': f"{ROOT/'src'}:{ROOT/'tests'}", 'PATH': '/usr/bin:/bin'})
    line = [l for l in p.stdout.splitlines() if l.startswith('{')]
    if line:
        rows.append(json.loads(line[-1]))
    else:
        err = [l for l in (p.stdout + p.stderr).splitlines() if 'Error' in l]
        rows.append(dict(method=m, device=dev, elemSize=es, T_ms=T * 1e3, failed=(err[-1] if err else 'unknown')[:200]))
    print(json.dumps(rows[-1]), flush=True)

# ---------------------------------------------------------------- report (after all runs)
sys.path[:0] = [str(ROOT / 'src')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotFrame import numbered

ok = [r for r in rows if 'failed' not in r]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for (m, dev), mk in ((('unrolled', 'cpu'), 'o--'), (('adjoint', 'cpu'), 'o-'), (('unrolled', 'cuda'), 's--'), (('adjoint', 'cuda'), 's-')):
    rr = sorted([r for r in ok if r['method'] == m and r['device'] == dev and r['elemSize'] == 3.0], key=lambda r: r['steps'])
    if not rr:
        continue
    lab = f'{m} ({dev})'
    mem = [r['peak_gpu_GB'] if dev == 'cuda' else r['peak_rss_GB'] for r in rr]
    ax[0].plot([r['steps'] for r in rr], mem, mk, label=lab)
    ax[1].plot([r['steps'] for r in rr], [r['wall_s'] for r in rr], mk, label=lab)
ax[0].set(xscale='log', yscale='log', xlabel='time steps N', ylabel='peak memory [GB] (CPU: process RSS, GPU: allocator)',
          title='memory vs horizon (3 mm mesh)')
ax[1].set(xscale='log', yscale='log', xlabel='time steps N', ylabel='wall time, J + dJ/dθ [s]', title='time vs horizon')
for a in ax:
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which='both')
fig.tight_layout()
figpath = numbered(ROOT / 'ResultsFig' / 'explicit', 'adjoint_benchmark.png')
fig.savefig(figpath, dpi=130)
json.dump(rows, open(numbered(ROOT / 'ResultsFig' / 'explicit', 'adjoint_benchmark.json'), 'w'), indent=1)


def match(r):
    """Gradient agreement with the unrolled case of the same device/mesh/horizon, if it exists."""
    u = [q for q in ok if q['method'] == 'unrolled' and q['device'] == r['device'] and q['elemSize'] == r['elemSize'] and q['steps'] == r['steps']]
    if r['method'] != 'adjoint' or not u:
        return ''
    a, b = r['g'], u[0]['g']
    return f"{max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip(a, b)):.1e}"


lines = ['# 7 — Adjoint benchmark: unrolled reverse AD vs checkpointed discrete adjoint', '',
         f'Generated automatically by `tests/run_adjoint_benchmark.py` after all runs finished ({datetime.datetime.now():%Y-%m-%d %H:%M}).',
         'Drop rig: EPU 45, r = 0.8 mm (Midplane symmetry, 3 design groups), 0.75 J, 6 m/s, fibre visco-hyperelastic beams,',
         'line + plate contact, friction μ = 0.5; placeholder objective (gradient verification only). K = √N checkpoints.', '',
         f'![benchmark](../ResultsFig/explicit/{figpath.name})', '',
         '| method | device | elemSize | horizon | steps | elements | wall [s] | peak RSS [GB] | peak GPU [GB] | grad rel. diff vs unrolled |',
         '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    if 'failed' in r:
        lines.append(f"| {r['method']} | {r['device']} | {r['elemSize']} | {r['T_ms']:.1f} ms | — | — | **failed** | — | — | {r['failed']} |")
    else:
        lines.append(f"| {r['method']} | {r['device']} | {r['elemSize']} | {r['T_ms']:.1f} ms | {r['steps']} | {r['ne']} | {r['wall_s']} | "
                     f"{r['peak_rss_GB']} | {r.get('peak_gpu_GB', '—')} | {match(r)} |")
(ROOT / 'notes' / '7_AdjointBenchmark.md').write_text('\n'.join(lines) + '\n')
print('report:', figpath, ROOT / 'notes' / '7_AdjointBenchmark.md')
