# B12: two valence-3 Y nodes joined by a strut of length Ls (current FHS geometry, linear, prescribed plate motion)

Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams; error vs solid when it exists.

| case | r | Ls [mm] | Ls / r | model | hold [N] | vs bare | error vs solid |
|---|---|---|---|---|---|---|---|
| B12a | 0.8 | 2.63 | 3.3 | bare beams | 32.08 | 1.000 |  |
| B12a | 0.8 | 2.63 | 3.3 | zone k = 3.5 | 34.28 | 1.069 |  |
| B12a | 0.8 | 2.63 | 3.3 | JOINT_FHS (lambda = r) | 33.43 | 1.042 |  |
| B12a | 0.8 | 2.63 | 3.3 | JOINT_FHS (no decay) | 33.59 | 1.047 |  |
| B12a | 0.8 | 2.63 | 3.3 | rule R5 (set 3) | 34.47 | 1.075 |  |
| B12a | 0.8 | 2.63 | 3.3 | rule R5 (set 4 refit) | 34.66 | 1.080 |  |
| B12b | 0.8 | 4.5 | 5.6 | bare beams | 30.27 | 1.000 |  |
| B12b | 0.8 | 4.5 | 5.6 | zone k = 3.5 | 32.23 | 1.065 |  |
| B12b | 0.8 | 4.5 | 5.6 | JOINT_FHS (lambda = r) | 31.54 | 1.042 |  |
| B12b | 0.8 | 4.5 | 5.6 | JOINT_FHS (no decay) | 31.69 | 1.047 |  |
| B12b | 0.8 | 4.5 | 5.6 | rule R5 (set 3) | 32.4 | 1.071 |  |
| B12b | 0.8 | 4.5 | 5.6 | rule R5 (set 4 refit) | 32.57 | 1.076 |  |
| B12c | 0.8 | 7 | 8.8 | bare beams | 28.2 | 1.000 |  |
| B12c | 0.8 | 7 | 8.8 | zone k = 3.5 | 29.92 | 1.061 |  |
| B12c | 0.8 | 7 | 8.8 | JOINT_FHS (lambda = r) | 29.35 | 1.041 |  |
| B12c | 0.8 | 7 | 8.8 | JOINT_FHS (no decay) | 29.48 | 1.045 |  |
| B12c | 0.8 | 7 | 8.8 | rule R5 (set 3) | 30.07 | 1.066 |  |
| B12c | 0.8 | 7 | 8.8 | rule R5 (set 4 refit) | 30.21 | 1.071 |  |
| B12d | 0.6 | 2.63 | 4.4 | bare beams | 17.89 | 1.000 |  |
| B12d | 0.6 | 2.63 | 4.4 | zone k = 3.5 | 18.79 | 1.051 |  |
| B12d | 0.6 | 2.63 | 4.4 | JOINT_FHS (lambda = r) | 18.46 | 1.032 |  |
| B12d | 0.6 | 2.63 | 4.4 | JOINT_FHS (no decay) | 18.52 | 1.035 |  |
| B12d | 0.6 | 2.63 | 4.4 | rule R5 (set 3) | 18.87 | 1.055 |  |
| B12d | 0.6 | 2.63 | 4.4 | rule R5 (set 4 refit) | 18.94 | 1.059 |  |
| B12e | 0.5 | 2.63 | 5.3 | bare beams | 12.27 | 1.000 |  |
| B12e | 0.5 | 2.63 | 5.3 | zone k = 3.5 | 12.78 | 1.042 |  |
| B12e | 0.5 | 2.63 | 5.3 | JOINT_FHS (lambda = r) | 12.64 | 1.030 |  |
| B12e | 0.5 | 2.63 | 5.3 | JOINT_FHS (no decay) | 12.67 | 1.033 |  |
| B12e | 0.5 | 2.63 | 5.3 | rule R5 (set 3) | 12.82 | 1.046 |  |
| B12e | 0.5 | 2.63 | 5.3 | rule R5 (set 4 refit) | 12.87 | 1.049 |  |
