# B7 / B8: column + oblique strut, centre node valence 4 / 6 (current FHS geometry, linear, prescribed motion)

Reaction = sum RF3 at the bottom plate, end of the ramp [N]; stiffening = / bare beams (ours).

| case | model | end of ramp [N] | end of hold [N] | stiffening vs bare |
|---|---|---|---|---|
| B7a | bare beams | 69.93 | 69.95 | 1.000 |
| B7a | JOINT_FHS | 85.61 | 85.59 | 1.224 |
| B7a | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 70.16 | 70.17 | 1.003 |
| B7a | rule without angle weight (p = 0, a = 1.30) | 97.5 | 97.51 | 1.394 |
| B7b | bare beams | 57.4 | 57.39 | 1.000 |
| B7b | JOINT_FHS | 66.43 | 66.42 | 1.157 |
| B7b | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 58.85 | 58.86 | 1.025 |
| B7b | rule without angle weight (p = 0, a = 1.30) | 71.78 | 71.74 | 1.250 |
| B7c | bare beams | 47.7 | 47.71 | 1.000 |
| B7c | JOINT_FHS | 53.54 | 53.51 | 1.122 |
| B7c | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 50.88 | 50.85 | 1.067 |
| B7c | rule without angle weight (p = 0, a = 1.30) | 55.89 | 55.86 | 1.172 |
| B8a | bare beams | 31.14 | 31.18 | 1.000 |
| B8a | JOINT_FHS | 32.6 | 32.84 | 1.047 |
| B8a | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 32.22 | 32.38 | 1.035 |
| B8a | rule without angle weight (p = 0, a = 1.30) | 32.68 | 32.93 | 1.050 |
| B8b | bare beams | 46.7 | 46.77 | 1.000 |
| B8b | JOINT_FHS | 51.49 | 51.16 | 1.102 |
| B8b | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 54.73 | 53.93 | 1.172 |
| B8b | rule without angle weight (p = 0, a = 1.30) | 50.8 | 51.05 | 1.088 |
