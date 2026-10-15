# B7 / B8: column + oblique strut, centre node valence 4 / 6 (current FHS geometry, linear, prescribed motion)

Reaction = sum RF3 at the bottom plate [N]. For the Abaqus rows the hold column is the MEAN over the last 70 % of the hold (the linear solid rings by +-6 % after the ramp); stiffening = hold value / bare beams hold value.

| case | model | end of ramp [N] | hold [N] (solid: mean) | stiffening vs bare (hold) |
|---|---|---|---|---|
| B7a | bare beams | 43.91 | 44.45 | 1.000 |
| B7a | JOINT_FHS | 52.3 | 52.89 | 1.190 |
| B7a | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 44 | 44.53 | 1.002 |
| B7a | rule without angle weight (p = 0, a = 1.30) | 50.9 | 51.52 | 1.159 |
| B7a | **Abaqus solid** | 57.29 | 57.44 | 1.292 |
| B7b | bare beams | 42.9 | 43.08 | 1.000 |
| B7b | JOINT_FHS | 48.62 | 48.64 | 1.129 |
| B7b | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 43.57 | 43.75 | 1.016 |
| B7b | rule without angle weight (p = 0, a = 1.30) | 48.94 | 48.94 | 1.136 |
| B7b | **Abaqus solid** | 53.11 | 53.01 | 1.231 |
| B7c | bare beams | 42.58 | 42.76 | 1.000 |
| B7c | JOINT_FHS | 46.7 | 46.64 | 1.091 |
| B7c | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 44.46 | 44.66 | 1.045 |
| B7c | rule without angle weight (p = 0, a = 1.30) | 47.3 | 47.39 | 1.108 |
| B7c | **Abaqus solid** | 49.68 | 49.62 | 1.160 |
| B8a | bare beams | 31.14 | 31.18 | 1.000 |
| B8a | JOINT_FHS | 32.6 | 32.84 | 1.053 |
| B8a | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 32.22 | 32.38 | 1.038 |
| B8a | rule without angle weight (p = 0, a = 1.30) | 32.68 | 32.93 | 1.056 |
| B8a | **Abaqus solid** | 36.2 | 34.67 | 1.112 |
| B8b | bare beams | 46.7 | 46.77 | 1.000 |
| B8b | JOINT_FHS | 51.49 | 51.16 | 1.094 |
| B8b | rule M3 (p = 4.10, zl = 0.71, b = 0.53) | 54.73 | 53.93 | 1.153 |
| B8b | rule without angle weight (p = 0, a = 1.30) | 50.8 | 51.05 | 1.091 |
| B8b | **Abaqus solid** | 56.45 | 54.18 | 1.158 |
