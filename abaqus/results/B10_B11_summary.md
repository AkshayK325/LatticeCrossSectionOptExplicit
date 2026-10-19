# B10 / B11 / B9d blind predictions (current FHS geometry, linear, prescribed plate motion)

Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams; error vs solid when it exists.

| case | model | end of ramp [N] | hold [N] | vs bare | error vs solid |
|---|---|---|---|---|---|
| B10a | bare beams | 10.78 | 10.82 | 1.000 | -49.4 % |
| B10a | zone k = 3.5 | 16.34 | 16.63 | 1.536 | -22.2 % |
| B10a | JOINT_FHS | 16.84 | 17.12 | 1.582 | -19.9 % |
| B10a | rule R5 | 16.14 | 16.39 | 1.514 | -23.3 % |
| B10a | **Abaqus solid** | 21.68 | 21.38 | 1.975 |  |
| B10b | bare beams | 5.273 | 4.18 | 1.000 | -31.6 % |
| B10b | zone k = 3.5 | 6.312 | 5.524 | 1.321 | -9.6 % |
| B10b | JOINT_FHS | 6.944 | 5.929 | 1.418 | -3.0 % |
| B10b | rule R5 | 6.214 | 5.462 | 1.306 | -10.6 % |
| B10b | **Abaqus solid** | 6.545 | 6.111 | 1.462 |  |
| B10c | bare beams | 9.194 | 8.18 | 1.000 | -46.9 % |
| B10c | zone k = 3.5 | 12.01 | 12.1 | 1.479 | -21.4 % |
| B10c | JOINT_FHS | 12.63 | 12.88 | 1.575 | -16.3 % |
| B10c | rule R5 | 12.77 | 12.88 | 1.574 | -16.3 % |
| B10c | **Abaqus solid** | 15.69 | 15.39 | 1.882 |  |
| B10d | bare beams | 9.023 | 8.188 | 1.000 | -46.6 % |
| B10d | zone k = 3.5 | 11.55 | 12.14 | 1.483 | -20.8 % |
| B10d | JOINT_FHS | 12.22 | 12.91 | 1.577 | -15.7 % |
| B10d | rule R5 | 12.34 | 12.91 | 1.576 | -15.7 % |
| B10d | **Abaqus solid** | 15.67 | 15.32 | 1.871 |  |
| B11a | bare beams | 42.27 | 42.5 | 1.000 | -14.3 % |
| B11a | zone k = 3.5 | 47.27 | 47.46 | 1.117 | -4.3 % |
| B11a | JOINT_FHS | 45.32 | 45.38 | 1.068 | -8.5 % |
| B11a | rule R5 | 46.76 | 46.78 | 1.101 | -5.7 % |
| B11a | **Abaqus solid** | 49.59 | 49.61 | 1.167 |  |
| B11b | bare beams | 14.8 | 14.69 | 1.000 | -21.2 % |
| B11b | zone k = 3.5 | 17.44 | 17.32 | 1.179 | -7.1 % |
| B11b | JOINT_FHS | 17.68 | 17.59 | 1.198 | -5.6 % |
| B11b | rule R5 | 19.34 | 19.29 | 1.313 | +3.5 % |
| B11b | **Abaqus solid** | 18.74 | 18.64 | 1.269 |  |
| B11c | bare beams | 42.28 | 42.92 | 1.000 | -18.2 % |
| B11c | zone k = 3.5 | 50.54 | 51.02 | 1.188 | -2.8 % |
| B11c | JOINT_FHS | 47.22 | 47.57 | 1.108 | -9.3 % |
| B11c | rule R5 | 50.77 | 51.18 | 1.192 | -2.5 % |
| B11c | **Abaqus solid** | 52.35 | 52.47 | 1.222 |  |
| B11d | bare beams | 14.12 | 14.13 | 1.000 | -31.5 % |
| B11d | zone k = 3.5 | 17.7 | 17.77 | 1.258 | -13.8 % |
| B11d | JOINT_FHS | 18.79 | 18.85 | 1.334 | -8.6 % |
| B11d | rule R5 | 20.83 | 20.82 | 1.474 | +0.9 % |
| B11d | **Abaqus solid** | 20.51 | 20.63 | 1.460 |  |
| B9d | bare beams | 5.002 | 2.377 | 1.000 | -29.0 % |
| B9d | zone k = 3.5 | 5.454 | 3.416 | 1.437 | +2.0 % |
| B9d | JOINT_FHS | 5.676 | 3.248 | 1.367 | -3.0 % |
| B9d | rule R5 | 5.471 | 3.498 | 1.472 | +4.5 % |
| B9d | **Abaqus solid** | 4.867 | 3.349 | 1.409 |  |
| B10e | bare beams | 10.78 | 10.82 | 1.000 | -49.8 % |
| B10e | zone k = 3.5 | 16.34 | 16.63 | 1.536 | -22.9 % |
| B10e | JOINT_FHS | 16.84 | 17.12 | 1.582 | -20.6 % |
| B10e | rule R5 | 16.14 | 16.39 | 1.514 | -24.0 % |
| B10e | **Abaqus solid** | 21.91 | 21.58 | 1.993 |  |
| B10f | bare beams | 10.78 | 10.82 | 1.000 | -48.5 % |
| B10f | zone k = 3.5 | 16.34 | 16.63 | 1.536 | -20.9 % |
| B10f | JOINT_FHS | 16.84 | 17.12 | 1.582 | -18.5 % |
| B10f | rule R5 | 16.14 | 16.39 | 1.514 | -22.0 % |
| B10f | **Abaqus solid** | 21.24 | 21.01 | 1.941 |  |
