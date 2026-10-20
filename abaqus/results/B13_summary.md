# B13: C4 unit cell with decoupled radii of the long (6.35 mm) and short (2.63 mm) struts (current FHS geometry, linear, prescribed plate motion)

Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams; error vs solid when it exists.

| case | r long | r short | model | hold [N] | vs bare | error vs solid |
|---|---|---|---|---|---|---|
| B10b | 0.6 | 0.6 | bare beams | 4.18 | 1.000 | -31.6 % |
| B10b | 0.6 | 0.6 | zone k = 3.5 | 5.524 | 1.321 | -9.6 % |
| B10b | 0.6 | 0.6 | JOINT_FHS (lambda = r) | 5.929 | 1.418 | -3.0 % |
| B10b | 0.6 | 0.6 | JOINT_FHS (no decay) | 6.235 | 1.492 | +2.0 % |
| B10b | 0.6 | 0.6 | rule R5 (set 3) | 5.462 | 1.306 | -10.6 % |
| B10b | 0.6 | 0.6 | rule R5 (set 4 refit) | 5.73 | 1.371 | -6.2 % |
| B10b | 0.6 | 0.6 | rule R5 + short-strut law (gain 1.5, l0 4.6) | 5.538 | 1.325 | -9.4 % |
| B10b | 0.6 | 0.6 | **Abaqus solid** | 6.111 | 1.462 |  |
| B13a | 0.6 | 0.8 | bare beams | 5.137 | 1.000 | -40.5 % |
| B13a | 0.6 | 0.8 | zone k = 3.5 | 6.56 | 1.277 | -24.1 % |
| B13a | 0.6 | 0.8 | JOINT_FHS (lambda = r) | 7.583 | 1.476 | -12.2 % |
| B13a | 0.6 | 0.8 | JOINT_FHS (no decay) | 7.987 | 1.555 | -7.6 % |
| B13a | 0.6 | 0.8 | rule R5 (set 3) | 7 | 1.363 | -19.0 % |
| B13a | 0.6 | 0.8 | rule R5 (set 4 refit) | 7.179 | 1.398 | -16.9 % |
| B13a | 0.6 | 0.8 | rule R5 + short-strut law (gain 1.5, l0 4.6) | 8.817 | 1.716 | +2.1 % |
| B13a | 0.6 | 0.8 | **Abaqus solid** | 8.639 | 1.682 |  |
| B13b | 0.6 | 1 | bare beams | 5.583 | 1.000 | -53.6 % |
| B13b | 0.6 | 1 | zone k = 3.5 | 8.563 | 1.534 | -28.8 % |
| B13b | 0.6 | 1 | JOINT_FHS (lambda = r) | 10.35 | 1.854 | -14.0 % |
| B13b | 0.6 | 1 | JOINT_FHS (no decay) | 10.97 | 1.964 | -8.9 % |
| B13b | 0.6 | 1 | rule R5 (set 3) | 10.3 | 1.844 | -14.4 % |
| B13b | 0.6 | 1 | rule R5 (set 4 refit) | 10.54 | 1.888 | -12.4 % |
| B13b | 0.6 | 1 | rule R5 + short-strut law (gain 1.5, l0 4.6) | 12.94 | 2.318 | +7.5 % |
| B13b | 0.6 | 1 | **Abaqus solid** | 12.03 | 2.155 |  |
| B10a | 0.8 | 0.8 | bare beams | 10.82 | 1.000 | -49.4 % |
| B10a | 0.8 | 0.8 | zone k = 3.5 | 16.63 | 1.536 | -22.2 % |
| B10a | 0.8 | 0.8 | JOINT_FHS (lambda = r) | 17.12 | 1.582 | -19.9 % |
| B10a | 0.8 | 0.8 | JOINT_FHS (no decay) | 18.49 | 1.708 | -13.5 % |
| B10a | 0.8 | 0.8 | rule R5 (set 3) | 16.39 | 1.514 | -23.3 % |
| B10a | 0.8 | 0.8 | rule R5 (set 4 refit) | 17.7 | 1.635 | -17.2 % |
| B10a | 0.8 | 0.8 | rule R5 + short-strut law (gain 1.5, l0 4.6) | 21.03 | 1.942 | -1.7 % |
| B10a | 0.8 | 0.8 | **Abaqus solid** | 21.38 | 1.975 |  |
| B13c | 0.8 | 0.6 | bare beams | 8.15 | 1.000 | -52.2 % |
| B13c | 0.8 | 0.6 | zone k = 3.5 | 14.16 | 1.738 | -17.0 % |
| B13c | 0.8 | 0.6 | JOINT_FHS (lambda = r) | 15.12 | 1.855 | -11.4 % |
| B13c | 0.8 | 0.6 | JOINT_FHS (no decay) | 16.66 | 2.044 | -2.4 % |
| B13c | 0.8 | 0.6 | rule R5 (set 3) | 14.78 | 1.814 | -13.4 % |
| B13c | 0.8 | 0.6 | rule R5 (set 4 refit) | 15.5 | 1.902 | -9.2 % |
| B13c | 0.8 | 0.6 | rule R5 + short-strut law (gain 1.5, l0 4.6) | 15.6 | 1.914 | -8.6 % |
| B13c | 0.8 | 0.6 | **Abaqus solid** | 17.07 | 2.094 |  |
| B13d | 0.8 | 1 | bare beams | 13.19 | 1.000 | -52.4 % |
| B13d | 0.8 | 1 | zone k = 3.5 | 19.58 | 1.484 | -29.3 % |
| B13d | 0.8 | 1 | JOINT_FHS (lambda = r) | 22.01 | 1.669 | -20.6 % |
| B13d | 0.8 | 1 | JOINT_FHS (no decay) | 23.67 | 1.794 | -14.6 % |
| B13d | 0.8 | 1 | rule R5 (set 3) | 20.1 | 1.524 | -27.5 % |
| B13d | 0.8 | 1 | rule R5 (set 4 refit) | 21.61 | 1.638 | -22.0 % |
| B13d | 0.8 | 1 | rule R5 + short-strut law (gain 1.5, l0 4.6) | 27.25 | 2.066 | -1.7 % |
| B13d | 0.8 | 1 | **Abaqus solid** | 27.71 | 2.101 |  |
