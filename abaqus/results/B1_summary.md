# B1: single EPU 45 strut, prescribed ramp-and-hold, beam (ours) vs Abaqus solid

Ours with the Prony branches relaxing the nominal axial force (current) or the Kirchhoff-type force (candidate). Abaqus columns appear once the run is extracted. Reaction = sum RF at the bottom; work = int RF dU during the ramp.

| case | law | peak reaction [N] | at end of ramp [N] | at end of hold [N] | ramp work [N mm] |
|---|---|---|---|---|---|
| B1a | ours, nominal | 155 | 81.96 | 21.16 | -56.87 |
| B1a | ours, kirchhoff | 139.3 | 72.56 | 21.16 | -53.12 |
| B1b | ours, nominal | 31.38 | 21.38 | 21.16 | -12.27 |
| B1b | ours, kirchhoff | 28.18 | 21.32 | 21.16 | -11.27 |
| B1c | ours, nominal | 76.45 | -33.45 | -11 | -34.09 |
| B1c | ours, kirchhoff | 81.76 | -36.65 | -11 | -35.88 |
| B1d | ours, nominal | 16.88 | 7.866 | 2.385 | -1.181 |
| B1d | ours, kirchhoff | 16.64 | 7.717 | 2.385 | -1.169 |
| B1e | ours, nominal | 1.037 | -0.5877 | -0.1051 | -0.5073 |
| B1e | ours, kirchhoff | 1.037 | -0.5877 | -0.1051 | -0.5073 |
| B1f | ours, nominal | 0.1491 | -0.1054 | -0.1046 | -0.1217 |
| B1f | ours, kirchhoff | 0.1491 | -0.1054 | -0.1046 | -0.1217 |
| B1g | ours, nominal | 2.463 | -1.108 | -0.3399 | -0.9324 |
| B1g | ours, kirchhoff | 2.463 | -1.108 | -0.3399 | -0.9324 |
| B1e2 | ours, nominal | 1.037 | -0.5877 | -0.1051 | -0.5073 |
| B1e2 | ours, kirchhoff | 1.037 | -0.5877 | -0.1051 | -0.5073 |
