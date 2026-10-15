# B6: full BCC r 1.0 cell with the current FHS geometry vs B4 (old geometry), prescribed plate motion

Reaction = sum RF3 at the bottom plate; stiffening = end of ramp / bare beams (ours).

| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | vs bare |
|---|---|---|---|---|---|---|
| B6a | ours, bare beams | 405 | 401 | 401.9 | 201.6 | 1.000 |
| B6a | ours, JOINT_FHS | 502.1 | 500.9 | 492.3 | 250.2 | 1.249 |
| B6a | ours, JOINT_FHS + plate_layer | 456.8 | 456.2 | 449.6 | 227.4 | 1.138 |
| B6a | ours, simple rule a = 1 | 527 | 525.2 | 518.4 | 262.6 | 1.310 |
| B6a | **Abaqus solid (current geometry)** | 472.8 | 456 | 463.4 | 241.6 | 1.137 |
| B6a | **Abaqus solid (B4, old geometry)** | 416.8 | 408.9 | 402.2 | 218.7 | 1.020 |
| B6b | ours, bare beams | 133.1 | 62.98 | 18.23 | 89.5 | 1.000 |
| B6b | ours, JOINT_FHS | 164.8 | 77.66 | 22.74 | 112.1 | 1.233 |
| B6b | ours, JOINT_FHS + plate_layer | 149.9 | 70.55 | 20.58 | 101.4 | 1.120 |
| B6b | ours, simple rule a = 1 | 174.6 | 82.51 | 24.11 | 118.6 | 1.310 |
| B6b | **Abaqus solid (current geometry)** | 147.6 | 66.93 | 19.78 | 102.9 | 1.063 |
| B6b | **Abaqus solid (B4, old geometry)** | 133.2 | 59.23 | 17.61 | 93.92 | 0.940 |
