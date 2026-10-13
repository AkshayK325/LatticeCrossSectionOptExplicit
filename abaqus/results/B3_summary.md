# B3: BCC centre-node joint specimen (8 diagonals, FHS joint), prescribed plate motion

Reaction = sum RF3 at the bottom plate; work = |int RF dU| during the ramp. Joint stiffening = reaction at the end of the ramp / the bare-beam value (ours: what JOINT_FHS adds; Abaqus: what the solid joint adds).

| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | stiffening vs bare (ramp end) |
|---|---|---|---|---|---|---|
| B3a | ours, bare beams | 64.98 | 62.27 | 62.37 | 31.96 | 1.000 |
| B3a | ours, JOINT_FHS | 72.56 | 72 | 69.81 | 36.35 | 1.156 |
| B3a | **Abaqus solid** | 80.13 | 78.08 | 73.33 | 39.21 | 1.254 |
| B3b | ours, bare beams | 21.77 | 10.94 | 2.692 | 13.13 | 1.000 |
| B3b | ours, JOINT_FHS | 24.51 | 12.09 | 3.041 | 15.13 | 1.105 |
| B3b | **Abaqus solid** | 24.5 | 11.8 | 3.177 | 16.18 | 1.078 |
| B3c | ours, bare beams | 3.86 | 2.709 | 2.689 | 2.94 | 1.000 |
| B3c | ours, JOINT_FHS | 4.377 | 3.066 | 3.044 | 3.339 | 1.132 |
| B3c | **Abaqus solid** | 4.586 | 3.202 | 3.176 | 3.52 | 1.182 |
