# B5: simple cubic cells (no internal joint), prescribed plate motion

Reaction = sum RF3 at the bottom plate; work = |int RF dU| during the ramp; stiffening = reaction at the end of the ramp / bare beams. Current FHS geometry for the solid and the joint factors.

| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | stiffening vs bare |
|---|---|---|---|---|---|---|
| B5a | ours, bare beams | 170.1 | 169.4 | 169.8 | 42.41 | 1.000 |
| B5a | ours, JOINT_FHS | 187.8 | 187.5 | 187.6 | 46.83 | 1.107 |
| B5a | ours, JOINT_FHS + plate_layer | 172.3 | 171.7 | 172.2 | 42.99 | 1.014 |
| B5a | **Abaqus solid** | 191.7 | 189.4 | 189.5 | 47.03 | 1.118 |
| B5b | ours, bare beams | 53.18 | 24.42 | 7.424 | 18.43 | 1.000 |
| B5b | ours, JOINT_FHS | 58.72 | 27.03 | 8.215 | 20.38 | 1.107 |
| B5b | ours, JOINT_FHS + plate_layer | 53.88 | 24.76 | 7.527 | 18.68 | 1.014 |
| B5b | **Abaqus solid** | 56.55 | 25.99 | 7.907 | 19.67 | 1.064 |
| B5c | ours, bare beams | 382.6 | 381.1 | 381.9 | 95.41 | 1.000 |
| B5c | ours, JOINT_FHS | 445.2 | 444.1 | 444.2 | 111.1 | 1.165 |
| B5c | ours, JOINT_FHS + plate_layer | 390.6 | 389.5 | 390.4 | 97.48 | 1.022 |
| B5c | **Abaqus solid** | 453.2 | 452.9 | 452.2 | 111.2 | 1.188 |
| B5d | ours, bare beams | 382.6 | 381.1 | 381.9 | 95.41 | 1.000 |
| B5d | ours, JOINT_FHS | 428.9 | 428.1 | 428 | 107 | 1.123 |
| B5d | ours, JOINT_FHS + plate_layer | 389.3 | 388.3 | 389.2 | 97.17 | 1.019 |
| B5d | **Abaqus solid** | 445.2 | 442.1 | 442.3 | 109.1 | 1.160 |
