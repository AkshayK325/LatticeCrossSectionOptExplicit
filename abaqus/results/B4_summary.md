# B4: full BCC r 1.0 cell (H1e solid), prescribed plate motion

Reaction = sum RF3 at the bottom plate; work = |int RF dU| during the ramp; stiffening = reaction at the end of the ramp / bare beams. JOINT_FHS (old geometry) uses the joint factors of the FHS geometry the solid was built from (before 2026-10-11); (current) those of the present FHS code.

| case | model | peak [N] | end of ramp [N] | end of hold [N] | ramp work [N mm] | stiffening vs bare |
|---|---|---|---|---|---|---|
| B4a | ours, bare beams | 405 | 401 | 401.9 | 201.6 | 1.000 |
| B4a | ours, JOINT_FHS (old geometry) | 496.2 | 494.8 | 486.7 | 247.2 | 1.234 |
| B4a | ours, JOINT_FHS (current) | 502.1 | 500.9 | 492.3 | 250.2 | 1.249 |
| B4b | ours, bare beams | 133.1 | 62.98 | 18.23 | 89.5 | 1.000 |
| B4b | ours, JOINT_FHS (old geometry) | 162.8 | 76.68 | 22.44 | 110.7 | 1.218 |
| B4b | ours, JOINT_FHS (current) | 164.8 | 77.66 | 22.74 | 112.1 | 1.233 |
| B4c | ours, bare beams | 25.88 | 18.36 | 18.23 | 19.2 | 1.000 |
| B4c | ours, JOINT_FHS (old geometry) | 31.89 | 22.6 | 22.45 | 23.64 | 1.231 |
| B4c | ours, JOINT_FHS (current) | 32.32 | 22.91 | 22.75 | 23.95 | 1.248 |
