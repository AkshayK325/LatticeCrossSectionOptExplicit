# B9: octet r 1.0 / 1.5 and C12 r 0.5 unit cells, prescribed plate motion (current FHS geometry, linear)

Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams.

| case | model | end of ramp [N] | hold [N] | vs bare | error vs solid |
|---|---|---|---|---|---|
| B9a | bare beams | 126.8 | 127.5 | 1.000 |  |
| B9a | zone k = 3.5 | 184.5 | 185.4 | 1.454 |  |
| B9a | JOINT_FHS | 189 | 188.7 | 1.480 |  |
| B9a | rule R1 | 199.5 | 199.8 | 1.567 |  |
| B9a | rule R2 | 221 | 220.2 | 1.727 |  |
| B9a | rule R3 | 232.2 | 231.1 | 1.812 |  |
| B9b | bare beams | 322.7 | 324 | 1.000 |  |
| B9b | zone k = 3.5 | 596.7 | 595.8 | 1.839 |  |
| B9b | JOINT_FHS | 621 | 618.1 | 1.908 |  |
| B9b | rule R1 | 704 | 703.9 | 2.173 |  |
| B9b | rule R2 | 871.6 | 871.7 | 2.691 |  |
| B9b | rule R3 | 978.1 | 975.8 | 3.012 |  |
| B9c | bare beams | 5.002 | 0.8098 | 1.000 |  |
| B9c | zone k = 3.5 | 5.454 | 2.533 | 3.128 |  |
| B9c | JOINT_FHS | 5.678 | 1.854 | 2.289 |  |
| B9c | rule R1 | 5.445 | 3.026 | 3.737 |  |
| B9c | rule R2 | 5.443 | 2.872 | 3.547 |  |
| B9c | rule R3 | 5.487 | 2.821 | 3.484 |  |
