# B9: octet r 1.0 / 1.5 and C12 r 0.5 unit cells, prescribed plate motion (current FHS geometry, linear)

Reaction = sum RF3 at the bottom plate, hold value [N] (solid: mean over the last 70 % of the hold); stiffening = / bare beams.

| case | model | end of ramp [N] | hold [N] | vs bare | error vs solid |
|---|---|---|---|---|---|
| B9a | bare beams | 126.8 | 127.5 | 1.000 | -37.2 % |
| B9a | zone k = 3.5 | 184.5 | 185.4 | 1.454 | -8.7 % |
| B9a | JOINT_FHS | 189 | 188.7 | 1.480 | -7.1 % |
| B9a | rule R1 | 199.5 | 199.8 | 1.567 | -1.6 % |
| B9a | rule R2 | 221 | 220.2 | 1.727 | +8.4 % |
| B9a | rule R3 | 232.2 | 231.1 | 1.812 | +13.8 % |
| B9a | **Abaqus solid** | 203.3 | 203 | 1.592 |  |
| B9b | bare beams | 322.7 | 324 | 1.000 | -58.9 % |
| B9b | zone k = 3.5 | 596.7 | 595.8 | 1.839 | -24.3 % |
| B9b | JOINT_FHS | 621 | 618.1 | 1.908 | -21.5 % |
| B9b | rule R1 | 704 | 703.9 | 2.173 | -10.6 % |
| B9b | rule R2 | 871.6 | 871.7 | 2.691 | +10.7 % |
| B9b | rule R3 | 978.1 | 975.8 | 3.012 | +23.9 % |
| B9b | **Abaqus solid** | 786.1 | 787.5 | 2.431 |  |
| B9c | bare beams | 5.002 | 0.8098 | 1.000 | -69.5 % |
| B9c | zone k = 3.5 | 5.454 | 2.533 | 3.128 | -4.5 % |
| B9c | JOINT_FHS | 5.678 | 1.854 | 2.289 | -30.1 % |
| B9c | rule R1 | 5.445 | 3.026 | 3.737 | +14.1 % |
| B9c | rule R2 | 5.443 | 2.872 | 3.547 | +8.3 % |
| B9c | rule R3 | 5.487 | 2.821 | 3.484 | +6.4 % |
| B9c | **Abaqus solid** | 4.867 | 2.653 | 3.276 |  |
