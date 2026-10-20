# Plateau-problem gradient check — joint `none`

Actual plateau-design problem: example3(elemSize=3) lattice, Midplane, EPU 45 ViscoHyper, F-delta MSE objective (80 N plateau), design = 3 group radii mapped to all rows; frozen-schedule objective (nominal adaptive dt replayed). adjoint = checkpointed discrete adjoint; AD = unrolled reverse autograd; FD = central differences, h = 1e-5 (normalised x). dir = unit directions of the 3 groups and 3 random directions over all design rows.

| T [ms] | start (3 groups) | J | adjoint vs AD (max rel) | dir | adjoint·d | FD·d | rel err (adj vs FD) |
|---|---|---|---|---|---|---|---|
| 3.5 | [0.15, 0.55, 0.9] | 1.9114e-01 | n/a (full drop) | group 0 | -8.603642e-01 | -8.603623e-01 | 2.2e-06 |
| 3.5 | [0.15, 0.55, 0.9] | 1.9114e-01 | n/a (full drop) | group 1 | -1.561831e-01 | -1.561823e-01 | 4.7e-06 |
| 3.5 | [0.15, 0.55, 0.9] | 1.9114e-01 | n/a (full drop) | group 2 | -1.258700e-01 | -1.258696e-01 | 3.1e-06 |
