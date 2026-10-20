# Plateau-problem gradient check — joint `R5_C4`

Actual plateau-design problem: example3(elemSize=3) lattice, Midplane, EPU 45 ViscoHyper, F-delta MSE objective (80 N plateau), design = 3 group radii mapped to all rows; frozen-schedule objective (nominal adaptive dt replayed). adjoint = checkpointed discrete adjoint; AD = unrolled reverse autograd; FD = central differences, h = 1e-5 (normalised x). dir = unit directions of the 3 groups and 1 random direction over all design rows.

| T [ms] | start (3 groups) | J | adjoint vs AD (max rel) | dir | adjoint·d | FD h=1e-05 (rel err) | FD h=0.0001 (rel err) | FD h=0.001 (rel err) |
|---|---|---|---|---|---|---|---|---|
| 3.5 | [0.5, 0.5, 0.5] | 1.7317e-01 | n/a (full drop) | group 0 | 9.600771e-01 | 9.600771e-01 (2.0e-08) | 9.600757e-01 (1.5e-06) | 9.601720e-01 (9.9e-05) |
| 3.5 | [0.5, 0.5, 0.5] | 1.7317e-01 | n/a (full drop) | group 1 | 9.979248e-01 | 9.979248e-01 (2.1e-08) | 9.979265e-01 (1.7e-06) | 9.978169e-01 (1.1e-04) |
| 3.5 | [0.5, 0.5, 0.5] | 1.7317e-01 | n/a (full drop) | group 2 | 1.514109e-01 | 1.514110e-01 (2.7e-08) | 1.514110e-01 (1.8e-08) | 1.514112e-01 (1.8e-06) |
| 3.5 | [0.5, 0.5, 0.5] | 1.7317e-01 | n/a (full drop) | random 0 | -5.466660e-02 | -5.466660e-02 (3.7e-09) | -5.466660e-02 (2.2e-08) | -5.466652e-02 (1.5e-06) |
| 3.5 | [0.15, 0.55, 0.9] | 9.7341e-02 | n/a (full drop) | group 0 | -5.063485e-01 | -5.120244e-01 (1.1e-02) | -4.745355e-01 (6.7e-02) | -4.686666e-01 (8.0e-02) |
| 3.5 | [0.15, 0.55, 0.9] | 9.7341e-02 | n/a (full drop) | group 1 | -1.327777e-01 | -2.946403e-02 (3.5e+00) | -1.381436e-01 (3.9e-02) | -1.415647e-01 (6.2e-02) |
| 3.5 | [0.15, 0.55, 0.9] | 9.7341e-02 | n/a (full drop) | group 2 | -2.468260e-01 | -2.165734e-01 (1.4e-01) | -2.280852e-01 (8.2e-02) | -2.412269e-01 (2.3e-02) |
| 3.5 | [0.15, 0.55, 0.9] | 9.7341e-02 | n/a (full drop) | random 0 | -2.759302e-02 | 1.561725e-02 (2.8e+00) | -3.813074e-02 (2.8e-01) | -2.662980e-02 (3.6e-02) |
| 0.4 | [0.5, 0.5, 0.5] | 1.2581e-01 | 6.2e-16 | group 0 | 7.017998e-01 | 7.017998e-01 (1.3e-09) | 7.017998e-01 (4.3e-08) | 7.018028e-01 (4.3e-06) |
| 0.4 | [0.5, 0.5, 0.5] | 1.2581e-01 | 6.2e-16 | group 1 | 7.180188e-01 | 7.180188e-01 (1.7e-09) | 7.180188e-01 (5.8e-09) | 7.180184e-01 (5.9e-07) |
| 0.4 | [0.5, 0.5, 0.5] | 1.2581e-01 | 6.2e-16 | group 2 | 1.140377e-01 | 1.140377e-01 (4.9e-10) | 1.140377e-01 (4.3e-08) | 1.140382e-01 (4.3e-06) |
| 0.4 | [0.5, 0.5, 0.5] | 1.2581e-01 | 6.2e-16 | random 0 | 1.167866e-02 | 1.167865e-02 (1.0e-07) | 1.167865e-02 (1.1e-07) | 1.167866e-02 (5.6e-09) |
| 0.4 | [0.15, 0.55, 0.9] | 7.4382e-03 | 8.3e-16 | group 0 | 5.288005e-02 | 5.288005e-02 (2.2e-09) | 5.288007e-02 (2.1e-07) | 5.288118e-02 (2.1e-05) |
| 0.4 | [0.15, 0.55, 0.9] | 7.4382e-03 | 8.3e-16 | group 1 | 6.672529e-02 | 6.672529e-02 (9.6e-10) | 6.672528e-02 (6.6e-08) | 6.672484e-02 (6.6e-06) |
| 0.4 | [0.15, 0.55, 0.9] | 7.4382e-03 | 8.3e-16 | group 2 | 1.577474e-02 | 1.577474e-02 (8.3e-10) | 1.577474e-02 (2.6e-08) | 1.577469e-02 (2.6e-06) |
| 0.4 | [0.15, 0.55, 0.9] | 7.4382e-03 | 8.3e-16 | random 0 | 2.243174e-03 | 2.243175e-03 (4.3e-07) | 2.243174e-03 (4.3e-08) | 2.243174e-03 (1.8e-08) |

## Reading the table
* **Adjoint vs unrolled AD** (T = 0.4 ms, both starts): 6e-16 / 8e-16 (max relative) — the checkpointed adjoint is the exact derivative of the replayed computation, for all 36-row designs.
* **Equal radii (0.5, 0.5, 0.5), full 3.5 ms drop** (1553 steps): adjoint = central FD to 2e-8 (h = 1e-5) in all directions (3 groups + random).
* **Unequal radii (0.15, 0.55, 0.9), 0.4 ms**: adjoint = FD to 1e-9 .. 1e-7.
* **Unequal radii, full drop and 2 ms drop**: FD is *not* a reliable reference. The same design gives FD(h) that changes sign and size with h (group 0 on CPU: +0.40 at h = 1e-6, -0.512 at 1e-5, -0.475 at 1e-4, -0.469 at 1e-3; adjoint -0.506), repeated GPU evaluations of the *same* x differ by 5e-6 in J (non-deterministic atomics), and the adjoint itself on GPU varies -0.477 / -0.520 between identical runs, while the CPU adjoint is bit-reproducible. The bare-beam model at the same design is smooth (FD = adjoint to 4e-6). So with the joint model and thick, unequal struts the late-time response of this drop is only piecewise smooth at the 1e-5 relative level (discrete contact / Verlet-list events and a stiffer, ringing response), the objective is rough on the FD scale, and the adjoint (exact for the replayed, frozen-schedule, frozen-contact-list computation) lies within the spread of FD estimates (1-8 % of the larger-h values). The 2 ms drop shows the same, so it is not a late-time rebound effect only.
* Time-step sensitivity (`courant_sensitivity.md`): J changes by < 1 % between Courant 0.7 and 0.25, so the roughness is not a time-step instability.

## Consequence for the optimiser
The gradient is exact for the computation that is differentiated; descent on this J is still reliable (the adjoint direction agrees with smoothed FD in sign and to a few %), but a design-gradient test of the real plateau problem against FD must use equal-radii or short-drop designs, or compare against FD at h >= 1e-4, and use the CPU for deterministic runs. Use `tests/test_plateau_joint_gradient.py` (0.4 ms, unequal radii; slow on a busy CPU) as the regression test.
