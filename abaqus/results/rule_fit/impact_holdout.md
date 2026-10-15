# Impact hold-out of the closed-form joint rule (not used in any fit)

Ours energy-matched to each Abaqus solid; error = ours / solid - 1 [%]: max stroke, energy absorbed at 50 / 75 % of the crush, peak force.
Baseline = the joint model recorded for the case (JOINT_SV for B2, JOINT_FHS for F / H), re-run with the current code. Rule = M3 fit (results/rule_fit/M3.json) with the EPU factor mu = 1 for H
(the fitted mu = 0.17-0.41 describes the fast prescribed ramps and made the C4 H cells 14-16 % too soft in stroke). F / H BCC solids (F1e/f, H1e/f) were built with the pre-2026-10-11 FHS geometry and are unreliable for stiffness.

| case | baseline stroke / E50 / E75 / peak | rule M3 stroke / E50 / E75 / peak |
|---|---|---|
| B2a | +2.1 / -6.7 / -3.5 / +1.3 | +5.5 / -11.9 / -9.2 / +6.8 |
| B2c | +1.2 / -8.4 / -3.2 / -7.9 | +3.3 / -13.6 / -7.2 / -5.9 |
| B2d | -3.2 / +4.0 / +7.7 / -1.7 | -5.2 / +8.5 / +10.3 / +21.3 |
| B2e | +0.9 / -12.8 / -1.6 / +9.7 | +0.8 / -13.5 / -0.7 / +7.2 |
| B2f | -0.2 / -5.4 / -3.1 / -1.8 | -0.6 / -6.4 / -1.1 / +1.2 |
| B2g | -1.3 / +9.7 / -4.8 / -1.5 | -1.6 / +11.1 / -7.7 / -13.9 |
| B2h | +1.0 / +2.5 / -0.5 / +9.5 | +1.0 / +0.9 / -1.2 / -2.7 |
| F1a | -7.3 / +3.7 / +7.1 / +20.3 | +1.1 / -11.4 / -8.0 / +21.1 |
| F1c | -0.4 / -7.8 / -0.4 / +12.9 | +6.4 / -17.9 / -12.1 / +6.2 |
| F1d | -5.1 / +4.2 / +8.7 / +4.0 | -2.4 / -0.9 / +4.8 / +26.9 |
| F1e | -2.2 / -11.4 / -3.4 / +26.2 | +1.0 / -18.0 / -11.3 / +19.6 |
| F1f | -2.0 / -6.4 / -4.3 / -1.4 | +3.1 / -18.4 / -14.9 / -5.6 |
| H1a | -5.1 / +1.3 / +2.9 / +4.8 | +7.9 / -11.9 / -9.3 / -5.8 |
| H1c | +0.7 / -2.2 / -1.2 / -5.1 | +11.3 / -11.8 / -10.3 / -9.1 |
| H1d | -6.6 / +5.9 / +6.3 / -4.8 | -2.7 / +1.7 / +2.3 / -17.4 |
| H1e | -12.6 / +9.9 / +11.9 / +26.4 | -8.4 / +3.4 / +6.6 / +16.4 |
| H1f | -7.1 / +5.8 / +6.9 / +8.1 | +0.0 / -3.8 / -1.7 / +1.8 |
| **mean abs, B2 (7)** | 1.4 / 7.1 / 3.5 / 4.8 | 2.6 / 9.4 / 5.3 / 8.4 |
| **mean abs, F1 (5)** | 3.4 / 6.7 / 4.8 / 13.0 | 2.8 / 13.3 / 10.2 / 15.9 |
| **mean abs, H1 (5)** | 6.4 / 5.0 / 5.8 / 9.8 | 6.1 / 6.5 / 6.0 / 10.1 |
| **mean abs, all (17)** | 3.5 / 6.4 / 4.6 / 8.7 | 3.7 / 9.7 / 7.0 / 11.1 |
