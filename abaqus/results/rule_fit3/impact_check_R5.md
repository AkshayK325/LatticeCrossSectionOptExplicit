# Impact check of rule R5 (results/rule_fit3/R5.json; a 3.45, zl 0.77, gamma 0, b 0.15, p -0.77, wsat 2.47; mu = 1) on the FHS impact solids

Never fitted on impact data. Only the FHS solids F1 / H1 are valid for an FHS-fitted rule (the B2 solids have cylinder + sphere joints). CAVEAT: all F / H solids were built with the pre-2026-10-11 FHS geometry, whose BCC cell is 11-13 % SOFTER than the current geometry (B4 vs B6); a rule that matches the current geometry will look over-stiff against them. Treat as a consistency check, not a validation.

error = ours / solid - 1 [%]: max stroke / energy at 50 % crush / 75 % / peak force; baseline = JOINT_FHS re-run with the current code.

| case | JOINT_FHS | rule R5 |
|---|---|---|
| F1a | -7.3 / +3.7 / +7.1 / +20.3 | -5.3 / -2.9 / +1.8 / +32.0 |
| F1c | -0.4 / -7.8 / -0.4 / +12.9 | +1.6 / -13.4 / -5.6 / +18.0 |
| F1d | -5.1 / +4.2 / +8.7 / +4.0 | -9.0 / +11.3 / +14.0 / +40.6 |
| F1e | -2.2 / -11.4 / -3.4 / +26.2 | +0.8 / -16.3 / -11.4 / +27.8 |
| F1f | -2.0 / -6.4 / -4.3 / -1.4 | +2.0 / -15.1 / -13.0 / +2.5 |
| H1a | -5.1 / +1.3 / +2.9 / +4.8 | -0.9 / -3.9 / -1.7 / +4.2 |
| H1c | +0.7 / -2.2 / -1.2 / -5.1 | +5.0 / -6.7 / -5.3 / +3.1 |
| H1d | -6.6 / +5.9 / +6.3 / -4.8 | -11.8 / +11.8 / +11.8 / -9.9 |
| H1e | -12.6 / +9.9 / +11.9 / +26.4 | -8.8 / +4.5 / +7.2 / +18.5 |
| H1f | -7.1 / +5.8 / +6.9 / +8.1 | -1.9 / -0.7 / +0.7 / +2.7 |
| **mean abs, C4 (F1a, F1c, H1a, H1c)** | 3.4 / 3.8 / 2.9 / 10.8 | 3.2 / 6.7 / 3.6 / 14.3 |
| **mean abs, BCC (old geometry: F1e, F1f, H1e, H1f)** | 6.0 / 8.4 / 6.6 / 15.5 | 3.4 / 9.2 / 8.1 / 12.9 |
| **mean abs, C12 (F1d, H1d)** | 5.8 / 5.1 / 7.5 / 4.4 | 10.4 / 11.6 / 12.9 / 25.2 |
