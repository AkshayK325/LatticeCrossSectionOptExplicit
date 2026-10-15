# Impact hold-out of the second rule (R3 of fit_rule2.py: a 5.69, zl 0.74, gamma 0, b 0.19, p -0.81; mu = 1), never fitted on impact data

error = ours / solid - 1 [%]: max stroke / energy at 50 % crush / at 75 % / peak force; baseline = recorded joint model (JOINT_SV for B2, JOINT_FHS for F / H) re-run with the current code. B2h E75 is nan (solid and ours differ in the crush reached).

| case | baseline | rule R3 |
|---|---|---|
| B2a | +2.1 / -6.7 / -3.5 / +1.3 | -2.4 / +0.7 / +1.1 / +17.8 |
| B2c | +1.2 / -8.4 / -3.2 / -7.9 | -2.2 / -5.4 / +1.6 / +5.0 |
| B2d | -3.2 / +4.0 / +7.7 / -1.7 | -13.0 / +23.2 / +26.5 / +23.3 |
| B2e | +0.9 / -12.8 / -1.6 / +9.7 | -0.2 / -8.9 / -1.2 / +2.3 |
| B2f | -0.2 / -5.4 / -3.1 / -1.8 | -3.8 / +1.1 / -0.1 / +15.5 |
| B2g | -1.3 / +9.7 / -4.8 / -1.5 | -16.4 / +16.4 / +35.3 / +3.9 |
| B2h | +1.0 / +2.5 / -0.5 / +9.5 | -28.1 / +59.6 / nan / +69.0 |
| F1a | -7.3 / +3.7 / +7.1 / +20.3 | -6.5 / -1.5 / +3.9 / +32.5 |
| F1c | -0.4 / -7.8 / -0.4 / +12.9 | +0.7 / -12.0 / -4.1 / +20.3 |
| F1d | -5.1 / +4.2 / +8.7 / +4.0 | -10.3 / +14.4 / +17.0 / +38.8 |
| F1e | -2.2 / -11.4 / -3.4 / +26.2 | +0.3 / -14.3 / -10.7 / +14.9 |
| F1f | -2.0 / -6.4 / -4.3 / -1.4 | +0.2 / -13.6 / -10.1 / +7.9 |
| H1a | -5.1 / +1.3 / +2.9 / +4.8 | -2.6 / -2.2 / -0.0 / +6.5 |
| H1c | +0.7 / -2.2 / -1.2 / -5.1 | +3.7 / -5.5 / -4.2 / +4.3 |
| H1d | -6.6 / +5.9 / +6.3 / -4.8 | -13.7 / +14.1 / +13.9 / -6.7 |
| H1e | -12.6 / +9.9 / +11.9 / +26.4 | -9.9 / +6.0 / +8.5 / +14.1 |
| H1f | -7.1 / +5.8 / +6.9 / +8.1 | -4.7 / +2.8 / +4.0 / +9.6 |
| **mean abs, C4 / BCC cells (12)** | 3.5 / 6.8 / 4.1 / 10.5 | 3.1 / 6.2 / 4.1 / 12.6 |
| **mean abs, C12 / octet (5: B2d, B2g, B2h, F1d, H1d)** | 3.4 / 5.3 / 5.6 / 4.3 | 16.3 / 25.5 / 23.2 / 28.3 |
| **mean abs, all (17)** | 3.5 / 6.4 / 4.6 / 8.7 | 7.0 / 11.9 / 8.9 / 17.2 |

Gain scan (R3 with a bounded instead of 5.69; prescribed rms over the 10 specimens; short-strut impact cells stroke / E50): a = 1.5: 4.1 % (max 6.0); B2d -9.6 / +16, B2g -11.5 / +18, B2h -16.4 / +19, F1d -6.9 / +9, H1d -9.0 / +8.
a = 3.0: 2.7 % (max 5.0); B2d -11.8 / +21, B2g -14.5 / +19, B2h -23.5 / +42, F1d -9.1 / +12, H1d -12.0 / +12. A bounded gain does not remove the over-stiffening of C12 / octet: it comes from the zone length and the number of acute neighbours, not from the gain.
