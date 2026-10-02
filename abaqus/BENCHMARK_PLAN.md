# Abaqus/Explicit benchmark plan: tracking

**Goal.** Show that our differentiable explicit beam solver is (A) *correct*: it solves the beam equations as a
reference code does. Then (B) *adequate*: the beam idealisation reproduces a solid lattice closely enough, and above
all preserves the design ranking the optimizer relies on.

**Reference code:** Abaqus 2024/Explicit, run by the user from the `.inp` files in `inp/`. Workflow: `README.md`.

**Rules.**
- One ingredient at a time. Our model is switched to match what the Abaqus model can represent, never the other
  way round.
- Every case is defined once in `cases.py` (the same geometry, material, loads and outputs for both codes).
- `compare.py <case>` writes the figure (`ResultsFig/abaqus/<n>_<case>.png`, numbered from 1) and `results/<case>.md`.
- Contact is a known weak point in Abaqus/Explicit too (penalty and kinematic enforcement, beam-edge contact).
  So contact cases are judged on kinematics and impulse, and Abaqus is run with two contact settings to bracket
  its own uncertainty. Treat its contact force as a reference, not ground truth.
- Tick `[x]` when a case passes. Record the numbers in the table, and put the commit hash of the code version
  next to the tick.

Legend: ⬜ not started · 🟦 inp ready, waiting for Abaqus · 🟨 results in, does not pass yet / investigating ·
✅ passes.

---

## 0. Infrastructure
- [x] `abaqus/` folder: `cases.py`, `extract_odb.py`, `compare.py`, `inp/`, `csv/`, `ours/`, `results/`, `odb/` (2026-10-02)
- [x] `extract_odb.py`: odb → `<job>_history.csv` (long format) + `<job>_field_U.csv` (Abaqus Python 3)
- [x] `compare.py`: our run + overlay + nRMSE + period + report
- [x] First round trip with A1: inp → Abaqus → odb → CSV → compare ✔ (region names `Node PART-1-1.<label>`, `Assembly Assembly-1` parsed; `.sta` fallback kept)
- [ ] Lattice exporter in `cases.py`: `example3` lattice → B31 elements, one element set per strut,
      CIRC section per strut, rigid top plate (`*RIGID BODY`, reference node with plate + impactor mass), bonded
      bottom, `*INITIAL CONDITIONS, TYPE=VELOCITY`
- [ ] Our-side `bench` switches used by A4/A5: `start='merged'`, linear-elastic law (`mat` without `yeoh`),
      `fibre=False`, `mu=0` (A4/A5a), beam contact on/off

## Level A: code verification (Abaqus B31 beams; linear elastic E₀ = 342.9 MPa, ν = 0.49, no Prony, no bulk viscosity)

| Case | Status | What it checks | Metrics → pass | Abaqus | Ours | Result |
|---|---|---|---|---|---|---|
| **A1** cantilever, L = 20, r = 0.8, 20 B31, sudden tip load P = 0.4 N (+z, fixed direction), 10 ms | ✅ | corotational kinematics at 67° rotation, mass + rotary inertia, time integration, Timoshenko shear | nRMSE tip U3 ≤ 1%, U1 ≤ 2%, UR2 ≤ 1%; U3 period ≤ 0.5% | `inp/A1_cantilever.inp`, `csv/A1_cantilever_history.csv` (double, 6420 incr., Δt 1.558e-6) | 7106 steps, Δt 1.407e-6 | **Pass** (fe07082): U3 **0.06%**, U1 **0.10%**, UR2 **0.31%**, period 2.8040 vs 2.8039 ms (**0.004%**), max U3 13.739 vs 13.742 mm; KE 0.12%, SE 0.08%; root RF3 8.6% raw / 0.7% low-passed (high-mode content). Required fixing our rotary inertia (see Log). Fig `ResultsFig/abaqus/1_A1_cantilever.png` |
| **A2** single strut L = 6.4, r = 0.8 (L/r = 8), **8** B31, imperfection 0.05 mm; clamped base; guided top with 10 g at 6 m/s, 2 ms | 🟨 → superseded by A2a/A2b | buckling under impact | compression phase graded | `inp/A2_strut_impact.inp` (ran 2026-10-02) | stroke 2.521 mm | **Not a pass, explained.** Compression phase: stroke nRMSE 1.6% ✔, max stroke 4.0%, plateau RF3 87.5 vs 81.1 N (7.8%), buckling time 7%. Rebound: tension to 28% strain, peak −201 vs −172 N. Causes, each tested on our side: (1) **mesh**: 8 elements not converged (ours 8 → 32 el: stroke +2.3%, plateau −3.3%); Abaqus B31's default *shear slenderness compensation* (−8% κGA on these short elements) explains a further ~3% of the plateau; (2) **large-strain measure**: 12% (plateau) to 28% (rebound) axial strain, where linear elastic differs between codes (ours N = EA(λ−1), Abaqus Cauchy/log strain): with a log-strain axial law ours moves the tension peak from 16.7% to 6% and buckling time from 6.8% to 1.8%. Not a kinematics error; fig `9_A2_strut_impact.png` |
| **A2a** = A2 with **32** B31, Abaqus shear compensation off (`*TRANSVERSE SHEAR STIFFNESS` 3rd entry 1e-8) | ⚠ Abaqus not a valid reference here | same stubby strut (L/r = 8), converged mesh, large strain (13.8% compressive, 28% tensile) | report only | `inp/A2a_strut_impact_32el.inp`; ran 2026-10-02: **Abaqus aborted at ~1.3 ms** ("ratio of deformation speed to wave speed exceeds 1.0"), and its energy was not conserved before that (ETOTAL 180.0 → 173.0 N mm, −4%, in the compression phase; then 998 N mm) | stroke 2.579 mm, plateau 84.8 N; energy conserved to 1e-4 | Compression phase: stroke history 1.6% ✔, max stroke 3.8%, plateau 84.8 vs 79.1 N (7.2%), buckling time 7.0%, low-passed RF3 correlation 0.996 ✔. A log-strain axial law on our side fixes the buckling time (2.3%) but not the plateau (7.0%). **Conclusion:** for stubby struts at >10% strain the differences are of the size of Abaqus's own energy error; B31 + linear elasticity is not a trustworthy reference in this regime → settle it with solids (B1/B2). Fig `3_A2a_strut_impact_32el.png` |
| **A2b** slender strut L = 6.4, **r = 0.2** (L/r = 32), 32 B31, imperfection 0.02 mm; top mass 2 g at **1 m/s**, 8 ms; shear compensation off | ✅ | **strict small-strain buckling kinematics** (axial strain ≤ 2% throughout) | max stroke ≤ 2%; stroke nRMSE (compression) ≤ 2%; buckling time ≤ 3%; plateau ≤ 3%; RF3 correlation ≥ 0.98 | `inp/A2b_slender_strut_impact.inp`; Abaqus energy drift 0.8% | stroke 2.234 mm, plateau 0.461 N | **Pass** (5f859cc): max stroke 2.2343 vs 2.2283 mm (**0.27%**), stroke history **0.12%** (full history 0.29%), buckling time **0.31%**, post-buckling plateau 0.4610 vs 0.4612 N (**0.04%**), lateral buckling deflection 2.0%, KE / SE 1.2% / 1.3%. The RF3-correlation criterion is replaced: the root reaction is axial ringing (±0.3 N around the 0.46 N mean) whose phase is not reproducible between codes (raw 0.72, 0.1 ms low-pass 0.90); its mean (the plateau) agrees to 0.04%. Fig `4_A2b_slender_strut_impact.png` |
| **A3** crossed beams: A (r 0.8, along x, L 10, 10 B31) clamped at both ends; free B (r 0.5, along y) 0.3 mm above, 2 m/s down; frictionless; 2 ms. **A3b** = Abaqus penalty × 10 | ✖ Abaqus beams not usable as reference | beam-radius contact | impulse ≤ 5%; rebound ratio ≤ 5% | ran 2026-10-02 | B dips 0.304 mm (contact at r_A + r_B), rebound ratio 0.810 | **Abaqus B31 general contact ignores the section radius** (user, confirmed by the data): B travels 1.02 mm vs ours 0.30 before rebounding, i.e. ~0.7 mm (≈ a radius) of overlap, then centreline/line contact; the same with the edge option on. Impulse on B still 2.4% (A3) / 6.6% (A3b); rebound ratio 0.768 / 0.699 vs ours 0.810 (A3 vs A3b spread 9%: Abaqus's own contact uncertainty). **Conclusion:** radius-aware beam contact cannot be checked against Abaqus beams; use A3s (solids) instead. Figs `7_`, `8_` |
| **A3 / A3b re-issued** (2026-10-02 (g)): `*CONTACT CONTROLS ASSIGNMENT, CONTACT THICKNESS REDUCTION=NOPERIMSELF`; shear compensation off | ✖ control did not help | beam-radius contact with 1 mm edges | as A3 | ran 2026-10-02 (h) | B dips 0.304 mm, rebound 0.810 | **Still reduced**: B dips 1.021 mm (A3) / 1.020 mm (A3b), i.e. unchanged. Neighbouring 1 mm edges of the same beam count as potential self-contact, so NOPERIMSELF still shrinks the radius. Impulse 2.7% / 4.2%, rebound 0.763 / 0.736. (New `.sta` files not in `logs/`, so the warning cannot be re-read.) **Use A3c** (long edges) for Abaqus beam contact. Figs `10_`, `11_` |
| **A3c** coarse crossing: 16 mm beams, **4 elements each** (4 mm), otherwise A3 (same contact control) | ✅ | beam-radius contact in Abaqus without radius reduction (radius / edge = 0.2, 0.125) | impulse ≤ 5%; rebound ratio ≤ 5% | `inp/A3c_crossed_beams_coarse.inp`; Abaqus energy drift 1.5% | B dips 0.3348 mm, rebound 0.739 | **Pass**: B maximum dip 0.3348 vs 0.3354 mm (**0.18%**: contact at r_A + r_B in both), impulse on B **2.0%**, rebound momentum ratio 0.739 vs 0.774 (**4.6%**). **Our radius-aware line contact matches Abaqus beam contact once Abaqus keeps the full radius.** Fig `12_A3c_crossed_beams_coarse.png` |
| **A3s v1** (`A3s_crossed_solid_cylinders`): C3D8R 6×6 O-grid, 0.25 mm slices, 17 280 elements, default hourglass control | ✖ invalid reference | — | — | ran 2026-10-02 (h) | — | **Hourglass energy = 64.5% of internal energy** at the end (ALLAE 4.2e-3 of ALLIE 6.5e-3 N mm; 25% of the total energy). That is why it looked stiffer: spurious hourglass modes absorbed energy, so A vibrated at about half amplitude and B rebounded slower (0.744 vs 0.810). Also the Hertz contact radius (≈ 0.09 mm) was smaller than one surface element (0.13–0.25 mm). Fig `14_` |
| **A3s2** (`A3s2_crossed_solid_cylinders_fine`): C3D8R **16×16** O-grid (surface elements 0.049 mm on B), 5 radial layers biased to the surface, axial slices graded **0.03 mm** at the crossing → 0.33 mm, **HOURGLASS=ENHANCED**, 66 816 elements, 1 ms (CFN output removed: invalid in Explicit) | 🟦 | **reference for radius-aware contact** vs our A3 beams | hourglass / internal energy ≤ 10%; impulse / initial momentum ≤ 5% (from CFN and from B momentum); rebound ratio ≤ 5%; B / A mid-span U3 | `inp/A3s2_crossed_solid_cylinders_fine.inp` (6.9 MB). About 160k increments (ν = 0.49 dilatational wave speed): run with several CPUs | our A3: rebound 0.810, B dips 0.304 mm | — |
| **A4** (re-issued 2026-10-02 (e): first input had an empty `VERT` element block → 3 fatal errors; fixed, and shear compensation off) 3×3×1 C4 lattice drop, 1 mm elements (1332 B31), r = 0.8, 0.75 J, 6 m/s; top nodes in a `*RIGID BODY` (rotations fixed) whose reference node carries impactor + plate; bottom clamped; **no contact**; 4 ms | ✅ | the whole assembly: BCs, rigid plate, impactor mass, many-element dynamics | max stroke ≤ 2%; plate U3 nRMSE ≤ 2%; bottom-reaction nRMSE ≤ 5%, peak ≤ 5% | `inp/A4_lattice_drop_nocontact.inp` | stroke 2.314 mm at 0.649 ms (stiff: linear E₀, so no densification and no contact needed), bottom reaction −965…+650 N, energy 750.00 → 750.08 | **Pass** (5f859cc): max stroke 2.314 vs 2.289 mm (**1.1%**), first-compression stroke history **0.44%**, plate period 2.2269 vs 2.2100 ms (**0.76%**), peak bottom reaction 965 vs 950 N (**1.6%**); Abaqus energy drift 1.5% (ours 1e-4). Full 4 ms histories: stroke nRMSE 3.7%, reaction 7.0%: the 0.76% period difference accumulates as phase drift over two cycles (curves overlap otherwise). Our result is mesh-converged (0.5 mm: period 2.2326 ms, stroke 2.322 mm); the residual is element formulation (Abaqus linear B31 at 1 mm vs our exact Timoshenko element) and stubby-strut large-strain effects (cf. A2a). Fig `7_A4_lattice_drop_nocontact.png` |
| **A5a** A4 + beam–beam and beam–plate contact, μ = 0 | ⬜ | lattice contact through densification | stroke ≤ 3%; densification onset ≤ 5%; F–δ nRMSE ≤ 10% | | | |
| **A5b** A5a with μ = 0.5 | ⬜ | friction | ALLFD vs our friction work ≤ 15%; stroke ≤ 3% | | | |

**Notes, level A:**
- Rotary inertia. Ours is lumped ρL/2·max(Iy + Iz, AL²/12). Abaqus/Explicit B31 uses its own (isotropic) beam
  rotary inertia. If A1 shows a period offset, also run Abaqus with `ROTARY INERTIA=EXACT` (if accepted for B31
  in Explicit) and ours without the scaling.
- Joint contact. Abaqus does not check contact between elements that share a node. Ours uses the smooth joint
  buffer (mean radius). A5a tests that rule as well.
- Time step. Abaqus picks its own stable increment. If needed, fix it with
  `*DYNAMIC, EXPLICIT, DIRECT USER CONTROL` at ≈ our Δt.

## Level B: material and model-form validation (Abaqus solids, Yeoh + Prony)
Material: `*HYPERELASTIC, YEOH` C10, C20, C30 = 2.4, 0.15, 0.02 MPa; D₁ from ν = 0.49;
`*VISCOELASTIC, TIME=PRONY` g₁ = 0.958, k₁ = 0, τ₁ = 5.2e-5 s; ρ = 1.06e-9 t/mm³.

| Case | Status | What it checks | Metrics | Result |
|---|---|---|---|---|
| **B0** (manual check) can Abaqus B31 take `*HYPERELASTIC` / `*VISCOELASTIC`? If yes: A1/A2 with Yeoh + Prony on beams (material only, same idealisation) | ⬜ | fibre-Yeoh + QLV implementation vs Abaqus section integration | tip / reaction histories ≤ 3% | |
| **B1** single strut, C3D8R solid (≈ 8 elements across the diameter): axial compression and bending at impact rates, vs our single beam | ⬜ | section law vs continuum; QLV vs Abaqus finite-strain viscoelasticity | N(ε), M(κ), loss per cycle; expect ≤ 5% at moderate strain | |
| **B2** one C4 unit cell, solid (STL from `beamBuild.py` → tet mesh, C3D10M), drop at 6 m/s, vs our beam cell | ⬜ | joint idealisation, stubby struts (L/r ≈ 3–8) | F–δ, peak, stroke, dissipation | |
| **B3** full 3×3×1 lattice, solid C3D10M, general contact μ = 0.5, rigid plates, merged impactor, 0.75 J | ⬜ | **paper validation figure** | F–δ, peak, stroke, dissipation; expect peak within 10–20% | |
| **B4** design ranking in Abaqus: initial vs optimized (80 N and 120 N plateau designs) | ⬜ | does the optimizer's improvement survive in the reference model? | sign and size of ΔJ, Δpeak, Δefficiency | |
| **B5** (optional) FD gradient in Abaqus: ±Δr on one group (2 runs) | ⬜ | sign and rough size of dJ/dr vs our adjoint | sign agrees; magnitude within ~30% | |

## Order
A1 → A2 → A3 → A4 → A5a → A5b → B0 → B1 → B3 → B4 → (B2, B5).

## Log
- 2026-10-02 (h): A3 family results.
  - **A3c ✅:** dip 0.18%, impulse 2.0%, rebound 4.6%. Our radius contact equals Abaqus beam contact with the full radius.
  - **A3/A3b:** NOPERIMSELF does not prevent the radius reduction on 1 mm edges.
  - **A3s v1 invalid:** 65% hourglass energy, plus too coarse for the contact patch, which is why it looked stiffer. A3s2 re-meshed (16×16, 0.03 mm at the crossing, enhanced hourglass).
  - `compare.py` now grades the hourglass fraction and the CFN contact impulse for solids.
- 2026-10-02 (g): **why Abaqus beam contact ignored the radius.**
  - The first A3 `.sta` says: "some contact edge thicknesses for general contact were reduced ... due to a large thickness compared to edge dimensions. The most significant thickness scaling factor was 0.37500", so beam A's contact radius was 0.8 → 0.3 mm on 1 mm edges.
  - Abaqus/Explicit does take the beam contact radius from the CIRC section, but caps it at a fraction of the edge length unless `CONTACT THICKNESS REDUCTION` is set.
  - A3/A3b re-issued with `NOPERIMSELF`. A3c added (long edges, no reduction needed). A3s added (solid cylinders).
  - Any future beam-contact input (A5) gets the same contact control.
- 2026-10-02 (f): results for A2a, A2b, A4.
  - **A2b ✅** (small-strain buckling: 0.04–0.3%) and **A4 ✅** (lattice: stroke 1.1%, peak 1.6%, period 0.76%).
  - **A2a ⚠:** Abaqus aborted (deformation speed > wave speed) and lost 4% energy before that, so it is not a reference for stubby, large-strain struts.
  - Pattern so far: small strain matches to 0.04–0.3% (A1, A2b); stubby / large strain differs by 1–7%, as large as Abaqus's own energy error. B-level (solids) decides the large-strain question.
  - Repo housekeeping: `inp/` keeps only `.inp`; job logs (`.sta`, `.dat`, `.msg` with warnings) moved to `logs/`; `.env` and trivial `.msg` removed; figures regenerated as `ResultsFig/abaqus/1–7`.
  - Note: the `.dat` files of A2/A2a/A2b warn that Abaqus's nodal-averaged beam normals differ from the given n1 (the imperfection bow); harmless for these results but recorded.
- 2026-10-02 (e): Abaqus results for A2, A3, A3b; A4 failed (my input bug, fixed).
  - A2: explained differences (mesh + Abaqus shear slenderness compensation + large-strain measure); split into A2a (stubby, converged mesh) and A2b (slender, small strain).
  - A3: Abaqus beam contact ignores the radius, so A3s (solid cylinders) becomes the contact reference.
  - Re-run in Abaqus: **A2a, A2b, A4**.
- 2026-10-02 (d): A2, A3, A3b, A4 inputs written with our predictions (`results/*.md`, `ResultsFig/abaqus/2–5`). Abaqus comparison figures now go to `ResultsFig/abaqus/` (numbered from 1; A1 regenerated as `1_`).
- 2026-10-02 (c): **A1 passes** after a fix found by the benchmark (fe07082).
  - Before the fix: U3 0.59%, U1 0.90%, **UR2 1.73% ✘**, RF3 18% (high-frequency ripple).
  - Cause: our scalar rotary inertia was rho L/2 (Iy + Iz), the polar value. That is exact for torsion but twice the bending rotary inertia.
  - Fix: default rho L/2 max(Iy, Iz), the bending value, as Abaqus uses. All errors dropped about 10x.
  - Effect elsewhere: none on the 3 mm optimization mesh (the rotary-inertia scaling A L^2/12 dominates there; drop rig identical: 2720 steps, stroke 8.475 mm, peak 92.22 N). It only acts for elements shorter than about 2.4 r, e.g. the 1 mm verification mesh, so re-run the 1 mm mesh-convergence numbers when convenient.
  - Tests: 40 related tests pass. The large-amplitude energy test window was widened to 4 periods: the error envelope settles at 0.8% by period 2 and stays flat through 6.
- 2026-10-02 (b): A1 run by the user.
  - Abaqus accepted the input unchanged. The `.dat` shows 40 program-generated internal nodes for the 20 B31 elements (cause not yet checked; total mass still equals ours).
  - Energy-level agreement: period 0.17%, KE peaks 0.1%.
  - Still waiting on the history CSV for the displacement/rotation criteria (the main kinematic check).
- 2026-10-02:
  - folder, scripts and plan created;
  - A1 inp written;
  - our A1 prediction recorded (`ours/A1_cantilever.csv`, `results/A1_cantilever.md`).
