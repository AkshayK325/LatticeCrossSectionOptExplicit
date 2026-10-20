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
| **A2** single strut L = 6.4, r = 0.8 (L/r = 8), **8** B31, imperfection 0.05 mm; clamped base; guided top with 10 g at 6 m/s, 2 ms | 🟨 → superseded by A2a/A2b | buckling under impact | compression phase graded | `inp/A2_strut_impact.inp` (ran 2026-10-02) | stroke 2.521 mm | **Not a pass, explained.** Compression phase: stroke nRMSE 1.6% ✔, max stroke 4.0%, plateau RF3 87.5 vs 81.1 N (7.8%), buckling time 7%. Rebound: tension to 28% strain, peak −201 vs −172 N. Causes, each tested on our side: (1) **mesh**: 8 elements not converged (ours 8 → 32 el: stroke +2.3%, plateau −3.3%); Abaqus B31's default *shear slenderness compensation* (−8% κGA on these short elements) explains a further ~3% of the plateau; (2) **large-strain measure**: 12% (plateau) to 28% (rebound) axial strain, where linear elastic differs between codes (ours N = EA(λ−1), Abaqus Cauchy/log strain): with a log-strain axial law ours moves the tension peak from 16.7% to 6% and buckling time from 6.8% to 1.8%. Not a kinematics error; (figure removed: superseded) |
| **A2a** = A2 with **32** B31, Abaqus shear compensation off (`*TRANSVERSE SHEAR STIFFNESS` 3rd entry 1e-8) | ⚠ Abaqus not a valid reference here | same stubby strut (L/r = 8), converged mesh, large strain (13.8% compressive, 28% tensile) | report only | `inp/A2a_strut_impact_32el.inp`; ran 2026-10-02: **Abaqus aborted at ~1.3 ms** ("ratio of deformation speed to wave speed exceeds 1.0"), and its energy was not conserved before that (ETOTAL 180.0 → 173.0 N mm, −4%, in the compression phase; then 998 N mm) | stroke 2.579 mm, plateau 84.8 N; energy conserved to 1e-4 | Compression phase: stroke history 1.6% ✔, max stroke 3.8%, plateau 84.8 vs 79.1 N (7.2%), buckling time 7.0%, low-passed RF3 correlation 0.996 ✔. A log-strain axial law on our side fixes the buckling time (2.3%) but not the plateau (7.0%). **Conclusion:** for stubby struts at >10% strain the differences are of the size of Abaqus's own energy error; B31 + linear elasticity is not a trustworthy reference in this regime → settle it with solids (B1/B2). (figure removed: Abaqus run invalid) |
| **A2b** slender strut L = 6.4, **r = 0.2** (L/r = 32), 32 B31, imperfection 0.02 mm; top mass 2 g at **1 m/s**, 8 ms; shear compensation off | ✅ | **strict small-strain buckling kinematics** (axial strain ≤ 2% throughout) | max stroke ≤ 2%; stroke nRMSE (compression) ≤ 2%; buckling time ≤ 3%; plateau ≤ 3%; RF3 correlation ≥ 0.98 | `inp/A2b_slender_strut_impact.inp`; Abaqus energy drift 0.8% | stroke 2.234 mm, plateau 0.461 N | **Pass** (5f859cc): max stroke 2.2343 vs 2.2283 mm (**0.27%**), stroke history **0.12%** (full history 0.29%), buckling time **0.31%**, post-buckling plateau 0.4610 vs 0.4612 N (**0.04%**), lateral buckling deflection 2.0%, KE / SE 1.2% / 1.3%. The RF3-correlation criterion is replaced: the root reaction is axial ringing (±0.3 N around the 0.46 N mean) whose phase is not reproducible between codes (raw 0.72, 0.1 ms low-pass 0.90); its mean (the plateau) agrees to 0.04%. Fig `2_A2b_slender_strut_impact.png` |
| **A3** crossed beams: A (r 0.8, along x, L 10, 10 B31) clamped at both ends; free B (r 0.5, along y) 0.3 mm above, 2 m/s down; frictionless; 2 ms. **A3b** = Abaqus penalty × 10 | ✖ Abaqus beams not usable as reference | beam-radius contact | impulse ≤ 5%; rebound ratio ≤ 5% | ran 2026-10-02 | B dips 0.304 mm (contact at r_A + r_B), rebound ratio 0.810 | **Abaqus B31 general contact ignores the section radius** (user, confirmed by the data): B travels 1.02 mm vs ours 0.30 before rebounding, i.e. ~0.7 mm (≈ a radius) of overlap, then centreline/line contact; the same with the edge option on. Impulse on B still 2.4% (A3) / 6.6% (A3b); rebound ratio 0.768 / 0.699 vs ours 0.810 (A3 vs A3b spread 9%: Abaqus's own contact uncertainty). **Conclusion:** radius-aware beam contact cannot be checked against Abaqus beams; use A3s (solids) instead. (figures removed: Abaqus radius reduced) |
| **A3 / A3b re-issued** (2026-10-02 (g)): `*CONTACT CONTROLS ASSIGNMENT, CONTACT THICKNESS REDUCTION=NOPERIMSELF`; shear compensation off | ✖ control did not help | beam-radius contact with 1 mm edges | as A3 | ran 2026-10-02 (h) | B dips 0.304 mm, rebound 0.810 | **Still reduced**: B dips 1.021 mm (A3) / 1.020 mm (A3b), i.e. unchanged. Neighbouring 1 mm edges of the same beam count as potential self-contact, so NOPERIMSELF still shrinks the radius. Impulse 2.7% / 4.2%, rebound 0.763 / 0.736. (New `.sta` files not in `logs/`, so the warning cannot be re-read.) **Use A3c** (long edges) for Abaqus beam contact. (figures removed: Abaqus radius reduced) |
| **A3c** coarse crossing: 16 mm beams, **4 elements each** (4 mm), otherwise A3 (same contact control) | ✅ | beam-radius contact in Abaqus without radius reduction (radius / edge = 0.2, 0.125) | impulse ≤ 5%; rebound ratio ≤ 5% | `inp/A3c_crossed_beams_coarse.inp`; Abaqus energy drift 1.5% | B dips 0.3348 mm, rebound 0.739 | **Pass**: B maximum dip 0.3348 vs 0.3354 mm (**0.18%**: contact at r_A + r_B in both), impulse on B **2.0%**, rebound momentum ratio 0.739 vs 0.774 (**4.6%**). **Our radius-aware line contact matches Abaqus beam contact once Abaqus keeps the full radius.** Fig `3_A3c_crossed_beams_coarse.png` |
| **A3s v1** (`A3s_crossed_solid_cylinders`): C3D8R 6×6 O-grid, 0.25 mm slices, 17 280 elements, default hourglass control | ✖ invalid reference | — | — | ran 2026-10-02 (h) | — | **Hourglass energy = 64.5% of internal energy** at the end (ALLAE 4.2e-3 of ALLIE 6.5e-3 N mm; 25% of the total energy). That is why it looked stiffer: spurious hourglass modes absorbed energy, so A vibrated at about half amplitude and B rebounded slower (0.744 vs 0.810). Also the Hertz contact radius (≈ 0.09 mm) was smaller than one surface element (0.13–0.25 mm). (figure removed: invalid reference) |
| **A3s2** (`A3s2_crossed_solid_cylinders_fine`): C3D8R **16×16** O-grid (surface elements 0.049 mm on B), 5 radial layers biased to the surface, axial slices graded **0.03 mm** at the crossing → 0.33 mm, **HOURGLASS=ENHANCED**, 66 816 elements, 1 ms (CFN output removed: invalid in Explicit) | ✅ | **reference for radius-aware contact** vs our A3 beams | hourglass / internal energy ≤ 10%; impulse / initial momentum ≤ 5% (from CFN and from B momentum); rebound ratio ≤ 5%; B / A mid-span U3 | `inp/A3s2_crossed_solid_cylinders_fine.inp` (6.9 MB). About 160k increments (ν = 0.49 dilatational wave speed): run with several CPUs | our A3: rebound 0.810, B dips 0.304 mm | **Pass**: hourglass / internal energy **5.7%** (v1: 64.5%), impulse on B / initial momentum 1.810 vs 1.823 (**0.70%**), rebound momentum ratio 0.810 vs 0.823 (**1.55%**), Abaqus energy drift 0.01%. A's post-impact vibration amplitude and frequency now match ours (v1's 'stiffer' look was hourglassing). B's axis dips 6.4% deeper in the solid (0.325 vs 0.304 mm): the local section flattening/ovalisation a beam model does not have. **Our radius-aware line contact reproduces solid-cylinder contact.** Notes: the run warned that history requests (35 944, the per-node V3 of B) exceed `max_history_requests` = 10 000, but all node histories were written; the field CSV is from the pre-64522bd input (axis nodes only). Fig `4_A3s2_crossed_solid_cylinders_fine.png` |
| **A4** (re-issued 2026-10-02 (e): first input had an empty `VERT` element block → 3 fatal errors; fixed, and shear compensation off) 3×3×1 C4 lattice drop, 1 mm elements (1332 B31), r = 0.8, 0.75 J, 6 m/s; top nodes in a `*RIGID BODY` (rotations fixed) whose reference node carries impactor + plate; bottom clamped; **no contact**; 4 ms | ✅ | the whole assembly: BCs, rigid plate, impactor mass, many-element dynamics | max stroke ≤ 2%; plate U3 nRMSE ≤ 2%; bottom-reaction nRMSE ≤ 5%, peak ≤ 5% | `inp/A4_lattice_drop_nocontact.inp` | stroke 2.314 mm at 0.649 ms (stiff: linear E₀, so no densification and no contact needed), bottom reaction −965…+650 N, energy 750.00 → 750.08 | **Pass** (5f859cc): max stroke 2.314 vs 2.289 mm (**1.1%**), first-compression stroke history **0.44%**, plate period 2.2269 vs 2.2100 ms (**0.76%**), peak bottom reaction 965 vs 950 N (**1.6%**); Abaqus energy drift 1.5% (ours 1e-4). Full 4 ms histories: stroke nRMSE 3.7%, reaction 7.0%: the 0.76% period difference accumulates as phase drift over two cycles (curves overlap otherwise). Our result is mesh-converged (0.5 mm: period 2.2326 ms, stroke 2.322 mm); the residual is element formulation (Abaqus linear B31 at 1 mm vs our exact Timoshenko element) and stubby-strut large-strain effects (cf. A2a). Fig `5_A4_lattice_drop_nocontact.png` |
| **A5a** 3×3×1 C4 lattice drop, **3 mm elements** (444 B31; edges 2.6–3.2 mm, radius/edge ≤ 0.31, so Abaqus should keep the full contact radius), r 0.8, linear elastic **E = 40 MPa** (E₀ too stiff to densify, E∞ bottoms out), 0.75 J at 6 m/s, impactor bonded to the rigid top plate, **beam–beam contact only** (no plate contact in either code), frictionless, 4 ms | 🟨 Abaqus B31 too coarse | lattice beam–beam contact through densification (onset, stroke, reaction) | max stroke ≤ 3%; time of max stroke ≤ 5%; stroke nRMSE (loading) ≤ 3%; peak bottom reaction ≤ 10%; low-passed reaction correlation (loading) ≥ 0.95. Contact *penetration* is not graded: ours is a physical (Hertz-type) compliance, Abaqus a numerical penalty | `inp/A5a_lattice_drop_contact_mu0.inp`. **Check the .sta** for 'contact edge thicknesses ... were reduced' and for initial overclosures at the joints | max stroke 8.835 mm at 2.42 ms, peak reaction 212 N, contact active from ~0.6 ms (max contact energy 18.7 N mm), max beam overlap 0.45 mm | Ran 2026-10-03; .sta: contact radius reduced at most ×0.986 (full radius effectively used), no initial overclosures. **Not a pass:** max stroke 8.84 vs 7.81 mm (13%), time of max stroke 3.6% ✔, loading stroke nRMSE 6.2%, peak reaction 212 vs 163 N (31%), low-passed reaction correlation 0.69; Abaqus energy drift 3.8%. Abaqus is stiffer from 0.5 ms on, before contact matters (plate decelerates faster, reaction ~30% higher), which points to the element, not to contact. **Ours is mesh-converged:** max stroke 8.835 / 8.984 / 9.046 mm at 3 / 1.5 / 1 mm (peak 212 / 223 / 229 N); our exact Timoshenko element is accurate with 1–2 elements per strut. Abaqus B31 (linear, one integration point) with 1–2 elements per strut and the shear slenderness compensation switched off resists sway only through shear and is too stiff. → re-run as A5a-s / A5a-q. Fig `6_A5a_lattice_drop_contact_mu0.png` |
| **A5b** = A5a with Coulomb friction μ = 0.5 (ours: regularised Coulomb at line-contact points; Abaqus: `*FRICTION` 0.5 in general contact) | 🟨 as A5a | friction in lattice contact | as A5a + friction dissipation (ours W_fric vs ALLFD) ≤ 15% | `inp/A5b_lattice_drop_contact_mu05.inp` | max stroke 8.712 mm, peak reaction 188 N, friction work 58.5 N mm (7.8% of 750) | Ran 2026-10-03: max stroke 8.71 vs 7.59 mm (15%), time of max stroke 7.5%, peak reaction 188 vs 159 N (18%), friction dissipation 58.5 vs 42.5 N mm (38%); Abaqus energy drift 8.9%. Same element issue as A5a; repeat with the element that passes A5a-s/q. Fig `7_A5b_lattice_drop_contact_mu05.png` |
| **A5a-s** = A5a with Abaqus B31 and its **default** shear slenderness compensation (0.25; ≈ 0.46 × κGA on 3 mm struts, which makes 1–2 B31 per strut approximate bending) | 🟨 loading ✅, densification peak ✘ | does the coarse-B31 correction recover the converged beam response? | as A5a | `inp/A5as_lattice_drop_contact_mu0_B31comp.inp` | ours at 3 mm (as A5a): 8.835 mm, 212 N; converged 9.05 mm, 229 N | Ran 2026-10-03 (contact radius ×0.986, no overclosures, energy drift 2.4%). **Loading phase passes:** max stroke 8.835 vs 8.994 mm (**1.8%**; 0.6% from our converged 9.046), loading stroke nRMSE **0.54%**, time of max stroke 4.9%, low-passed reaction correlation 0.96; plate velocity, KE and SE overlap to 2.3 ms. **Densification differs:** peak bottom reaction 212 vs 179 N (18%), and on unloading ours springs back faster (KE at 4 ms 470 vs 330 N mm; Abaqus keeps 400 vs 290 N mm of strain energy). That is the contact law at densification: ours is a material (Hertz-type) compliance with up to 0.45 mm strut indentation, Abaqus a stiff numerical penalty, so struts jam differently. For one crossing our contact matched solids within 0.7–1.6% (A3s2), so B3 (solid lattice) decides the lattice case. Fig `8_A5as_lattice_drop_contact_mu0_B31comp.png` |
| **A5a-q** = A5a with **quadratic B32** (3-node; mid-side nodes added), same 3 mm mesh | ✖ Abaqus radius halved | an Abaqus beam that bends correctly within one element per strut while keeping the long contact edges | as A5a; **check the .sta** for contact-radius reduction (B32 edges may be split at the mid-side node) | `inp/A5aq_lattice_drop_contact_mu0_B32.inp` | as above | Ran 2026-10-03: Abaqus split the B32 contact edges at the mid-side node and **reduced the contact radius ×0.493** (.sta), so beams overlap more: max stroke 9.604 mm (8% over ours), peak 175 N. Not a valid contact comparison. Fig `9_A5aq_lattice_drop_contact_mu0_B32.png` |
| **A5b-s** = A5a-s with Coulomb friction μ = 0.5 (B31, default shear compensation, full contact radius) | ✅ mechanics / ✘ friction work | friction in lattice contact, with the element that passes A5a-s | as A5a-s + friction dissipation ≤ 15% | `inp/A5bs_lattice_drop_contact_mu05_B31comp.inp` | ours (3 mm): max stroke 8.712 mm, peak 188 N, friction work 58.5 N mm | Ran 2026-10-03 (contact radius ×0.986, no overclosures, energy drift 2.8%). **Mechanics pass:** max stroke 8.712 vs 8.703 mm (**0.11%**), time of max stroke **1.1%**, loading stroke nRMSE **0.24%**, peak bottom reaction 188.3 vs 192.8 N (**2.3%**), low-passed reaction correlation 0.978. **Friction dissipation fails:** 58.5 vs 40.6 N mm (44%); ours is 1.4–2.2× higher from contact onset on. Not micro-slip: raising our regularisation velocity 1 → 100 mm/s changes it by 1%. **Suspected model gap (to fix and test):** our slip velocity uses the beam centreline velocities only (no spin ω × r at the surface), so rolling contact counts as sliding, and the friction force is applied without its moment r n × F_t. This affects the energy split (friction is 5–8% of the budget), not stroke or force. Fig `10_A5bs_lattice_drop_contact_mu05_B31comp.png` |

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
| **B2a** ONE C4 unit cell (12.7 mm cube, 36 struts, r 0.8) as a **solid**: strut cylinders + joint spheres fused (gmsh/OpenCASCADE), trimmed at the plate planes, **C3D10M h = 0.4 mm** (39 665 elements, Netgen-optimised; shortest altitude 0.088 mm → est. ~73k increments), bottom face clamped, top face tied to a rigid plate carrying the impactor (0.75/9 J at 6 m/s), linear elastic E = 40 MPa, ν 0.49, frictionless general self-contact, 4 ms | ✔ valid reference / ✘ beam model 24% too soft | joint idealisation + densification contact on a solid reference (P1, P2), cheaply | max stroke, time of max stroke, loading stroke history, peak / correlation of the bottom reaction (as A5); expect the solid to be stiffer (joint material) | `inp/B2a_unit_cell_solid_h040.inp` (6 MB). Ours: the same cell as beams, 1 mm elements | max stroke 10.23 mm, peak bottom reaction 16.7 N, contact energy max 0.59 N mm | Ran 2026-10-05 (3.3 h wall; time step 3.4e-8 s at start, falling to 8.2e-9 s as elements are crushed; 318k increments; energy drift 0.9%). Solid: max stroke 8.323 mm at 2.42 ms, peak bottom reaction 18.4 N. **Our beam cell is softer:** stroke 10.23 mm (+23%), time 2.89 ms (+20%), peak 16.7 N; the solid's reaction is higher from 0.5 ms on, before contact matters, so the cell's **elastic stiffness** differs. Cause: joint material. The solid has a fused lump at each node, while our beams meet at a point and bend right up to the node centre; the struts are short (L/r = 3–8). **Test:** stiffening only the end element of each strut (about 1 mm ≈ 1.25 r, mass and contact unchanged) by ×2.0 gives stroke 8.187 mm (**1.0%**), time 2.33 ms (3.3%), peak 22.0 N (6%), loading stroke nRMSE **0.42%** (×1.7: 3.8%, ×2.3: 4.6%). One-parameter calibration on this case only, so it still needs an independent check. Fig `12_B2a_unit_cell_solid_h040.png` |
| **B2a-fine** = B2a with **h = 0.27 mm** (82 625 C3D10M, shortest altitude 0.062 mm, ~103k increments) | ✅ solid converged | solid mesh convergence (B2a vs B2a-fine) before any full-lattice solid run | stroke and peak change between h 0.4 and 0.27 | `inp/B2a_unit_cell_solid_h027.inp` (12.6 MB) | as B2a | Ran 2026-10-05 (7.2 h wall; 357k increments; energy drift 1.3%). Solid max stroke 8.267 vs 8.323 mm at h 0.4 (**0.7%**), time 2.41 vs 2.42 ms, peak 20.7 vs 18.4 N (the peak is a noisy maximum). **h = 0.4 mm is converged for stroke and timing.** Fig `13_B2a_unit_cell_solid_h027.png` |
| **B2c** (validation of the joint model, written 2026-10-05): the same C4 cell with **r = 0.6** (L/r 4.4–11 instead of 3.3–8), solid C3D10M h = 0.30 mm (47 578 elements, shortest altitude 0.064 mm), impact energy scaled by (0.6/0.8)⁴ to keep a comparable crush (26.4 N mm at 6 m/s), otherwise B2a | ✅ joint model validated | **independent check of k = 3.5** (calibrated on B2a only) | max stroke ≤ 3%; time of max stroke ≤ 5%; loading stroke nRMSE ≤ 3% | `inp/B2c_unit_cell_solid_r060_h030.inp` (7.4 MB) | **Blind prediction, recorded before the Abaqus run.** With joint k = 3.5: max stroke **8.328 mm at 2.39 ms**, peak bottom reaction **7.32 N**. Without the joint model: 9.791 mm at 2.80 ms, 5.85 N | Ran 2026-10-05 (1.9 h wall, 139k increments; time step 2.4e-8 → 2.3e-8 s; energy drift 0.3%). Solid: max stroke 8.367 mm at 2.44 ms, peak 6.89 N. **Blind prediction with k = 3.5: stroke 8.328 mm (0.46%), time 2.39 ms (2.0%), loading stroke nRMSE 0.34%, peak 7.32 N (6.3%)** ✔. Without the joint model: 9.791 mm (17%), 2.80 ms (15%), nRMSE 6.3%. Low-passed reaction correlation 0.81 (0.74 without joints): the reaction is high-frequency ringing in both codes; level and trend match. **The geometric zone + k = 3.5, calibrated at r = 0.8, predicts r = 0.6 to < 0.5% in stroke.** Fig `14_B2c_unit_cell_solid_r060_h030.png` |
| **B2d** (topology check, written 2026-10-05): ONE **C12** unit cell (`example3(name='C12')`: 72 nodes, **96 struts of 2.3–2.9 mm**, node valence 2–3, different angles), **r = 0.5** (L/r 4.6–5.7), solid C3D10M **h = 0.25 mm** (77 549 elements, shortest altitude 0.050 mm), 17 N mm at 6 m/s (plate share 0.1 g), E = 40 MPa, frictionless self-contact, 4 ms. Ours: beams with 0.5 mm elements, joint k = 3.5 (unchanged) | ✘ k is topology-dependent | is the geometric zone + k = 3.5 **topology-independent**? (calibrated on C4 only) | max stroke ≤ 3%; time of max stroke ≤ 5%; loading stroke nRMSE ≤ 3% | `inp/B2d_unit_cell_C12_solid_r050_h025.inp` (12.2 MB); est. 6–12 h (2× B2a elements, ~0.57× initial step) | **Blind prediction, recorded before the Abaqus run.** With joint k = 3.5: max stroke **7.915 mm at 2.32 ms**, peak **7.41 N**. Without the joint model: 9.529 mm at 2.79 ms, 4.83 N | Ran 2026-10-05 (2.7 h wall, 120k increments; time step 1.9e-8 → 3.3e-8 s; energy drift 0.65%). Solid: max stroke 8.748 mm at 2.54 ms, peak 6.04 N. **Blind prediction with k = 3.5 fails: 7.915 mm (9.5% too stiff)**, time 2.32 ms (8.7%), peak 7.41 N (23%), loading nRMSE 4.3%. Uncorrected beams are 8.9% too soft (9.529 mm), so the solid lies between the two. A fit gives k ≈ 1.7–1.8 for C12 (k = 1.9: stroke 3.0%, time 2.2%, nRMSE 1.2%) vs 3.5 for C4. **Checked and rejected:** an axial-vs-bending explanation (both cells are bending-dominated, axial energy 0.3% / 1.0%). Zone geometry is sensible on both (valence-2 C12 kinks at 150° get 0.27 r, valence-3 joints r; strut fraction in zones 0.32 vs 0.25). Fig `15_B2d_unit_cell_C12_solid_r050_h025.png` |
| **B2e–h** (topology check of the fused-section joint model, written 2026-10-05): ONE **BCC** ('B': 9 nodes, 20 struts of 11.0 / 12.7 mm, valence 4 and 8) or **octet** ('O': 14 nodes, 32 struts of 8.98 mm, valence 3/6/8) unit cell, r = 1.0 and 1.5, solid C3D10M h = r/2, E = 40 MPa, frictionless self-contact. These cells are **stretch-dominated** (≈ 4 mm crush means 23–58% axial strain, the A2a large-strain regime), so they are run in the **small-strain first compression** only: ~0.7 mm stroke (≈ 6%, axial strain ≤ 11%), **T = 0.5 ms**, output every 2 µs. Ours: beams with elemSize = r | ✔ ran / ≈ ±10% | is the fused-section model right for stretch-dominated topologies too? | max stroke ≤ 3%; time of max stroke ≤ 5%; peak reaction ≤ 10% | see rows below | blind predictions below | Ran 2026-10-05 (3–11 min each; energy drift ≤ 0.4%). **Input issue found:** for cells with struts in the plate planes the solid started with 12–16% less kinetic energy than ours (its impactor mass subtracted *our* top-node mass, which includes those struts). Compared **energy-matched** (ours re-run at the solid's KE(0), i.e. the same total moving mass); inputs fixed for the future. Peak reaction is wave-dominated in this 0.5 ms window, so not graded. Results in the summary table below. |
| B2e BCC r 1.0, 20 N mm | ✔ ran | | | `inp/B2e_unit_cell_BCC_solid_r100_h050.inp` (39 766 el., min alt. 0.087 mm) | **fused-section: 0.7484 mm @ 0.227 ms, peak 72.4 N**; bare 0.8434 mm @ 0.248 ms, 57.9 N; k 3.5: 0.7418 mm @ 0.221 ms, 76.7 N | — |
| B2f BCC r 1.5, 45 N mm | ✔ ran | | | `inp/B2f_unit_cell_BCC_solid_r150_h075.inp` (33 303 el., 0.116 mm) | **fused-section: 0.6975 mm @ 0.207 ms, peak 176.4 N**; bare 0.8340 mm @ 0.252 ms, 134.9 N; k 3.5: 0.6852 mm @ 0.203 ms, 183.5 N | — |
| B2g octet r 1.0, 16 N mm | ✔ ran | | | `inp/B2g_unit_cell_octet_solid_r100_h050.inp` (65 662 el., 0.070 mm) | **fused-section: 0.7485 mm @ 0.275 ms, peak 46.3 N**; bare 0.8866 mm @ 0.341 ms, 36.7 N; k 3.5: 0.7433 mm @ 0.278 ms, 61.8 N | — |
| B2h octet r 1.5, 45 N mm | ✔ ran | | | `inp/B2h_unit_cell_octet_solid_r150_h075.inp` (50 085 el., 0.103 mm) | **fused-section: 0.7176 mm @ 0.236 ms, peak 153.0 N**; bare 1.0017 mm @ 0.338 ms, 97.8 N; k 3.5: 0.7324 mm @ 0.243 ms, 171.3 N | — |
| **B2i–k** (graded Midplane designs on the C4 cell, written 2026-10-05): top / bottom radii differ, mid = their mean (3 design variables); solid h = r_min/2; impactor mass from the solid's top-face mass (fixed rule); E = 40 MPa | 🟦 | does the joint model hold when struts of different radii meet at a joint (joint sphere = largest radius at the node, as in the model)? | as A5 / B2 (stroke ≤ 3%, time ≤ 5%) | see below | blind predictions below (four joint models) | — |
| B2i top 1.0, bottom 0.6, mid 0.8, 45 N mm | ✅ stroke (Abaqus stopped at 2.70 ms, one distorted element at the bottom plate, still moving at -512 mm/s: its 8.359 mm is a lower bound; est. true max 8.39-8.42) | | | `inp/B2i_unit_cell_C4_graded_top100_bot060.inp` (71.6k el., min alt. 0.040 mm) | **fused+SV: 8.2312 mm @ 2.589 ms, peak 10.02 N**; fused 7.9766 mm @ 2.559 ms; k 3.5: 7.9422 mm @ 2.543 ms; bare 9.8966 mm @ 2.988 ms | — |
| B2j top 0.6, bottom 1.0, mid 0.8, 42 N mm | ✅ stroke (run stopped by the user at 3.17 ms after 14.5 h: the time increment collapsed 20x, 3.4e-8 to 1.8e-9 s, at an element by the top plate) | | | `inp/B2j_unit_cell_C4_graded_top060_bot100.inp` (72.8k el., 0.035 mm) | **fused+SV: 8.0857 mm @ 2.489 ms, peak 15.76 N**; fused 7.8407 mm @ 2.412 ms; k 3.5: 7.7812 mm @ 2.353 ms; bare 9.7447 mm @ 2.908 ms | — |
| B2k top 0.7, bottom 0.9, mid 0.8 (= twin-recovery design), 70 N mm | 🟦 | | | `inp/B2k_unit_cell_C4_graded_top070_bot090.inp` (46.7k el., 0.032 mm) | **fused+SV: 8.5729 mm @ 2.533 ms, peak 17.81 N**; fused 8.2846 mm @ 2.451 ms; k 3.5: 8.2558 mm @ 2.390 ms; bare 10.3263 mm @ 2.920 ms | — |
| (dropped) top 1.1 / bottom 0.5 | ✖ | | | gmsh slivers where a 0.5 mm strut leaves a 1.1 mm joint sphere (min alt. 0.0065–0.012 mm even at h 0.35): ~a day in Abaqus; replaced by B2k | — | — |
| **B3** full 3×3×1 lattice, solid C3D10M, general contact μ = 0.5, rigid plates, merged impactor, 0.75 J | ⬜ | **paper validation figure** | F–δ, peak, stroke, dissipation; expect peak within 10–20% | |
| **B4** design ranking in Abaqus: initial vs optimized (80 N and 120 N plateau designs) | ⬜ | does the optimizer's improvement survive in the reference model? | sign and size of ΔJ, Δpeak, Δefficiency | |
| **B5** (optional) FD gradient in Abaqus: ±Δr on one group (2 runs) | ⬜ | sign and rough size of dJ/dr vs our adjoint | sign agrees; magnitude within ~30% | |

## Joint-stiffness model (2026-10-05), from B2a
**Problem.** A beam model of a non-slender lattice misses the stiffness of the fused material at the nodes. Ours was 23% too soft in stroke against the solid unit cell (B2a), from the elastic phase on.

**Literature (grounding).**
- Luxner, Stampfl & Pettermann (2005, J. Mater. Sci. 40): beam models with the stiffness adapted near the vertices.
- Smith, Guan & Cantwell (2013, Int. J. Mech. Sci. 67) and Labeas & Sunaric (2010, Strain 46): thicker strut ends in beam models of BCC lattices.
- 'Modified beam' models: rigid strut extremities whose length is calibrated against solid models and found to depend on strut diameter and cell topology (e.g. Int. J. Mech. Sci. 2023, S0020740323005015).
- Portela, Greer & Kochmann (2018, Extreme Mech. Lett. 22, 138–148, doi:10.1016/j.eml.2018.06.004): node effects dominate the stiffness of non-slender lattices.
- (Full citations to verify before the paper.)

**Ours** (`latticeSetup.joint_factors`, `explicitFE.JointScaledLaw`; `drop_rig(..., joint={'k': 3.5})`).
- **Zone length is geometric, not fitted:** the distance from the node at which a strut's cross-section has fully left every neighbouring strut's cylinder, ℓ = (r_b + r_a cos θ)/sin θ (= r cot(θ/2) for equal radii). Smooth-maximised over the neighbours, capped at L/2. On the C4 cell: ℓ = 0.58–1.0 r, median 1.0 r.
- **Stiffness:** inside a zone the section is k times stiffer (stiffness only: mass and contact use the true radii). Per element, s = 1/((1 − φ) + φ/k), with φ the zone fraction of its length, so the strut compliance is mesh-independent (identical at 0.5 / 1 / 3 mm elements, test).
- **One constant**, k = 3.5, calibrated once on B2a: stroke 0.4%, loading history 0.44%, peak 1.0%, time of max stroke 2.5% (k = 3: 1.4%, k = 4: 1.7%). Interpretation: a ≈ 1.37× wider effective section over ≈ one radius.
- **Differentiable in the design:** ℓ scales with the radii, so the adjoint carries how a radius change moves the stiffening. Adjoint vs FD through a full EPU 45 drop with joints on: **7e-11, 5e-10, 2e-9** (3 Midplane groups).

**Possible contributions** (honest scope):
1. geometric, topology-free zone length (the literature calibrates the length per topology / diameter);
2. differentiable joint stiffening inside an adjoint-based design loop;
3. validation in impact dynamics with contact and densification against solids (the literature is mostly quasi-static elastic).

**Topology-aware joint model: fused-section stiffness (2026-10-05), `src/jointSection.py`, `drop_rig(joint={'mode': 'section'})`.**
- **Method:**
  - Each strut is sliced along its axis. In each slice, take the solid's material (union of the strut circle, the neighbouring cylinders at both of its nodes and the joint spheres, trimmed at the plates), keeping only the part connected to the strut's own section.
  - The ratios to the bare strut of its second moments about the element's local y and z axes (about the section centroid), its polar moment and its area give per-direction stiffening profiles β(s) ≥ 1 (exactly 1 away from the nodes).
  - Each element gets the series average l_e / ∫ ds/β(s) per resultant (My by I_y ratio, Mz by I_z ratio, torsion by the mean, axial by area).
  - **No fitted constant**: no zone length and no k.
- **Results against the three solid cells:**

  | variant | C4 r 0.8 (B2a) | C4 r 0.6 (B2c) | C12 r 0.5 (B2d) |
  |---|---|---|---|
  | bare beams | +23% | +17% | +8.9% |
  | geometric zone, k = 3.5 (fitted on B2a) | −0.4% (fit) | +0.46% | −9.5% |
  | fused section, mean principal I (scalar) | −0.5% | −0.4% | −10.1% |
  | fused section, min principal I (scalar) | +11.1% | +7.7% | +1.2% |
  | **fused section, anisotropic (I_y, I_z per local axis)** | **+0.1%** (nRMSE 0.48%) | **+0.2%** (0.36%) | **−5.1%** (2.1%) |
  | anisotropic + shear scaled by area ratio | +0.8% | +0.5% | −5.1% |

  (Max-stroke error vs the solid. Our C12 result is mesh-converged: 0.5 → 0.25 mm elements changes it by 0.25%.)
- **Reading:**
  - The anisotropic fused-section model is **parameter-free**. It reproduces C4 at two radii to ≤ 0.2% and halves the C12 error (bare +8.9%, k-model −9.5%, this −5.1%).
  - The scalar mean/min variants bracket C12, so the direction of the joint stiffening is what distinguishes the topologies. The anisotropic form captures most of it.
- **Remaining C12 gap (−5%), candidate causes:**
  - the fused node region is itself compliant (neighbour material attached at an angle does not fully take part in the strut's bending, which a plane-section union assumes);
  - solid discretization of B2d (h/r = 0.5, converged within 0.7% for C4 only; finer B2d not run).
- **Seven solid cells, energy-matched** (max-stroke error vs the solid; negative = ours too stiff):

  | cell | bare beams | k = 3.5 | **fused section (aniso)** |
  |---|---|---|---|
  | C4 r 0.8 (B2a) | +22.9% | −1.1% | **−0.7%** |
  | C4 r 0.6 (B2c) | +15.5% | −1.8% | **−1.2%** |
  | C12 r 0.5 (B2d) | +7.3% | −10.9% | **−6.6%** |
  | BCC r 1.0 (B2e) | +10.0% | −3.5% | **−3.0%** |
  | BCC r 1.5 (B2f) | +11.8% | −8.1% | **−6.4%** |
  | octet r 1.0 (B2g) | +9.6% | −7.9% | **−6.9%** |
  | octet r 1.5 (B2h) | +25.7% | −7.9% | **−9.5%** |

  **Conclusion (2026-10-05):**
  1. Joint stiffening is essential in every topology (bare beams 7–26% too soft).
  2. The parameter-free fused-section model is best or tied on 6 of 7 cells: ≤ 1.2% on C4 (the design lattice), 3–9.5% elsewhere.
  3. The residual is systematically too stiff. That is consistent with the model's upper-bound assumption: all fused material in a slice fully takes part in the strut's bending, whereas material attached at an angle only partly does. 90° C4 corners come closest.
  4. Next refinements to consider: a participation reduction for angled fused material (from the local geometry, not fitted), and solid mesh convergence for the stretch-dominated cells (coarse C3D10M is stiff, so refinement would likely widen, not close, the gap).
- **Saint-Venant participation** (2026-10-05), `joint={'mode': 'section', 'decay': 1.0}`.
  - Fused material outside the strut's own circle counts with weight exp(−d/λ), with d its distance from the circle and **λ = r** (the section's own size). Rationale: bending stress diffuses into side-attached material only over about one section size (Saint-Venant; cf. the effective width of flanges).
  - λ = r was chosen *before* evaluation and **not tuned**; only λ = r has been run.
  - Energy-matched max-stroke error vs the seven solids: C4 r 0.8 **+2.8%**, C4 r 0.6 **+1.2%**, C12 **−3.2%**, BCC r 1.0 **+0.9%**, BCC r 1.5 **−0.2%**, octet r 1.0 **−1.3%**, octet r 1.5 **+1.0%**. **Mean absolute 1.5%** (fused section without decay 4.9%, k = 3.5 5.9%, bare 15.2%), and no longer one-sided. Time of max stroke within 7%; loading nRMSE ≤ 2.8%.
- **Prior art to cite** (2026-10-05 search):
  - Parametrized superelements for lattice joints (Comput. Mech. 2022, doi:10.1007/s00466-022-02176-9): solid-meshed joints, reduced and surrogate-fitted; differentiable, elastic; needs solid analyses per joint family.
  - 'Junction beams' of individually determined length (Int. J. Mech. Mater. Des. 2023, doi:10.1007/s10999-023-09659-4).
  - Rigid-zone factors at beam-column joints (structural frame practice, 0–1, often 0.5).
  - Ours needs no solid analysis, works inside explicit impact with contact, and adds the Saint-Venant participation idea (not found in the lattice literature in this search; verify further before claiming novelty).
- **Costs and open items:**
  - The stiffer joint elements lower our stable time step: C12 needs 2.3× more steps.
  - Not yet differentiable in r. Plan: keep the factors as a leaf tensor in the adjoint (exact ∂J/∂s_e) and add their geometric Jacobian ds_e/dr by finite differences of the slice computation.
- **Next:** blind checks on BCC and octet (B2e–h, written).
- **Deferred (user, 2026-10-05):** a finer B2d solid mesh, to bound how much of the −5% C12 gap is the solid's own discretization. Do it if C12 remains the outlier after B2e–h.

**Status after B2d (2026-10-05)** (scalar k-model).
- Within one topology the model is predictive across radius: B2c blind, stroke 0.46%.
- **Across topologies a single k is not enough:** the C12 cell (B2d) needs k ≈ 1.7–1.8 against 3.5 for C4. The blind k = 3.5 is 9.5% too stiff there, and uncorrected beams 8.9% too soft. This matches the literature finding that the end-zone length/stiffness depends on topology, so claim (1) 'topology-free' does **not** hold for this model form.
- Candidate topology-aware refinement (to test against B2a / B2c / B2d without new Abaqus runs, then confirm blind on a new topology, e.g. BCC 'B' or octet 'O'): derive the zone stiffening from the geometry of the fused section itself, i.e. the second moment of area of the union of the strut with the neighbouring material in slices along the zone, instead of one k.
- Earlier note (still true): **validated on an independent case of the same topology, B2c (r = 0.6, L/r 4.4–11).** Blind prediction vs solid: stroke 0.46%, time 2.0%, loading history 0.34% (uncorrected beams: 17%). Next: B3 (full lattice) and a different topology, to test 'topology-free'.

## Level F: solids of the print geometry (FHS joints), new 2026-10-06

Plan and naming: `notes/15_FHS_SolidValidationPlan.md`.
- Name pattern: `F<level><letter>_unit_cell_<cell>_FHS_solid_r<r x100>_h<h x100>`.
- The letter mirrors the B2 twin: same rig, loading, material, contact, output and mesh size; only the joint is
  different (FHS, src/nodeJoint.py, instead of cylinders + joint spheres).
- Geometry STL: `STL/abaqus_fhs/<case>.stl`.

| Case | Twin | Content | Status |
|---|---|---|---|
| F1a_unit_cell_C4_FHS_solid_r080_h040 | B2a | C4 r 0.8, 42.8k C3D10M (h 0.4, 289 boundary slivers removed), fixed mass scaling BELOW MIN to B2a's 3.37e-8 s (263 el, actual +0.0034 % mass), 4 ms | 🟨 ran to 3.958 of 4 ms (4.7 h); it then failed on element 5808, the 2nd-smallest tet (altitude 0.012 mm, a remaining one-face sliver at the clamped bottom plate), during the rebound. The loading branch and max stroke are valid. |
| F1c_unit_cell_C4_FHS_solid_r060_h030 | B2c | C4 r 0.6, 62.4k C3D10M (h 0.30), 240 el. mass-scaled (+0.03 %), T 3 ms | ✅ completed (2.5 h) |
| F1d_unit_cell_C12_FHS_solid_r050_h025 | B2d | C12 r 0.5, 103.8k C3D10M (h 0.25), 776 el. (+0.06 %), T 3 ms | ✅ completed (1.8 h) |
| F1e_unit_cell_BCC_FHS_solid_r100_h050 | B2e | BCC r 1.0, 41.7k C3D10M (h 0.50), 214 el. (+0.03 %), T 0.5 ms | ✅ completed (3 min) |
| F1f_unit_cell_BCC_FHS_solid_r150_h075 | B2f | BCC r 1.5, 27.3k C3D10M (h 0.75), 233 el. (+0.08 %), T 0.5 ms | ✅ completed (1.3 min) |
| F1g / F1h (octet r 1.0 / 1.5) | B2g / B2h | octet | ⬜ blocked: at some plate nodes the FHS surface meets the plate almost tangentially, so the gmsh remesh overlaps the plate facets ("overlapping facets", PLC error). Fix pending (see below). |
| F1i_unit_cell_C4_graded_top100_bot060_FHS | B2i | C4 graded 1.0 / 0.6, 98.2k C3D10M (h 0.30), 1194 el. (+0.20 %), T 3 ms | ❌ failed at 1.455 ms (6.3 mm stroke, ~75 % of the crush): excessive distortion of an ordinary element (alt. 0.13 mm) at a thin-strut root on the clamped bottom plate; energy conserved until then. Rerun as F1i2 |
| F1j_unit_cell_C4_graded_top060_bot100_FHS | B2j | C4 graded 0.6 / 1.0, 96.0k C3D10M (h 0.30), 208 el. (+0.02 %), T 3 ms | ❌ failed at 1.315 ms (6.1 mm): deformation speed > wave speed at a thin-strut root under the tied top plate. Rerun as F1j2 |

All F1c-j: robust F mesh (Delaunay 2D + smoothing, safe sliver removal), `*SECTION CONTROLS, DISTORTION CONTROL=YES`,
fixed mass scaling BELOW MIN to the B2 twin's time increment. FHS vertices are kept ≥ 5 µm off the plate planes
before the trim (`nodeJoint.PLATE_GAP`): the trim otherwise left 0.006 µm edges that made gmsh's surface
reparametrisation hang (F1c). F1a was made before that; its .inp stays the record.

| F1i2 / F1j2 results (2026-10-09) | | **both failed again**: F1i2 at 1.54 ms (6.2 mm), F1j2 at 0.80 ms (4.2 mm, earlier than F1j); deformation speed > wave speed at a thin-strut root on a plate (z = 0.23 / 12.6 mm); energy conserved until the last increment. Plate refinement does not prevent it. | ❌ partial |
| F1i2_unit_cell_C4_graded_top100_bot060_FHS_plateref_sym | B2i | the graded twin again, h/2 within 1 mm of both plates (where F1i distorted), and FHS plate joints symmetric about the plates (needed: with the finer plate mesh the thin bottom-strut joints meet the plate tangentially and the remesh overlaps); 205.3k C3D10M, 6589 el. (+0.22 %), 31 MB, T 3 ms | 🟦 inp ready |
| F1j2_unit_cell_C4_graded_top060_bot100_FHS_plateref | B2j | the graded twin again, h/2 within 1 mm of both plates (plain FHS joints); 211.9k C3D10M, 8286 el. (+0.26 %), 32 MB, T 3 ms | 🟦 inp ready |

### Results 2026-10-08: F1c, F1d, F1e, F1f (F1i / F1j failed)

**F vs B2 twin (Abaqus vs Abaqus).** Energy absorbed vs stroke, independent of the B2e/f impactor-energy shortfall:

| pair | max stroke F / B2 | energy absorbed at 25 / 50 / 75 / 95 % of the B2 max stroke |
|---|---|---|
| F1a / B2a (C4 r 0.8) | 8.747 / 8.323 (+5.1 %) | -1.1 / +1.0 / -1.3 / -5.3 % |
| F1c / B2c (C4 r 0.6) | 8.227 / 8.367 (-1.7 %) | +4.1 / +4.6 / +5.0 / +6.3 % |
| F1d / B2d (C12) | 8.620 / 8.748 (-1.5 %) | +6.9 / +6.3 / +6.6 / +5.8 % |
| F1e / B2e (BCC r 1.0) | 0.768 / 0.696 (B2e KE 15 % short) | +25 / +15 / +8.6 / +3.5 % |
| F1f / B2f (BCC r 1.5) | 0.719 / 0.671 (B2f KE 16 % short) | +24 / +18 / +13 / +12 % |

- **The FHS print joint is stiffer than cylinder + sphere on C4 r 0.6, C12 and above all BCC.** The webs at acute
  pairs act at small strain.
- **C4 r 0.8 (F1a) is the exception.** It is equal until ~75 % of the crush and then softer (late contact).
- **This corrects step 1 (2026-10-08 morning).** There the FHS joint factors were judged "wrong sign" from F1a alone.

**Joint-model tuning on F1a-f.** Beam models energy-matched to each solid.

Max-stroke error, ours vs Abaqus:

| joint model | F1a | F1c | F1d | F1e | F1f | RMS |
|---|---|---|---|---|---|---|
| sphere factors, λ = r (JOINT_SV) | -2.10 | +4.33 | -0.19 | +1.11 | +3.54 | 2.72 % |
| **FHS slices, λ = r (JOINT_FHS)** | -4.75 | +1.67 | -2.82 | -1.60 | -1.05 | 2.72 % |
| FHS slices, λ = 0.75 r | -3.32 | +2.72 | -1.68 | -0.43 | +0.97 | 2.11 % |
| FHS slices, λ = 0.5 r | -0.51 | +4.80 | +0.28 | +1.41 | +4.14 | 2.92 % |

Energy absorbed vs stroke at 50 / 75 % of the crush (where the joint stiffness acts):

| joint model | F1a | F1c | F1d | F1e | F1f | mean abs |
|---|---|---|---|---|---|---|
| sphere, λ = r | -6.1 / -1.8 | -12.9 / -8.2 | -2.5 / +0.1 | -16.9 / -11.1 | -16.6 / -14.7 | 9.1 % |
| **FHS, λ = r** | -0.9 / +2.4 | -9.6 / -4.3 | +0.8 / +5.7 | -12.1 / -5.4 | -8.3 / -6.8 | **5.6 %** |
| FHS, λ = 0.75 r | -3.6 / +0.2 | -11.0 / -6.0 | -0.2 / +2.7 | -13.5 / -8.3 | -12.5 / -10.6 | 6.9 % |

**Decision: JOINT_FHS** (the same untuned fused-section + Saint-Venant model, λ = r, with the actual FHS joint
sliced) for FHS print geometry.
- It gives a 40 % lower energy-curve error, with the same max-stroke accuracy.
- No new fitted constant.
- `cases.CASES` now uses it for all F cases.

**Graded F1i / F1j, partial curves** (the solids ran to 6.3 / 6.1 mm). Energy ours vs Abaqus at 2 / 4 / 6 mm:

| case | sphere joints | FHS joints |
|---|---|---|
| F1i | -33.6 / -11.5 / -6.5 % | -27.6 / -5.4 / -1.9 % |
| F1j | -21.9 / -10.8 / -3.9 % | -17.4 / -6.0 / +0.1 % |

JOINT_FHS is better again, on graded radii as well.

**Reports with JOINT_FHS** (`compare.py`, ResultsFig/abaqus 29-33), max stroke:

| case | error | |
|---|---|---|
| F1c | 1.70 % | ✔ |
| F1d | 2.78 % | ✔ |
| F1e | 1.35 % | ✔ |
| F1f | 0.80 % | ✔ |
| F1a | 4.73 % | ✘ (the late-crush contact effect) |

The time of max stroke is early by 8-9 % on F1a / e / f: the early-impact issue below.

**Next steps (2026-10-08 evening):**
1. Run F1i2 and F1j2 (graded, plate-refined).
2. **Early impact** (largest remaining error, shared by all joint models): energy deficit at 25 % crush, early time of
   max stroke. Investigate first-contact coupling, beam lumped / rotary mass and wave transit, with the existing
   B2 / F histories.
3. **F1a late-crush gap** (-4.7 %, C4 r 0.8 only): beam-beam contact geometry at densification.
4. **Octet F1g / h meshing:** both plain and symmetric FHS plate joints still fail the gmsh remesh. It needs a
   dedicated plate-wedge treatment, e.g. a short straight extension of the struts into the plates before the trim.
5. **JOINT_FHS in the optimiser:** the factors are not differentiable, and an FHS build costs 5-55 s. Precompute or
   fit a surrogate in the radii at a joint.

### Results 2026-10-09: F1i2 / F1j2 (partial) and conclusions

Energy absorbed [N mm] at 1 / 2 / 3 / 4 / 5 / 6 mm stroke:

| run | 1 | 2 | 3 | 4 | 5 | 6 mm |
|---|---|---|---|---|---|---|
| F1i (to 6.29 mm) | 3.85 | 8.45 | 13.28 | 19.17 | 25.30 | 31.05 |
| F1i2 (to 6.22, plate-refined + symmetric plate joints) | 3.99 | 9.09 | 15.14 | 21.51 | 28.17 | 35.93 |
| F1j (to 6.12) | 2.31 | 6.47 | 11.24 | 17.43 | 23.99 | 29.63 |
| F1j2 (to 4.17, plate-refined) | 2.37 | 6.65 | 11.55 | 17.84 | - | - |
| B2i (sphere joints) | 3.51 | 7.19 | 11.33 | 17.04 | 24.00 | 29.63 |
| B2j (sphere joints) | 2.12 | 5.99 | 10.30 | 16.53 | 22.90 | 28.76 |

- **Mesh sensitivity is small.** F1j2 vs F1j, the same geometry with a finer plate mesh: +2.3 to +2.7 %. The F
  meshes are converged to ~3 %.
- **The plate-joint geometry matters.** F1i2 vs F1i: +4 to +16 %. That is mostly the symmetric plate joints (more
  joint material at the plates), not the mesh. So F1i2 is a different geometry, and it is left out of the tuning
  set. The beam vs F1i2 error (-15 to -33 %) only shows the mismatch: the beam slices the plain FHS.
- **The graded FHS joints are stiffer than sphere joints too.** F1i / F1j vs B2i / B2j: +5 to +17 % early, +3 to +5 %
  at 6 mm. JOINT_FHS stays confirmed.
- **The graded FHS solids cannot reach max stroke with the linear elastic material.**
  - All four runs (F1i / j, F1i2 / j2) collapse an ordinary element at a thin-strut root on a plate at 4-6 mm.
    Refinement made it earlier.
  - Likely cause: linear elasticity at very large local compression lets elements invert.
  - **Change:** for the F (and later B) solids, use a hyperelastic law with the same small-strain E, ν (neo-Hooke),
    or the EPU 45 Yeoh law, the project's target material, in both codes. Until then, judge graded cells on the
    energy curve up to ~4 mm only.
- **The early-impact deficit is the largest remaining beam-model error.** It is seen in every cell and every joint
  model: -16 to -33 % at 1-2 mm in the graded cells, -10 to -35 % at 25 % crush elsewhere. It is not the plate
  struts: the beam rig bonds all plate-plane nodes, as the solid's tied face does.
  - **Next:** a static small-strain stiffness check (solid via scikit-fem P2 tets on the same mesh vs our beam
    model) separates static stiffness from the dynamics of first contact and wave transit.

**New, model-form (not joint): early impact.** At 25 % of the crush, every beam model absorbs 10-35 % less energy
than Abaqus, sphere or FHS alike (BCC -33 %, C4 r 0.6 -20 %, C4 r 0.8 -8-13 %, C12 -9 %). Together with the
earlier "time of max stroke early" and "mid-crush force high", the beam model's early dynamic response differs.
Candidates:
- the rigid-plate / impactor coupling at first contact (the solid's tied top-face mass);
- rotary / lumped mass of the beams;
- the stress-wave transit of the solid.

Check it with the B2a and F1a force and energy histories.

**Octet (F1g / h), options:**
1. A finer FE mesh only near the plates.
2. Symmetric plate joints (`lattice_fhs(mirror=True)`): implemented, but the exactly symmetric node leaves
   coincident facets at the plate; the trim needs an offset.
3. Run the octet pair after the others, as they are the least C4-like.

Beam prediction for F1a: provisional, the fused-section joint model of the sphere joint (`joint_shape='fhs'` not
yet implemented, plan step 2).

### Results 2026-10-08: B2i, B2j, F1a

Ours = fused-section + Saint-Venant joint model (λ = r, `cases.JOINT_SV`): the same model as the blind
predictions. `compare.py` now uses it for B2e-k and F1a; before, it used the plain fused model, which is 4.6 %
stiff on B2i/j.

| case | max stroke ours / Abaqus | time of max | mean crush force 30-90 % stroke ours / Abaqus |
|---|---|---|---|
| B2a (C4, sphere joints) | 8.565 / 8.323 mm (+2.9 %) | 2.42 ms (Abaqus) | 12.27 / 11.75 N (+4.4 %) |
| B2i (graded 1.0 / 0.6) | 8.231 / ≥ 8.359 mm (-1.5 %, ≈ -2 % against the extrapolated max) | 2.59 / ≥ 2.69 ms | 6.13 / 5.71 N (+7.4 %) |
| B2j (graded 0.6 / 1.0) | 8.086 / 8.216 mm (-1.6 %) | 2.49 / 2.69 ms (-7.5 %) | 6.34 / 5.89 N (+7.6 %) |
| F1a (C4, FHS joints) | 8.565 / 8.747 mm (-2.1 %; ours still has the sphere joint) | 2.46 / 2.62 ms | 12.27 / 11.43 N (+7.4 %) |

- **Blind graded predictions hold.** B2i and B2j were predicted before the runs within 1.5-2 % in stroke: the
  joint model works when the radii differ at a joint. The time of max stroke is 4-7.5 % early (ours a little
  stiff early). B2k was skipped (run time).
- **Joint shape matters.** F1a vs B2a, Abaqus vs Abaqus: the FHS (print) joint is softer than cylinder + sphere.
  Max stroke +5.1 % (8.747 vs 8.323 mm), mean crush force -2.7 %, peak bottom reaction -11 % (16.4 vs 18.4 N), with
  1.4 % less material. That is larger than the model's ±3 % accuracy, so the beam joint model needs the FHS shape
  (`joint_shape='fhs'`).
- **New finding: mid-crush force.** The beam model's mean crush force is 4-8 % high on all four C4-based cells, and
  its time of max stroke is early, while the max stroke is within 3 %. So the shape of the force-stroke curve
  differs (ours stiffer mid-crush, softer near the end). Not graded before.
- **Peak bottom reaction.** It is wave-dominated, with low correlation, in the solids. Mean crush force is the
  better force metric.

### Next steps (2026-10-08)

1. **FHS joint shape in the beam joint model.**
   - `jointSection.element_factors_aniso(joint_shape='fhs')`: slice membership from the FHS surface of each node.
   - Target F1a: 8.747 mm, against 8.565 with the sphere joint. Then the blind F cases.
2. **Robust F meshes (before writing more F inputs).**
   - Repair the remaining one-face boundary slivers too: collapse onto the boundary, or a local remesh, to a minimum
     altitude ≥ ~0.04 mm (as the B2i/j meshes).
   - Test `*SECTION CONTROLS, DISTORTION CONTROL=YES` on C3D10M in a short run (check the .dat).
3. **Shorter runs.** Every metric is on the loading branch, and the max stroke is at 2.4-2.7 ms. So use T = 3.0 ms
   for all new C4-type runs (B2k, F1c, F1i-k). That saves 25 % of the run time and avoids the rebound failures.
   B2j's 14.5 h came from the time step collapsing at a strut root by the top plate.
4. **Batch next.**
   - F1c (C4 r 0.6), F1d (C12), F1e-h (BCC / octet, 0.5 ms), F1i / F1j (graded FHS twins of B2i / j).
   - Optionally B2k, if a third graded point is wanted.
5. **Mid-crush force (+4-8 %).** Check against the large-strain measure (A2: engineering vs log strain above ~10 %)
   and the joint bending at large rotation. Use the B2a / F1a force-stroke curves first; no new Abaqus runs needed.

### Steps 1-2 done (2026-10-08)

**Step 1: FHS joint shape in the joint model.**
- `jointSection.element_factors_aniso(joint_shape='fhs')` is implemented. Exact slices: the FHS solid is rotated
  into each strut frame, cut with manifold3d and rasterised onto the slice grid; the Saint-Venant weights are
  unchanged.
- **Result: wrong sign.** The FHS factors are larger than the sphere factors (mean axial 1.094 vs 1.077,
  bending ≤ 1.42 vs 1.37). The beam model gets stiffer: 8.333 mm, against 8.565 with sphere factors and 8.747 in
  Abaqus F1a.
- **Why: the FHS joint is not stiffer.** Abaqus energy absorbed vs stroke is equal for B2a and F1a up to 6 mm
  (within ±2.7 %):

  | stroke | 1 mm | 3 mm | 5 mm | 6 mm | 7 mm | 8 mm |
  |---|---|---|---|---|---|---|
  | F1a vs B2a | +2.6 % | -0.1 % | -0.7 % | -1.1 % | -2.7 % | -4.5 % |

  - The FHS cell only falls behind in late crush (> 6 mm ≈ 50 % strain), i.e. densification / self-contact, where
    the bulky joint spheres of B2a press together earlier.
  - The 5.1 % extra stroke of F1a is that late-crush effect, not joint bending stiffness.
- **Plates are not the cause.** The plate footprints are equal (41.0 vs 41.7 mm2).
- **Decision.** Keep the sphere-joint factors (`JOINT_SV`) for FHS prints: Abaqus says the response is the same
  where the joint model acts. Treat the late-crush difference as contact geometry: next step 5b, how the beam
  model's densification (beam-beam contact) represents node bulk.
- `shape='fhs'` stays available for study, but is not used.

**Step 2: robust F meshes.**
- **Safe sliver removal.** `solid.remove_boundary_slivers` now restores removed tets at any boundary defect
  (`solid.boundary_defects`: non-manifold edge, pinched vertex), so the boundary is always a closed 2-manifold. The
  F1a mesh, as run, had one non-manifold boundary edge.
- **One-face slivers cannot be fixed inside the mesh.** Deleting them exposes equally thin neighbours. Interior
  vertex relaxation (`solid.relax_small_tets`) finds nothing to move: they sit on poor surface triangles at the
  plate rims (fold ≈ 58 deg) and at the mid-plane nodes.
- **Surface first.** `gmsh_from_stl` now meshes and smooths the surface first:

  | F mesh | bad surface triangles (< 10 deg) | tets < 0.088 mm |
  |---|---|---|
  | as F1a was run | 61 | 431 |
  | `F_MESH` (Delaunay 2D, 10 smoothing passes) | 8 | 205 |

  The minimum altitude stays ≈ 0.009 mm (a geometric feature). Mass scaling covers it in the time step (F1a:
  +0.0034 %).
- **New F runs (`F_RUN`).** T = 3.0 ms, and `*SECTION CONTROLS, DISTORTION CONTROL=YES` on the solid section.
  **Check:** that Abaqus accepts distortion control for C3D10M (input processing / .dat, seconds).
- **F1a.** Its committed .inp is the one that ran, and it stays the record; `F1a.fmesh` keeps its mesh settings.

## Level H: FHS solids with the project material, EPU 45 visco-hyperelastic (new 2026-10-09)

Why:
- The beam model used in the optimiser runs EPU 45 visco-hyperelastic (`explicitFE.ViscoHyperLaw`: Yeoh axial /
  fibre law + 1-term quasi-linear Prony), but every A / B2 / F validation so far used linear elastic E = 40 MPa.
- The graded F solids (F1i / j, F1i2 / j2) all distorted at 4-6 mm stroke at a thin-strut root on a plate; linear
  elasticity at large local compression is the likely cause.
- So the joint model (JOINT_FHS) and the beam material law must now be validated together on the real material.

**Material (both codes, from `config/materials.yaml` EPU_45 via `src/material.epu45()`):**
- Abaqus (`cases.epu45_block`):
  - `*HYPERELASTIC, YEOH, MODULI=LONG TERM` with C10, C20, C30 = 2.4, 0.15, 0.02 MPa and D1 = 2/K = 3.5e-4 /MPa;
  - `*VISCOELASTIC, TIME=PRONY` with g1 = 0.958, k1 = 0, τ1 = 5.2e-5 s;
  - ρ = 1.06e-9 t/mm³.
- K = E0 / (3 (1 - 2ν)) = 5714 MPa, from the instantaneous E0 = 342.9 MPa and ν = 0.49. Bulk does not relax
  (k1 = 0), so the solid's impact-rate E0 and ν equal the beam's. The dilatational wave speed equals the linear E0
  runs', so the mass-scaling DT calibration (0.308 · 0.22 h / c_d) is unchanged, with c_d from E0.
- Ours: `cases._a4_rig` passes `material.epu45()` when `material='epu45'`, so `drop_rig` builds `ViscoHyperLaw`
  (as in the optimiser). The joint model is JOINT_FHS.
- New outputs:
  - Abaqus `ALLCD` (viscous dissipation) in the energy output;
  - ours `ALLIE` (internal work) and `ALLCD` (material dissipation);
  - `compare.py` H cases add the viscous-dissipation signal.

**Cases.** The name is `H1<letter>_..._FHS_EPU45...`, and the letter mirrors the F twin (H1a ↔ F1a ↔ B2a).
- Same FHS geometry and mesh settings as F, distortion control on.
- Impact energy = the F energy × a scale, sized with our EPU 45 beam model for ~8 mm max stroke (63 % of the cell),
  as in F.
- T = 4 ms, because max stroke now comes at 2.9-3.6 ms instead of 2.4-2.7.
- BCC keeps the 0.5 ms first-compression window, with ×4 energy. At 0.5 ms the cell responds near E0, so the F
  energy gave only 0.3 mm.

| Case | Twin | Mesh | Energy | Ours (EPU 45, JOINT_FHS): max stroke, peak RF3, viscous loss / E | Est. Abaqus wall time |
|---|---|---|---|---|---|
| H1a_unit_cell_C4_FHS_EPU45_solid_r080_h040 | F1a | 45.0k C3D10M (robust F mesh), 188 el. scaled (+0.03 %), DT 1.15e-8 s | 66.7 N mm (×0.8) | 8.05 mm @ 2.95 ms, 16.2 N, 0.58 | ~14 h |
| H1c_unit_cell_C4_FHS_EPU45_solid_r060_h030 | F1c | 62.4k, 242 el. (+0.03 %) | 19.8 N mm (×0.75) | 7.73 mm @ 2.91 ms, 5.4 N, 0.60 | ~10 h |
| H1d_unit_cell_C12_FHS_EPU45_solid_r050_h025 | F1d | 103.8k, 776 el. (+0.06 %) | 12.8 N mm (×0.75) | 7.92 mm @ 2.98 ms, 3.9 N, 0.52 | ~7 h |
| H1e_unit_cell_BCC_FHS_EPU45_solid_r100_h050 | F1e | 41.7k, 212 el. (+0.03 %) | 80 N mm (×4) | 0.88 mm @ 0.32 ms, 156 N, 0.87 | ~10 min |
| H1f_unit_cell_BCC_FHS_EPU45_solid_r150_h075 | F1f | 27.3k, 233 el. (+0.08 %) | 180 N mm (×4) | 0.80 mm @ 0.29 ms, 351 N, 0.87 | ~5 min |
| H1i_unit_cell_C4_graded_top100_bot060_FHS_EPU45 | F1i | 97.9k, 1160 el. (+0.20 %) | 45 N mm (×1) | 8.57 mm @ 3.56 ms, 12.4 N, 0.59 | 30-50 h |
| H1j_unit_cell_C4_graded_top060_bot100_FHS_EPU45 | F1j | 96.0k, 207 el. (+0.02 %) | 42 N mm (×1) | 8.22 mm @ 3.22 ms, 13.2 N, 0.59 | 30-50 h |

Wall times are estimates: the F wall time × 2.93 (smaller DT) × T ratio. The increment also shrinks during the
crush.

**Run order:**
1. **H1e, H1f** (minutes): the first check that Abaqus accepts the material block, and a check of the
   instantaneous / viscous response (the first compression is E0- and Prony-dominated).
2. **H1a** (~14 h): the reference cell. It tests material + joint together at full crush, against F1a (linear)
   and B2a.
3. **H1j** (then H1i): does the hyperelastic law remove the thin-strut-root distortion of the graded cells?
4. **H1c, H1d:** the rest of the joint-model tuning set on the real material.

To halve the cost if needed: ν = 0.45 in both codes (≈ 2× larger DT). That changes the material, so decide
before the first H run is used for tuning.

### Plan around the material change (2026-10-09)

Done so far: the FHS print geometry (notes/13), F solids (linear E = 40), and JOINT_FHS chosen (energy-curve error
5.6 % vs 9.1 %). The open items:
- graded cells failing at 4-6 mm;
- the early-impact deficit (-10 to -35 % energy at 25 % crush in every cell and joint model);
- F1a late crush;
- octet unmeshable.

1. **Material on the solids (this step, done):** EPU 45 in Abaqus and in our rig, the H inputs written.
2. **Run H1e / f, then H1a (user).** On return, `compare.py H1a`:
   - the energy-stroke curve, peak RF3, max stroke and its time;
   - ALLCD vs our viscous loss;
   - the Abaqus added mass, distortion-control energy (ALLDC is not requested; check that ALLAE stays small) and
     hourglass.
   - Decision gate:
     - **if the ours-vs-H1a error ≈ ours-vs-F1a (±3 %)**, the material law is consistent and JOINT_FHS carries over;
     - **if not**, the error is material-model form → step 3.
3. **Material-only check (B1, only if step 2 fails the gate):** one straight strut, solid C3D10M vs our beam, axial
   and bending at impact rates, EPU 45. This separates the beam QLV / fibre-Yeoh law from Abaqus' finite-strain
   viscoelasticity (they differ by construction: QLV on resultants vs Prony on the deviatoric Kirchhoff stress).
   Cheap (minutes).
4. **Early-impact deficit (static check, parallel to 2):** small-strain static stiffness of the F1a / B2a solids
   (scikit-fem P2 tets, same mesh) vs our beam cell.
   - If the static stiffness matches, the deficit is dynamic: first-contact / tied-plate mass, rotary mass, or wave
     transit.
   - If not, it is the joint / plate stiffness at small strain.
   - With EPU 45 the first ~0.1 ms is near-E0 and strongly damped (BCC: 87 % of the energy dissipated), so H1e / f
     test the early response directly.
5. **Graded cells (H1j / H1i):**
   - If they reach max stroke, JOINT_FHS is validated on graded designs at full crush.
   - If they still distort, refine locally at the thin-strut plate roots, or add plate fillets to the FHS plate
     joints (the print has a base plate anyway).
6. **JOINT_FHS for the optimiser:** the FHS joint factors as a function of the joint radii (precomputed table or
   surrogate: FHS slicing costs 5-55 s per lattice, too slow inside the adjoint loop). Validate on H1a / H1i / H1j.
7. **Octet (F1g / H1g):** extend the struts into the plates before the trim (no tangential FHS-plate meeting), then
   mesh. Last, because it is the least C4-like.
8. **B3 / B4, the full 3×3 lattice with EPU 45** (paper figure, B-level plan), with the joint model from 6. Use the
   symmetry / ν options in the cost table to keep it to a few hours.

## Cost: 3D solid (Abaqus) vs our beam model (for future planning)
| Model | Abaqus solid C3D10M | Abaqus beams B31 | Ours (beams) |
|---|---|---|---|
| unit cell, 4 ms, E = 40 MPa | h 0.4 mm: 39.7k el, 318k increments, **3.3 h** wall; h 0.27 mm: 82.6k el, 357k incr, **7.2 h** (user's machine) | — | **20 s** (1 mm elements, joint model, 1653 steps, CPU 4 threads, shared machine) |
| 3×3 lattice, 4 ms, E = 40 MPa, contact | est. ~9× unit cell ≈ **30 h** at h 0.4 (quarter symmetry ≈ 4× less, ν 0.45 ≈ 2× less) | **~1 s** (A5a-s, 444 B31) | **3 min** (3 mm elements, 7914 steps, CPU 4 threads, shared) |
| 3×3 lattice, real EPU 45, 8 ms, gradient | — | — | ~2–7 min per evaluation (value + adjoint gradient; 3 mm GPU ≈ 80–140 s at 3 ms) |

The solid time step falls 3–4× during the crush (element distortion): 3.4e-8 → 8.2e-9 s in B2a.

## Order
A1 → A2 → A3 → A4 → A5a → A5b → B0 → B1 → B3 → B4 → (B2, B5).

**Level-B staging (2026-10-03),** cheapest first. Solids are verification checkpoints; the beam model stays the design tool.
1. **B2a / B2a-fine:** one unit cell, E = 40 MPa, mesh convergence (minutes to under an hour each).
2. **B3a:** full 3×3 lattice solid at E = 40 MPa = the A5 conditions (~360k C3D10M at 0.4 mm; ~30 min on 8 CPUs). This decides the densification-contact difference seen in A5a-s.
3. **B3:** full lattice with real EPU 45 (Yeoh + Prony): the paper validation figure, a few hours, once.
4. **B4:** initial vs optimized design in the solid model (2 × B3).

Options:
- ν = 0.45 in both codes (≈ 2× cheaper);
- a quarter model on the two symmetry planes, if the crush stays symmetric;
- selective mass scaling of the few smallest tets (< 1% mass).

Meshing:
- gmsh (OpenCASCADE booleans + Netgen optimisation), installed in the env 2026-10-03.
- The manifold3d/TetGen route left slivers (shortest altitude 0.001 mm).

## Log
- 2026-10-09: **EPU 45 visco-hyperelastic on the solids (Level H).**
  - `cases.material_block(material='epu45')` / `epu45_block` (Yeoh + Prony, D1 from E0 and ν 0.49);
    `_a4_rig(material='epu45')` gives the beam ViscoHyperLaw; ALLCD output on both sides.
  - H1a, c, d, e, f, i, j written (energies sized with our model for ~8 mm stroke; T 4 ms; BCC 0.5 ms ×4 energy).
  - Plan steps 1-8 in "Plan around the material change".
- 2026-10-05 (i): **Saint-Venant participation (λ = r, untuned) brings all seven solid cells within ±3.2%** (mean 1.5%; before 4.9%).
  - Graded designs B2i–k written with blind predictions from four joint models.
  - Prior art recorded: parametrized superelements 2022, junction beams 2023, rigid-zone factors.
- 2026-10-05 (h): **B2e–h done.** Energy-matched, the fused-section joint model is within 3–9.5% on BCC / octet (bare 10–26%) and ≤ 1.2% on C4; residual systematically too stiff.
  - Fixed: solid impactor mass now from the solid's own top-face mass. `compare.py` energy-matches B2 cells whose solid KE(0) differs from nominal by > 1%.
- 2026-10-05 (g): B2e–h written (BCC / octet, r 1.0 / 1.5) with blind predictions: fused-section model, plus bare beams and the k = 3.5 model.
  - Small-strain first compression only (stretch-dominated cells reach 23–58% axial strain at a 4 mm crush). The joint effect there is 11–28% in stroke (bare vs fused section), and the fused-section and k models differ most in peak reaction (octet: 46 vs 62 N, 153 vs 171 N).
  - Finer B2d mesh deferred.
- 2026-10-05 (f): **topology-aware, parameter-free joint model** (anisotropic fused-section stiffness from the slice geometry): C4 r 0.8 +0.1%, C4 r 0.6 +0.2%, C12 −5.1% (k-model −9.5%, bare +8.9%). Not yet differentiable; next: blind third topology.
- 2026-10-05 (e): **B2d (C12):** the blind joint model with k = 3.5 is 9.5% too stiff; a fit needs k ≈ 1.75. So the geometric zone + one k transfers across radius (C4: B2c) but not across topology.
  - Plateau re-runs with joints (k = 3.5, C4): 80 N → radii 0.799 / 0.803 / 0.400, error ×0.448, efficiency 0.70; 120 N → 0.871 / 0.869 / 1.172, error ×0.278, efficiency 0.69.
- 2026-10-05 (d): **B2d** (C12 cell, r 0.5) written with a blind prediction: tests topology independence of the joint model. `example3(name=...)` exposes the notebook's lattice types ('C4', 'C12', 'B' BCC, 'O' octet).
- 2026-10-05 (c): **B2c ✅ joint model validated blind at r = 0.6:** stroke 0.46%, history 0.34% (without joints 17%). k = 3.5 carries over from r = 0.8.
- 2026-10-05 (b): **joint-stiffness model implemented** (geometric zones, k = 3.5 from B2a, mesh-independent, exact adjoint).
  - **B2c** (r = 0.6) written with a recorded blind prediction.
  - Plateau optimisations (80 N GPU, 120 N CPU) re-launched with joints.
  - Cost table added.
- 2026-10-05: **B2a (solid unit cell):** the solid reference is mesh-converged (0.7% between h 0.4 and 0.27).
  - **Our beam model is 23% too soft in stroke:** missing joint stiffness.
  - A ×2 stiffness on the strut end elements (≈ 1.25 r) matches the solid to 1% (stroke) / 0.42% (history): calibrated, still to be validated.
  - Cost lesson: the solid time step falls 3–4× during the crush (distortion), so a full-lattice solid at h 0.4 is ~9× B2a ≈ 30 h at the same CPU count, unless reduced (quarter symmetry ≈ 4×, ν 0.45 ≈ 2×).
  - History CSVs may now be split (`<name>_history_*.csv`); compare.py joins the parts.
- 2026-10-03 (c): **A5b-s:** mechanics pass (0.1–2.3%); friction dissipation +44% vs Abaqus.
  - Suspected cause: no beam-spin surface velocity and no friction moment in our friction model.
  - **Level A complete for mechanics.**
  - Level-B pipeline built: `abaqus/solid.py` (gmsh solid lattice → C3D10M), `example3(nx, ny)`. B2a / B2a-fine written.
- 2026-10-03 (b): **A5a-s** (B31 with Abaqus's default shear compensation) matches the loading phase: stroke 1.8%, history 0.54%.
  - Contact at densification differs (peak 18%, slower rebound in Abaqus): material compliance in ours vs Abaqus's penalty.
  - **A5a-q** invalid: B32 contact radius halved by Abaqus.
  - Lesson: for coarse lattice meshes in Abaqus use B31 with the default compensation, and keep edges ≥ ~2.7 mm so the contact radius is not reduced.
  - A5b-s written (friction).
- 2026-10-03: **A5a/A5b do not pass.** Abaqus at 3 mm is 13–15% stiffer in stroke from the start, before contact matters.
  - Ours converges with refinement (8.84 → 8.98 → 9.05 mm), so the 3 mm Abaqus B31 model is the outlier: 1–2 linear elements per strut with the slenderness compensation off.
  - Added A5a-s (B31, default compensation) and A5a-q (B32) on the same mesh, to keep the full contact radius.
- 2026-10-02 (j): old comparison figures deleted (including the three `ResultsFig/explicit/*_abaqus_A1*`).
  - Regenerated only for valid references: `ResultsFig/abaqus/1_A1, 2_A2b, 3_A3c, 4_A3s2, 5_A4`.
  - A5a/A5b written: 3 mm lattice mesh, E = 40 MPa, beam–beam contact only, full-radius contact control.
  - `drop_rig(start='merged')` now keeps lattice contact when requested (it used to switch all contact off).
- 2026-10-02 (i): **A3s2 ✅**: impulse 0.7%, rebound 1.6%, hourglass 5.7%.
  - Level A status: A1, A2b, A3c, A3s2, A4 pass; A2/A2a and A3/A3b differences explained (large strain; Abaqus contact-radius reduction).
  - **Next: A5** (lattice with contact). On 1 mm lattice elements Abaqus will reduce the beam contact radius as in A3, so either A5 on the 3 mm lattice mesh (check the .sta for the reduction warning) or go to B3 (solid lattice).
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

## Level B1: single EPU 45 strut (material-law check), new 2026-10-12

Why: the H cells (notes/15 "H results") show the beam model too stiff on BCC / C12 with EPU 45 (stroke up to
-12.5 %, peak +26 %), though it matched the same cells with linear E. B1 separates the strut material law from the
joint model.
- One strut, the H cells' material block, no impactor / joint / contact.
- Prescribed displacement: SMOOTH STEP over the ramp, then held.
- Reaction = sum RF3 (axial) or RF1 (bending) at BOT.

| case (abaqus/inp) | test | mesh | T | est. wall time |
|---|---|---|---|---|
| B1a_strut_axial_comp30_fast | short cylinder L 2 r 1, frictionless platens, 30 % compression in 0.3 ms + 0.7 ms hold | 2.2k C3D10M | 1 ms | minutes |
| B1b_strut_axial_comp30_slow | as B1a, ramp 3 ms + 1 ms hold | 2.2k | 4 ms | ~10 min |
| B1c_strut_axial_tens30_fast | as B1a in tension (stress-measure signature) | 2.2k | 1 ms | minutes |
| B1d_strut_axial_comp05_fast | as B1a, 5 % (BCC strain level) | 2.2k | 1 ms | minutes |
| B1e_strut_bend_fast | fixed-guided L 11 r 1, tip 1.1 mm in 0.3 ms + 0.7 ms hold | 10.2k | 1 ms | ~15-30 min |
| B1f_strut_bend_slow | as B1e, ramp 3 ms + 1 ms hold | 10.2k | 4 ms | ~1 h |
| B1g_strut_bend_stubby_fast | as B1e with L 5.5 (shear part, as BCC r 1.5) | 5.4k | 1 ms | ~15 min |
| B1e2_strut_bend_fast_h020 | B1e refined (h 0.2, 48 around): mesh convergence | 74k | 1 ms | ~1-3 h |

Order: B1a, B1c, B1d, B1e (they decide) → B1g, B1b, B1f → B1e2 (convergence of the bending answer).

Run and extract as for the other levels:
- `abaqus job=<name> input=<name>.inp double=both cpus=<n> interactive`
- `abaqus python extract_odb.py <name>.odb`

Then, here:
- `PYTHONPATH=src:tests python abaqus/b1_predict.py` adds the Abaqus rows to `abaqus/results/B1_summary.md`;
- `python abaqus/compare.py B1a ...` gives the overlays and pass marks (peak / ramp-end / hold-end reaction and
  ramp work within 5 %).

Blind predictions (ours, before Abaqus), nominal / kirchhoff relaxation:

| case | peak reaction [N] | end of ramp | end of hold | ramp work [N mm] |
|---|---|---|---|---|
| B1a | 155 / 139 | 82.0 / 72.6 | 21.2 / 21.2 | -56.9 / -53.1 |
| B1b | 31.4 / 28.2 | 21.4 / 21.3 | 21.2 / 21.2 | -12.3 / -11.3 |
| B1c | 76.5 / 81.8 | -33.5 / -36.7 | -11.0 / -11.0 | -34.1 / -35.9 |
| B1d | 16.9 / 16.6 | 7.87 / 7.72 | 2.39 / 2.39 | -1.18 / -1.17 |
| B1e | 1.04 (both) | -0.588 | -0.105 | -0.507 |
| B1f | 0.149 | -0.105 | -0.105 | -0.122 |
| B1g | 2.46 | -1.11 | -0.340 | -0.932 |
| B1e2 | as B1e (same beam) | | | |

The bending cases do not depend on the relax form (axial only). The reaction sign follows each code's RF
convention; compare.py compares like with like.

### Results 2026-10-12: Level B1 (all 8 runs)

Ours (current law) vs Abaqus:
- **Axial:** 0.5-2.1 % in all of peak, ramp end, relaxed hold and ramp work (B1a-d).
- **Bending:** -1 to -6 % (L 11; B1e-f, B1e2 mesh-converged within 0.8 %) and -8 to -9 % (L 5.5, B1g); ours softer.
- The Kirchhoff relax variant is off by 8-10 % at 30 % strain: rejected, nominal kept.
- **Conclusion:** the strut material law is validated; the H-cell excess stiffness (BCC / C12) is at the joint /
  cell level.
- Next: B3 joint specimen, linear and EPU 45 (notes/15 "B1 results").

## Level B3: BCC joint specimen + linear B1 twins, new 2026-10-12

Why: B1 validated the strut law (axial 0.5-2.1 %), so the BCC / C12 excess stiffness with EPU 45 sits at the joints.
- **B3** is the BCC r 1.0 cell's 8 diagonals only: the FHS centre node, single struts meeting the plates, so no plate
  corner joints. Same FHS geometry, h 0.5 mesh, mass scaling and plates as H1e.
- The top plate is driven down 1.0 mm (smooth step) instead of being hit by the impactor; no contact.
- B3 against H1e locates the excess (centre node or plate corner joints). Linear vs EPU 45 measures the joint
  stiffening per material.

| case (abaqus/inp) | test | mesh | T | est. wall time |
|---|---|---|---|---|
| B3a_bcc_node_linE0_fast | linear E0 = 342.9, ν 0.49; 1.0 mm in 0.3 ms + 0.2 ms hold | 21.4k C3D10M | 0.5 ms | ~5-10 min |
| B3b_bcc_node_EPU45_fast | EPU 45; 1.0 mm in 0.3 ms + 0.7 ms hold | 21.4k | 1 ms | ~10-20 min |
| B3c_bcc_node_EPU45_slow | EPU 45; 1.0 mm in 3 ms + 1 ms hold | 21.4k | 4 ms | ~40-80 min |
| B1f_lin_strut_bend_slow_linE0 | B1f with linear E0 (end-effect check) | 10.2k | 4 ms | ~1 h |
| B1g_lin_strut_bend_stubby_linE0 | B1g (L 5.5) with linear E0, 3 ms ramp | 5.4k | 4 ms | ~30-60 min |

Order: B3a, B3b (they decide) → B3c → B1g_lin, B1f_lin. Run and extract as for B1.
- Then: `PYTHONPATH=src:tests python abaqus/b3_predict.py` (adds the Abaqus rows and the solid's joint stiffening);
- `abaqus/b1_predict.py B1f_lin B1g_lin`; `compare.py` for the overlays.

**Blind predictions** (reaction at the end of the ramp; stiffening = JOINT_FHS / bare):

| case | bare beams | JOINT_FHS | our joint stiffening |
|---|---|---|---|
| B3a linear | 62.3 N | 72.0 N | 1.156 |
| B3b EPU 45 fast | 10.9 N | 12.1 N | 1.105 |
| B3c EPU 45 slow | 2.71 N | 3.07 N | 1.132 |

Linear twins (ours = Timoshenko fixed-guided within 0.7-0.9 %): B1f_lin -2.49 N, B1g_lin -8.10 N at the hold.

**What decides what:**
- **The Abaqus solid stiffening (B3 solid / ours bare) ≈ ours** (1.10-1.16) in both materials → the centre node is
  modelled right. The H1e excess then sits at the plate corner joints (1 diagonal + 3 edges in the plates): check
  their joint factors next (a B3-like corner specimen).
- **The solid stiffening is lower with EPU 45 than with linear** → the joint stiffening is material-dependent. Look at
  why (e.g. near-incompressibility once relaxed, or the joint zone's strain rate) before changing the joint model.
- **B1f_lin / B1g_lin show the same -5 / -9 % as the EPU 45 B1f / B1g** → the bending gap is the geometric
  clamped-end effect, not the material.

### Results 2026-10-12: Level B3 (all 5 runs)

Ours (JOINT_FHS) vs Abaqus on the BCC centre node alone:
- linear -7.8 %; EPU 45 fast +2.5 %; EPU 45 slow -4.2 % (reaction at the end of the ramp); ours slightly soft.
- The solid joint stiffening is material / rate dependent: 1.25 / 1.08 / 1.18 against ours 1.16 / 1.11 / 1.13.
- Linear bending twins: -4.4 / -7.9 % (EPU 45: -5.3 / -9.4 %), so the B1 bending gap is the geometric end effect.
- **Conclusion:** the centre node does not cause the H1e excess stiffness. Next: B4 (full BCC cell, prescribed
  motion; notes/15 "B3 results").

## Level B4: full BCC cell under prescribed plate motion, new 2026-10-12

Why: B3 cleared the BCC centre node; the H1e excess stiffness (ours -12.5 % stroke) sits at the plate corner joints
or in the impactor-driven loading.
- **B4 is the solid H1e actually ran:** nodes, elements, BOT / TOP sets and mass scaling are read from the H1e .inp,
  and verified identical.
- It is driven like B3 (top plate down 1.0 mm by a smooth step; no impactor, no contact).
- B4 vs B3 isolates the plate corner joints (1 diagonal + 3 edges in the plates). B4 vs H1e isolates the impactor /
  dynamics.

| case (abaqus/inp) | test | mesh | T | est. wall time |
|---|---|---|---|---|
| B4a_bcc_cell_linE0_fast | linear E0, ν 0.49; 1.0 mm in 0.3 ms + 0.2 ms hold | 41.7k C3D10M (= H1e) | 0.5 ms | ~10 min |
| B4b_bcc_cell_EPU45_fast | EPU 45; 1.0 mm in 0.3 ms + 0.7 ms hold | 41.7k | 1 ms | ~20 min |
| B4c_bcc_cell_EPU45_slow | EPU 45; 1.0 mm in 3 ms + 1 ms hold | 41.7k | 4 ms | ~1-1.5 h |

Order: B4a, B4b, then B4c. Then `PYTHONPATH=src:tests python abaqus/b4_predict.py`.

**Geometry note:** the H1e solid predates the FHS fix of 2026-10-11 (notes/17 P3.4).
- So ours gives JOINT_FHS with the joint factors sliced from that old geometry (like-for-like) and from the
  current code.
- They differ by ~1.5 % here. On H1e itself, old factors give -11.7 % instead of -12.6 % in stroke (peak
  unchanged), so the H conclusions stand.

**Blind predictions** (reaction at the end of the ramp; stiffening vs bare beams):

| case | bare | JOINT_FHS old geometry | JOINT_FHS current | our stiffening (old / current) |
|---|---|---|---|---|
| B4a linear | 401 N | 494.8 N | 500.9 N | 1.234 / 1.249 |
| B4b EPU 45 fast | 63.0 N | 76.7 N | 77.7 N | 1.218 / 1.233 |
| B4c EPU 45 slow | 18.4 N | 22.6 N | 22.9 N | 1.231 / 1.248 |

Our joint model adds +22-25 % on the full cell vs +10-16 % on the centre node alone (B3), so the plate corner
joints + edges are ~10 points of it.

**What decides what:**
- **B4b: ours (old geometry) too stiff by ~10 % or more** (like H1e) → the cause is in the cell, not the impactor. As
  B3 cleared the centre node, it is the plate corner joint factors (JOINT_FHS near the plates) with EPU 45; compare
  B4a to see whether it is EPU-specific.
- **B4b matches within a few % (as B3)** → the cell model is right under prescribed motion; the H1e error comes from
  the impactor-driven loading (impactor / plate mass, contact start, dynamic loading rate). Check those against H1e
  next.
- **B4a vs B4b:** the material / rate dependence of the joint stiffening at cell level (B3 showed 1.25 linear vs 1.08
  EPU fast at the centre node).

### Results 2026-10-13: Level B4 (all 3 runs)

- **Full BCC cell under prescribed motion:** the solid ≈ bare beams (-1.9 / +6.3 / +3.1 %); JOINT_FHS +21 / +29.5 /
  +27 % too stiff (linear / EPU fast / EPU slow).
- ~3/4 of the excess is the joint factors of the 4 vertical edge columns at the plate corners. Likely the fused-section
  slices there include the plate-bonded in-plate struts (double-counting the clamp).
- Switching plate-end factors off fixes r 1.0 (B4a +0.4 %, H1e stroke +1.5 %, peak +5.5 %) but makes r 1.5 too soft
  (H1f +17 %): rejected; fix the slicing at plates instead (notes/15 "B4 results").
- JOINT_FHS's selection by the F-series energy-curve error hid a +28 % peak-force error (F1e). Joint-model checks now
  need a prescribed-motion case and the peak force.

## Level B5: simple cubic cells (no internal joint), new 2026-10-13

Why: B4 put ~3/4 of the JOINT_FHS excess on the plate corner joints of the vertical columns. A simple cubic (SC) cell
has **no internal joint at all**: 4 vertical columns (plate to plate, axial) and 8 edges lying in the plates (cut in
half, bonded). Every joint is a plate joint, so SC isolates the plate-end factors with no BCC centre node or
diagonals. 2 × 2 cells add plate joints with 4 struts (edge) and 5 struts (centre) next to the 3-strut corners: does
the stiffening change with the joint type?

Setup as B3 / B4: top plate (rigid body REF) down 0.5 mm by a smooth step in 0.3 ms, then hold; bottom plate clamped;
no impactor, no contact. 0.5 mm = 3.9 % strain, because an r 1.0 clamped column buckles at ~6 %. Solid and joint
factors both use the **current** FHS geometry.

| case (abaqus/inp) | test | mesh | T | est. wall time |
|---|---|---|---|---|
| B5a_sc_1x1_r10_linE0 | 1 × 1 SC, r 1.0, linear E0, ν 0.49 | 21.8k C3D10M, h 0.5 | 0.5 ms | ~5 min |
| B5b_sc_1x1_r10_EPU45_fast | as B5a, EPU 45, hold 0.7 ms | 21.8k | 1 ms | ~10 min |
| B5c_sc_1x1_r15_linE0 | 1 × 1 SC, r 1.5 (L/r 8.5), linear | 15.1k, h 0.75 | 0.5 ms | ~5 min |
| B5d_sc_2x2_r10_linE0 | 2 × 2 SC, r 1.0, linear: plate joints with 3 / 4 / 5 struts | 54.0k, h 0.5 | 0.5 ms | ~15 min |

Order: B5a, B5c, B5d (linear: geometry only), then B5b. Then
`PYTHONPATH=src:tests python abaqus/b5_predict.py`. Outputs as B3 / B4 (RF3 summed on BOT, U3 / V3 of REF, energies).

**Meshing note:** the in-plate struts put a ring row of the FHS surface on the plate plane. Depending on the cell,
gmsh failed on the trimmed STL, either "wrong topology for parametrization" (B5c, a 5 µm strip of needle triangles)
or "overlapping facets" (B5d). Fix: `nodeJoint.lattice_fhs(plate_push='own' | 'out')` chooses the side the
near-plane vertices are pushed to. `b3_mesh` tries own, then out, then out with algo2d 6. Used: B5a own, B5c out,
B5d out + algo2d 6. The default stays 'own', so earlier geometries are unchanged.

**Blind predictions** (reaction at the end of the ramp; stiffening vs bare beams):

| case | bare | JOINT_FHS | JOINT_FHS + plate_layer | stiffening JOINT_FHS / + plate_layer |
|---|---|---|---|---|
| B5a 1×1 r 1.0 linear | 169.4 N | 187.5 N | 171.7 N | 1.107 / 1.014 |
| B5b 1×1 r 1.0 EPU 45 fast | 24.4 N | 27.0 N | 24.8 N | 1.107 / 1.014 |
| B5c 1×1 r 1.5 linear | 381.1 N | 444.1 N | 389.5 N | 1.165 / 1.022 |
| B5d 2×2 r 1.0 linear | 381.1 N | 428.1 N | 388.3 N | 1.123 / 1.019 |

(B5c and B5d bare are equal by construction: the columns carry the load axially, and 4 × 1.5² = 9 × 1.0² in area.)

**What decides what:**
- **Solid ≈ bare (≤ ~3 %)** → the plate-bonded layer adds no stiffening (the B4 reading). plate_layer is right in
  kind, and what is left of the B4 / H excess is elsewhere (diagonals at the plate corners).
- **Solid ≈ JOINT_FHS (+11-17 %)** → the plate-end factors are right for columns alone, so the B4 excess comes from how
  the diagonal and the columns interact at the BCC corner.
- **B5c vs B5a:** the radius trend of the plate-end stiffening (the crude "off" rule failed at r 1.5).
- **B5d vs B5a:** the joint-type trend. 4- and 5-strut plate joints have more in-plate material around each
  column; JOINT_FHS predicts +1.6 points more than B5a, plate_layer +0.5.

### plate_layer test (ours only, 2026-10-13): fails the targets, kept as an option (default off)

`jointSection.element_factors_aniso(plate_layer=True)`: at a strut end on a plate, remove from each slice the
material within the plate-bonded layer (height = radius of the in-plate struts at that node), except the strut's
own circle. Errors vs Abaqus, off → on:

| case | metric | off | plate_layer on |
|---|---|---|---|
| B4a linear (prescribed) | end of ramp / ramp work | +21.0 / +13.0 % | +11.1 / +3.5 % |
| B4b EPU fast (prescribed) | end of ramp / ramp work | +29.5 / +17.9 % | +18.6 / +7.5 % |
| B3a / B3b (no plate joints) | end of ramp | -7.8 / +2.5 % | unchanged ✔ |
| H1e BCC r 1.0 | stroke / E75 / peak | -11.7 / +10.8 / +26.3 % | -5.6 / +3.3 / +14.9 % |
| H1f BCC r 1.5 | stroke / E75 / peak | -5.6 / +5.0 / +4.1 % | +5.0 / -7.0 / -1.2 % |
| F1e BCC r 1.0 (lin) | stroke / E75 / peak | -1.6 / -5.4 / +28.2 % | +2.6 / -13.0 / +27.2 % |
| F1f BCC r 1.5 (lin) | stroke / E75 / peak | -1.1 / -6.8 / -4.5 % | +5.9 / -16.8 / +0.2 % |
| H1c C4 | stroke / E75 / peak | +3.4 / -3.5 / -10.3 % | +3.5 / -3.5 / -10.1 % |
| H1d C12 | stroke / E75 / peak | -3.5 / +3.2 / -11.3 % | -2.3 / +2.0 / -18.2 % |
| H1a C4 | stroke / E75 / peak | -1.5 / -0.4 / -3.3 % | -1.6 / -0.4 / -2.4 % |

- Halves the B4 / H1e excess but does not reach ±5 %; makes r 1.5 (H1f, F1f) and F1e too soft in energy, C12 peak
  worse. **Not adopted**: plate_layer=False stays the default; the B5 runs decide the plate-end factor directly.

### Results 2026-10-15: Level B5 (all 4 runs; Abaqus completed, ETOTAL drift ≤ 0.01 % of max KE)

Reaction at the end of the ramp, ours vs the Abaqus solid (error = ours / solid − 1):

| case | solid [N] | solid / bare | bare | JOINT_FHS | JOINT_FHS + plate_layer |
|---|---|---|---|---|---|
| B5a 1×1 r 1.0 linear | 189.4 | 1.118 | -10.6 % | **-1.0 %** | -9.3 % |
| B5b 1×1 r 1.0 EPU 45 fast | 25.99 | 1.064 | -6.0 % | **+4.0 %** (peak +3.8 %) | -4.7 % |
| B5c 1×1 r 1.5 linear | 452.9 | 1.188 | -15.9 % | **-1.9 %** | -14.0 % |
| B5d 2×2 r 1.0 linear | 442.1 | 1.160 | -13.8 % | **-3.2 %** | -12.2 % |

nRMSE of the reaction history (ours JOINT_FHS in compare.py): 1.3 % (B5c), 2.6 % (B5d); driven-plate U3 ≤ 1.4e-5.

- **The plate-end joint factors are right for columns alone.** On cells with no internal joint the solid is 6-19 %
  stiffer than bare beams and JOINT_FHS reproduces it within -3.2 / +4.0 %, at both radii (r 1.5: -1.9 %), and with
  3 / 4 / 5 struts at the plate joint (B5d, -3.2 %). Bare beams are 6-16 % too soft.
- **`plate_layer` is rejected for good:** it removes the stiffening the solid really has (-9 to -14 %). It stays in
  the code as a documented option, default off. The B4 "double counting the clamp" hypothesis is wrong.
- **Joint-type trend is small and in the right direction:** solid stiffening 1.118 (1×1) → 1.160 (2×2), JOINT_FHS
  1.107 → 1.123; the model catches about a third of the rise (-1.0 % → -3.2 %).
- **EPU 45 (B5b):** solid stiffening is 1.064 against 1.118 linear (rate dependence, as in B3); ours stays at 1.107 and
  is +4 % too stiff. Same direction as B3, much smaller than the B4 error.
- **So the B4 / H1e excess (+21-30 %) is not a plate-corner effect.** B3 (centre node + one diagonal per corner) and B5
  (columns at the corners) are each right, but the full BCC cell (B4) is not: the solid shows no stiffening over bare
  beams (1.02), JOINT_FHS +23 %. The error appears only when a column and a diagonal meet at the same plate corner.
- **Working hypothesis (not tested):** the fused-section slice perpendicular to a column counts the area and second
  moment of the oblique diagonal cut as if it carried the column's load; the inclined diagonal is only partly in the
  column's load path. Alone (B3, B5) there is nothing to over-count.

**Next (B6, cheap, before any code change):**
1. Offline: split JOINT_FHS's B4 factors by which neighbour strut contributes to each slice, then weight inclined
   neighbours by their load-path alignment (cos² to the strut axis) and see whether B4a / B4b / H1e / H1f / F1e / F1f /
   B5a-d all fall within ±5 % together (the B5 cases must not move).
2. If it works, confirm with one new Abaqus case that separates the hypothesis from alternatives: BCC with thin
   diagonals (r_d 0.5, columns r 1.0, linear, same prescribed motion). Hypothesis predicts the excess scales with
   the diagonal area; plate-corner "clamp" or centre-node explanations do not.

### Results 2026-10-15: Level B6 (B4 re-run with the current FHS geometry; both runs completed)

Reaction at the end of the ramp, ours vs the Abaqus solids (error = ours / solid − 1):

| case | solid, current geometry [N] | solid / bare | solid, old geometry (B4) | bare | JOINT_FHS | + plate_layer | simple valence rule a = 1 |
|---|---|---|---|---|---|---|---|
| B6a linear | 456.0 | 1.137 | 408.9 (1.020) | -12.1 % | +9.8 % | **0.0 %** | +15.2 % |
| B6b EPU 45 fast | 66.93 | 1.063 | 59.23 (0.940) | -5.9 % | +16.0 % (peak +11.7 %) | +5.4 % (peak +1.6 %) | +23.3 % |

- **The B4 anomaly was mostly the old-geometry solid.** The same BCC cell with the current FHS geometry is 11.5 % (linear)
  and 13 % (EPU) stiffer than the old one; the old solid sat at bare beams (1.02), the new one at 1.14 like the simple
  cubic cells. Every stiffness conclusion drawn from the old BCC solids (B4, H1e, H1f, F1e, F1f) is therefore suspect.
  The old and new geometry have the same connected column section (1.0 pi r^2 at 2.5-4 mm) and JOINT_FHS slicing sees
  only ~1.5 % difference, so the slicing is blind to what changed (corner / plate region).
- **With the new geometry the solid is additive:** SC columns 1.118 x 339 = 379 N/mm + B3 diagonals 1.25 x 62 = 78 N/mm =
  457 N/mm against 456 measured. (The "10 % non-additivity" noted earlier came from the old solid.)
- **Why the models overshoot on BCC:** the oblique diagonal at the corner lengthens the geometric zone of the
  column from 1.0 r (90 deg plate edges) to 1.93 r (54.7 deg diagonal) and adds area, but the solid shows the column's
  stiffening is that of the simple cubic corner. JOINT_FHS and the zone model both over-count oblique neighbours.
- **EPU: the stiffening is smaller than linear in all three cells (B3 1.25 → 1.08, B5 1.118 → 1.064, B6 1.137 →
  1.063), ours barely moves (1.249 → 1.233):** a second, separate error source (material / rate), +16 % in B6b.

**Simple angle-weighted rule (offline, `notes/18` section 6):** k_end = 1 + a * sum_nb (r_b/r_a)^2 sin^p(theta_ab), zone
length l_end = max_nb sin^p(theta) * (r_b + r_a cos theta)/sin theta (capped L/2). p = 4, a = 1 (linear), a = 0.5 (EPU 45):

| case | a = 1 | a = 0.5 |
|---|---|---|
| B5a / B5c / B5d (SC) | +0.1 / 0.0 / -2.7 % | -2.7 / -4.5 / -5.2 % |
| B6a (BCC, new geometry) | **-0.2 %** | -3.0 % |
| B3a centre node, linear | **-12.6 %** | -14.3 % |
| B5b / B3b / B6b (EPU) | +5.4 / -0.6 / +6.5 % | **+2.3 / -1.8 / +3.2 %** |

Without the angle weight (p = 0, valence only) B6a is +15 %; weighting only k but not the zone length helps little
(+11-13 %); weighting both fixes it. Fitted in sample on 8 cells with 2 numbers (a, p): **needs a hold-out** (impact cases F / H,
graded radii, octet) before use. Open: the centre node (B3a, linear) stays 12 % too soft for any a (needs a longer
zone); the material factor (a_EPU ~ 0.5 a_lin) is one more fitted constant.

## Level B7 / B8 and the closed-form joint rule (2026-10-16)

Why: after B6 the stiffening of all new-geometry solids (B3, B5, B6) is consistent and additive, but no simple rule had been
validated out of sample. `src/jointRule.py` is a closed-form, differentiable rule (k_end = 1 + mu a W, W = sum (r_b/r_a)^2 sin^p theta;
zone length l_end = zl * smooth-max sin^p theta (r_b + r_a cos theta)/sin theta * (1 + b max(W-2, 0))). Fitted on B3 + B5 + B6 with
`abaqus/fit_rule.py`, cross-validated by leaving a whole topology group out (`results/rule_fit/model_table.md`):

| model | parameters | in sample rms | held out: SC / centre node / BCC cell | pooled held-out |
|---|---|---|---|---|
| bare | - | 12.9 % | - | - |
| zone k = 3.5 (old, differentiable) | 1 (fitted on B2a) | 9.2 % | - | - |
| JOINT_FHS (slicing, not differentiable in r) | 0 | 6.8 % | - | - |
| rule M0 (a) | 1 + mu | 6.0 % | 5.2 / 9.1 / 17.0 % | 10.2 % |
| rule M1 (a, p) | 2 + mu | 4.7 % | 2.8 / 8.7 / 16.9 % | 9.6 % |
| **rule M3 (a, p, zl, b)** | 4 + mu | **1.9 %** | **4.4 / 6.6 / 4.3 %** | **5.2 %** |

- **Honest reading:** only M3 survives the held-out BCC cell (4.3 %); M0 / M1 drop the angle weight (p = 0) when the cell is left out and miss it by +17 %.
  M3's parameters are not identifiable across folds (a 0.4-3.9, zl 0.5-1.5, b 0.2-0.8, mu 0.16-0.41) although its predictions are; the pooled
  5.2 % beats JOINT_FHS (6.8 %, no fit) only modestly, with 5 fitted numbers on 9 cases.
- **The EPU factor mu is not a material constant:** fitted 0.17-0.41 on the fast prescribed ramps, but with it the C4 H cells are 14-16 % too soft
  in stroke; mu = 1 is comparable to JOINT_FHS. Not used for impact.
- **Impact hold-out** (`results/rule_fit/impact_holdout.md`, B2 / F / H solids, never fitted; mu = 1): mean |error| stroke 3.7 % vs 3.5 % for the recorded
  models, energy at 50 % crush 9.7 % vs 6.4 %, peak force 11.1 % vs 8.7 %. **At parity on stroke, worse on the energy curve and peak, especially the
  bending-dominated C4 cells (E50 -12 to -18 %)**: the rule was fitted on axial-dominated prescribed-motion cells; a separate bending stiffening
  was never tested on bending cells.
- **What the rule does give:** the exact design gradient with the joint in the loop (adjoint = unrolled AD 1e-9, = finite differences 1e-6;
  `tests/test_joint_rule_adjoint.py`), no FHS rebuild or slicing per design (milliseconds), and stroke accuracy at the level of JOINT_FHS.
  It does NOT yet beat JOINT_FHS in accuracy.

### New Abaqus runs (inputs written, run in this order; est. wall time, small meshes)

| case (abaqus/inp) | specimen | mesh | est. time |
|---|---|---|---|
| B7a_column_oblique30_linE0 | column + one oblique strut from the same plate node, 30 deg, r 1.0, H 6.35, linear, 0.25 mm | 3.5k C3D10M | ~3 min |
| B7b_column_oblique45_linE0 | same, 45 deg | 3.0k | ~3 min |
| B7c_column_oblique60_linE0 | same, 60 deg | 3.6k | ~3 min |
| B8a_bcc_node_n4_linE0 | centre node with 4 diagonals (2 up, 2 down), as B3a otherwise | 11.2k | ~5 min |
| B8b_bcc_node_n6_linE0 | centre node with 6 diagonals | 13.7k | ~5 min |

Then `PYTHONPATH=src:tests python abaqus/b78_predict.py` (predictions are already in `results/B78_summary.md`, recorded before the runs).

**Blind predictions** (reaction at the end of the ramp / bare beams):

| case | JOINT_FHS | rule M3 | rule without angle weight (p = 0) |
|---|---|---|---|
| B7a 30 deg | 1.224 | **1.003** | 1.394 |
| B7b 45 deg | 1.157 | **1.025** | 1.250 |
| B7c 60 deg | 1.122 | **1.067** | 1.172 |
| B8a n = 4 | 1.047 | 1.035 | 1.050 |
| B8b n = 6 | 1.102 | 1.172 | 1.088 |

**What decides what:** B7a separates the angle hypotheses by 40 points (M3 1.00, JOINT_FHS 1.22, no angle weight 1.39): a solid near 1.0 confirms the
sin^p weight, near 1.2-1.4 means acute neighbours do stiffen (the geometric zone is 3.7 r at 30 deg) and p must drop. B8a/b test the valence growth b
(M3 and the others differ by up to 8 points at n = 6).

### Data housekeeping (2026-10-16)
The 188 `*_field_U*.csv` frame dumps (7.0 GB; nothing reads them) were removed from the tree and git-ignored; they stay in git history and can be re-extracted
from the odb. `abaqus/README.md` updated (copy only the history CSV).

### Results 2026-10-17: B8 valid; B7 invalid and redesigned (supersedes the B7 definition and predictions above)

**B7 (first version) was invalid and its outputs were deleted.** The solid reaction was 24.5 / 16.0 / 5.9 N against 70 / 57 / 48 N for bare beams, i.e. the
oblique strut alone. Cause: in the .inp, the vertical column ended 5 um short of BOTH plates (nodes at z = 6.345 instead of 6.350), so neither end was in the BOT /
TOP sets and the column was neither clamped nor tied. A strut end lying exactly ON the plate plane gets its end ring pushed 5 um to the wrong side by
`nodeJoint.PLATE_GAP`; in B3 / B5 / B6 / B8 it never mattered (oblique ends are cut at an angle; B5 / B6 columns have in-plate struts). Audit of all other prescribed-motion
solids (B3a-c, B4a, B5a-d, B6a-b, B8a-b): every plate end reaches its plate. Safeguards added: `cases._check_plate_ends` (raises if a plate strut end has < 6 mesh nodes
on its plate) and `extend_ends` (single-strut plate ends are moved outward along the strut before the trim, so the cut face is full).

**New B7a-c (inputs regenerated, to be re-run):** a vertical column (r 1.0, H 12.7) with a T-joint at mid height and ONE oblique strut from that node up to the top plate,
at 30 / 45 / 60 deg to the column; 0.5 mm, linear; 4.2k / 3.9k / 5.4k C3D10M; each plate end has a full 80-node face. Blind predictions (hold value / bare beams):

| case | JOINT_FHS | rule M3 | rule without angle weight (p = 0) |
|---|---|---|---|
| B7a 30 deg | 1.191 | **1.002** | 1.159 |
| B7b 45 deg | 1.133 | **1.016** | 1.141 |
| B7c 60 deg | 1.097 | **1.044** | 1.111 |

(The earlier numbers 1.00 / 1.39 were for the invalid geometry.) The discriminating run is B7a: the angle-weighted rule predicts no stiffening from an acute neighbour,
the others 16-19 %.

**B8 (valid, blind):** solid reaction vs the models. The linear solid RINGS after the ramp (hold: +-5.5 % B3a, +-5.8 % B8a, +-6.1 % B8b, +-2.7 % B6a, +-0.6 % B5a; the beam
model does not ring), so a single ramp-end or end-of-hold sample of the solid carries that noise. The static value is the MEAN over the hold window:

| case | n | solid ramp-end | solid hold mean | solid hold / bare | JOINT_FHS | rule M3 | no angle weight |
|---|---|---|---|---|---|---|---|
| B8a | 4 | 36.20 | 34.67 | 1.112 | 1.053 (-5.3 %) | 1.038 (-6.6 %) | 1.056 (-5.0 %) |
| B8b | 6 | 56.45 | 54.18 | 1.158 | 1.094 (-5.6 %) | **1.153 (-0.5 %)** | 1.091 (-5.8 %) |
| B3a | 8 | 78.08 | 75.90 | 1.217 | 1.119 (-8.0 %) | 1.193 (-1.9 %) | - |

- **The stiffening of an interior node grows steadily with its valence in the solid:** 1.112, 1.158, 1.217 for n = 4, 6, 8 (about +0.03 per extra strut).
- **JOINT_FHS under-predicts by 5-8 % at every valence;** the rule without angle weight likewise.
- **M3's valence growth is supported at n = 6 (blind, -0.5 %) and n = 8 (-1.9 %), but not at n = 4 (-6.6 %):** the solid is already 11 % stiffer than bare beams
  with only 4 diagonals, while the rule gives 4 %. M3's growth is convex in n; the data are close to linear. So M3 is neither validated nor refuted: one blind hit, one miss.
- **Methodology correction:** the B3 / B8 centre-node errors quoted so far (e.g. B3a -12.6 % for the old rule, -7.8 % for JOINT_FHS at the ramp end) included the solid's ringing
  (+-5-6 %); with the hold mean they are -8.0 % (JOINT_FHS) and -1.9 % (M3). All earlier fits used single samples (ramp end, hold end) of the solid. They should be redone with the
  hold-window mean (and with B7 / B8 added) before any parameter value is trusted.

### Results 2026-10-17: B7 (valid) and the second fit (hold-window mean, 10 linear specimens, leave-one-group-out)

**B7a-c (T-joint at mid height + one oblique strut, solid hold mean / bare beams):** 30 deg **1.292**, 45 deg **1.231**, 60 deg **1.160** (ringing only 0.2-0.4 %).
Predictions: JOINT_FHS 1.190 / 1.129 / 1.091 (8-9 % low, right trend), first rule M3 1.002 / 1.016 / 1.045 (**refuted**: it said an acute neighbour adds nothing),
rule without angle weight 1.159 / 1.136 / 1.108 (flat in angle). **The more acute the neighbour, the stiffer the joint** (geometric zone 3.7 r at 30 deg, 1.6 r at 60 deg):
the sin^p weighting of the first fit was wrong in sign; the original geometric zone length was right. One oblique neighbour at mid height raises the column's reaction by
+35 % at 30 deg (the joint region is a stiff truss-like fused web, not only a shortened end).

**Second fit** (`abaqus/fit_rule2.py`; data B3a, B5a/c/d, B6a, B7a-c, B8a/b; residual = hold-window mean of ours / solid - 1; groups SC, STAR, TJOINT, CELL left out in turn):

| model (free) | in-sample rms / max | held-out per group (SC / STAR / TJOINT / CELL) | fitted |
|---|---|---|---|
| zone k = 3.5 (old, 1 constant) | 6.1 / 11.5 | - | - |
| JOINT_FHS (no fit) | 6.1 / 8.5 | - | - |
| R0 (a, zl) | 6.1 / 11.9 | 1.8 / 7.5 / 12.9 / 18.6 | a 5.94 (bound 6), zl 0.76 |
| R1 (+ gamma) | 4.1 / 7.1 | 5.5 / 6.4 / 14.6 / 3.9 | a 5.9, zl 0.89, gamma 0.0 |
| R2 (+ b) | 3.0 / 4.0 | 3.4 / 6.3 / 12.6 / 16.2 | a 5.27, zl 0.85, gamma 0.01, b 0.16 |
| **R3 (+ p)** | **2.3 / 4.7** | **1.0 / 5.1 / 17.0 / 4.0** | a 5.69, zl 0.74, gamma 0.0, b 0.19, **p = -0.81** |

- **Every fit pushes the gain a to its bound and gamma to 0:** the rule becomes a **rigid end zone** of length ~0.75-0.9 x the geometric zone, with in-plate-vs-oblique pairs at plates
  ignored (gamma = 0) and a small growth with valence (b ~ 0.2). The standard "rigid extremity" model of the literature, with a geometric, radius-dependent, differentiable length.
- **R3's negative angle exponent (more weight to acute neighbours) is cross-validated:** trained on SC + STAR + TJOINT it predicts the held-out BCC cell within +4.0 % (the first rule
  needed the BCC cell itself to get p; R0 / R2 miss it by 16-19 %), and held-out SC within 1.3 %, STAR within 6.1 %.
- **TJOINT held out fails for every model (13-17 %):** B7 is the only acute-angle data, so without it the fit cannot know; this is a data limit, not a model failure.
- Parameters are stable across folds except in the TJOINT fold (which drops to a ~ 0.5 and p > 0, the old optimum).

**Impact hold-out of R3 (never fitted on impact; `results/rule_fit2/impact_holdout_R3.md`), mean |error| stroke / E50 / E75 / peak:**

| cells | recorded models | R3 |
|---|---|---|
| C4 and BCC (12: B2a/c/e/f, F1a/c/e/f, H1a/c/e/f) | 3.5 / 6.8 / 4.1 / 10.5 | **3.1 / 6.2 / 4.1 / 12.6** |
| C12 and octet (5: B2d/g/h, F1d, H1d) | 3.4 / 5.3 / 5.6 / 4.3 | **16.3 / 25.5 / 23.2 / 28.3** |

- **On C4 and BCC R3 is at least as good as JOINT_FHS in stroke and energy** (the C4 design lattice: B2a stroke -2.4, E50 +0.7 vs +2.1, -6.7) **but over-stiff by 10-28 % on C12 and octet**
  (B2h octet r 1.5: stroke -28 %, peak +69 %).
- Bounding the gain does not help (a = 1.5: prescribed 4.1 %, B2h -16 %; a = 3: 2.7 %, B2h -24 %): the over-stiffening comes from the zone LENGTH and the number of acute neighbours at
  valence 6-12 nodes (zones overlap and cover most of a short strut), not from the gain. No prescribed-motion specimen has short struts or valence > 8 with acute angles,
  so the fit never saw this.

### Conclusions and status
1. The B4 / B7 / B3 anomalies were specimen or measurement defects (old geometry, floating column, solid ringing); with correct data the physics is consistent: the joint stiffens more the
   more acute the neighbour and the higher the valence, and the stiffening is additive over struts.
2. **R3 is a validated differentiable joint model for the C4 design lattice and BCC-type cells** (held-out specimens within 1-6 %, impact within the accuracy of JOINT_FHS) with the exact adjoint gradient
   (`tests/test_joint_rule_adjoint.py`); **it must NOT be used for C12 / octet / valence > 8 until constrained.** JOINT_FHS stays the reference where the lattice has short struts or many acute neighbours.
3. Not done / next: a prescribed-motion specimen with short struts and high valence (a C12 cell and an octet cell, linear, 0.5 mm; B9) to constrain zone length and cap at high valence, then a refit
   with leave-one-group-out including them; the EPU 45 factor (still unresolved, mu does not transfer); a derivative of the reaction w.r.t. the rule parameters through the solver (the fits are gradient-free).

## Level B9: short-strut and high-valence unit cells under prescribed plate motion (2026-10-16)

Why: the rule R3 (rigid zone ~0.74 x geometric length, p = -0.81) is held-out accurate on the long-strut specimens and on C4 / BCC impact, but over-stiffens the C12 and octet
IMPACT solids by 10-28 % (notes/18 section 11); no prescribed-motion specimen has short struts or the octet's valence-6/8 acute nodes. B9 supplies them cleanly: one unit cell
each, current FHS geometry, linear E0, top plate down 0.5 mm in 0.3 ms then hold 0.2 ms, bottom clamped, no contact (as B6). Plate-end audit passed (every plate node has a full face).

| case (abaqus/inp) | cell | mesh | est. wall time |
|---|---|---|---|
| B9b_octet_r15_linE0 | octet r 1.5 (L 8.98, L/r 6.0, valence 3/6/8) | 45.8k C3D10M, h 0.75 | ~25 min |
| B9a_octet_r10_linE0 | octet r 1.0 (L/r 9.0) | 54.7k, h 0.5 | ~30 min |
| B9c_C12_r05_linE0 | C12 r 0.5 (96 struts, L 2.3-2.9, L/r 4.6-5.7, valence 2/3) | 102.6k, h 0.25 | ~1-1.5 h |

Order: B9b, B9a, B9c. Then `PYTHONPATH=src:tests python abaqus/b9_predict.py` (blind predictions already in `results/B9_summary.md`). Compare the solid's HOLD-WINDOW MEAN (the linear solid rings).

**Blind predictions** (hold value / bare beams; the stiffening is large: these cells are joint dominated):

| model | B9a octet r 1.0 | B9b octet r 1.5 | B9c C12 r 0.5 |
|---|---|---|---|
| bare beams | 1.000 (127.5 N) | 1.000 (324 N) | 1.000 (0.81 N) |
| zone k = 3.5 | 1.454 | 1.839 | 3.128 |
| JOINT_FHS | 1.480 | 1.908 | 2.289 |
| rule R1 | 1.567 | 2.173 | 3.737 |
| rule R2 | 1.727 | 2.691 | 3.547 |
| rule R3 | 1.812 | 3.012 | 3.484 |

**What decides what:** the models differ by up to 1.6x on these cells. A solid near JOINT_FHS (1.5 / 1.9 / 2.3) confirms that the rule's rigid-zone saturation is the cause (and that
the impact over-stiffening was real); a solid near R3 would instead point at the impact solids / dynamic effects. The two new options in `jointRule` (zone cap `cap`, saturation of the neighbour
sum `wsat`; models R4 / R5 in `fit_rule2.py`) are fitted only AFTER B9, with the impact cells kept as the hold-out.


### Results 2026-10-16: B9 (octet r 1.0, octet r 1.5, C12 r 0.5; all runs completed, solid hold-window mean)

| model | B9a octet r 1.0 | B9b octet r 1.5 | B9c C12 r 0.5 |
|---|---|---|---|
| **Abaqus solid (hold mean / bare)** | **1.592** (203.0 N) | **2.431** (787.5 N) | **3.276** (2.653 N) |
| zone k = 3.5 | -8.7 % | -24.3 % | **-4.5 %** |
| JOINT_FHS | -7.1 % | -21.5 % | -30.1 % |
| rule R1 | **-1.6 %** | -10.6 % | +14.1 % |
| rule R2 | +8.4 % | +10.7 % | +8.3 % |
| rule R3 | +13.8 % | +23.9 % | +6.4 % |

- **No existing model is within +-5 % on all three cells.** JOINT_FHS, the zone model and R1 under-predict the octet (r 1.5: -22 / -24 / -11 %); R2 and R3 over-predict it; the truth (2.43 x bare) lies between R1 (2.17) and R2 (2.69).
  JOINT_FHS under-predicts C12 by 30 % (stiffening 2.29 vs 3.28 for the solid).
- **Ringing:** the octet solids ring by only 0.3-0.9 %; the C12 solid rings by 64 % of its mean with a period (~0.2 ms) equal to the hold window, so the C12 static value is uncertain by ~ +-5 %
  (the window means 2.65 / 2.93 depending on whether the first 30 % of the hold is skipped; ours R3 2.82 / 3.20). B9c should be re-run with a long hold (1 ms) before it is trusted below ~ 5 %.
- **Correction of the impact hold-out (notes/18 11):** the B2 solids (B2a-h) use cylinder + sphere joints, NOT the FHS joints the rule was fitted on, and the FHS octet impact solids (F1g/h) were never run, so the claim that
  R3 over-stiffens the OCTET in impact was a comparison with sphere-joint solids and is not a valid hold-out. The valid FHS impact solids are F1a/c/d/e/f and H1a/c/d/e/f only (stroke, R3 vs recorded model):
  F1a -6.5 / -7.3, F1c +0.7 / -0.4, F1d -10.3 / -5.1, F1e +0.3 / -2.2, F1f +0.2 / -2.0; H1a -2.6 / -5.1, H1c +3.7 / +0.7, H1d -13.7 / -6.6, H1e -9.9 / -12.6, H1f -4.7 / -7.1. R3 is better on 6 of 10, worse on the two C12 cells (F1d, H1d).
  (The BCC ones use the old geometry.) Impact stroke is also a weak test of the joint: the static B9 shows JOINT_FHS 21-30 % too soft on B9b / B9c while its impact strokes are within a few %.
- **Cost:** the rigid-zone rule needs 1.6-5.6 x more explicit steps than bare beams (B6a 5.5 x, B9b 5.6 x, C12 2.0 x); JOINT_FHS needs 1.8-3.4 x.

### Results 2026-10-16: refit with B9 (13 linear specimens, 21 fits: R3, R4, R5 each with 6 groups left out), selected model R5

Data: B3a, B5a/c/d, B6a, B7a-c, B8a/b, B9a-c (C12 weighted 0.5, its solid rings with a period ~ the hold window). Groups left out in turn: SC, STAR, TJOINT, CELL, OCT (B9a/b), C12 (B9c).

| model (free parameters) | in sample rms / max | held-out per group (SC / STAR / TJOINT / CELL / OCT / C12), max |abs| | pooled held-out rms / max |
|---|---|---|---|
| zone k = 3.5 (1 constant) | 9.0 / 24.3 | - | - |
| JOINT_FHS (none) | 11.7 / 30.1 | - | - |
| R3 (a, zl, gamma, b, p) | 2.7 / 5.7 | 2.9 / 6.2 / 15.6 / 1.7 / 21.1 / 2.3 | 9.1 / 21.1 |
| R4 (R3 + zone cap) | 2.9 / 5.7 | 2.7 / 6.2 / 13.5 / 1.8 / 13.0 / 2.4 | 7.1 / 13.5 |
| **R5 (R3 + saturation wsat of the neighbour sum)** | **2.7 / 5.5** | **3.0 / 6.2 / 9.7 / 2.0 / 7.8 / 1.5** | **5.0 / 9.7** |

**R5 = the selected rule:** k_end = 1 + a wsat tanh(W / wsat), W = sum (r_b/r_a)^2 sin^p(theta); zone length = zl x geometric (r_b + r_a cos theta)/sin theta x sin^p, x (1 + b (W - 2)+), cap L/2;
a = 3.45, zl = 0.77, gamma = 0, b = 0.15, p = -0.77, wsat = 2.47 (k_end <= ~9.5). Held-out errors (ours / solid - 1): SC -1.3, -1.1, -3.0; STAR -4.6, -4.5, -6.2; TJOINT -9.7, -6.9, -4.0;
CELL +2.0; OCT +7.8, +2.7; C12 -1.5 %. The in-sample fit is within 5.5 % on all 13 cases (B9a +4.0, B9b -0.5, B9c +0.7).

- **The octet failure of R3 was a calibration problem, not a model-form one:** refitting R3 with B9 brings the gain from 5.7 to 4.5 and the octet to +3 / 0 % in sample; but held out R3 still misses the octet by +12.6 / +21.1 %
  (a neighbour sum that grows without bound at valence 6-8), and saturating the neighbour sum (R5) fixes it (+7.8 / +2.7 %). The zone cap is not needed (R4's cap stays at its 0.5 bound).
- **Remaining weakness: the T-joint group** (B7, held out -9.7 / -6.9 / -4.0 %): B7 is the only acute-angle data; the angle exponent (p = -0.77) and the zone growth at acute angles rest on it.
- **Parameters are not tightly identified** (6 numbers on 13 cases; leave-out values of a range 1.3-5.6, wsat 2.8-11.6) but the predictions are stable; the held-out error is the honest accuracy: ~5 % rms, 10 % worst.
- **Differentiable and tested:** `tests/test_joint_rule_adjoint.py` passes for the first-rule parameters and for R5 (adjoint = unrolled AD to 1e-9, = finite differences to 1e-6).
- **Cost:** R5 needs 1.5-2.9 x the explicit steps of bare beams (JOINT_FHS 1.8-3.4 x); the element factors reach 9.4.
- **Impact check on the FHS solids** (`results/rule_fit3/impact_check_R5.md`; the F / H solids are OLD geometry, the BCC ones 11-13 % too soft): mean |error| stroke / E50 / E75 / peak, JOINT_FHS vs R5: C4 3.4 / 3.8 / 2.9 / 10.8 vs 3.2 / 6.7 / 3.6 / 14.3;
  BCC 6.0 / 8.4 / 6.6 / 15.5 vs 3.4 / 9.2 / 8.1 / 12.9; C12 5.8 / 5.1 / 7.5 / 4.4 vs 10.4 / 11.6 / 12.9 / 25.2. Because those solids are the soft old-geometry ones and impact stroke is a weak stiffness test (JOINT_FHS is 21-30 % too soft on the static B9b / B9c but within a few % in impact), this is
  only a consistency check; it is not evidence against R5 on C12.

### What is still missing (plan, in priority order)
1. **A prescribed-motion C4 cell (B10: r 0.8 and r 0.6, linear, current geometry).** C4 is the design lattice, and R5 has no static C4 data at all (the only C4 evidence is impact vs old-geometry solids). Cheap (42k / 62k elements, ~0.5 h).
2. **Unequal radii (B11).** The design variables are radii, but every specimen has equal radii: the (r_b / r_a)^2 dependence of the neighbour sum, hence the sign and size of dk/dr that the adjoint returns, is untested. T-joint (B7-type) with r 1.0 / 0.6 in both orders and a graded C4 cell.
3. **Impact solids with the CURRENT FHS geometry.** All F / H solids are old geometry (soft by 11-13 % in BCC). Regenerate F1d (C12) and F1e (BCC, 3 min) first, then F1a / H1a (hours); octet F1g / h can now be meshed with the route used for B9. Needed to validate contact + dynamics + densification with the new joint model.
4. **B9c with a long hold (1 ms)**, because its static value is uncertain by ~5 %; and an EPU 45 prescribed series (the joint stiffening in the fast EPU ramps is 6-17 % lower than the linear one and no constant transfers to impact).
5. **Section shape.** The rule uses the radius only; non-circular (VAE) sections are untested (back burner).
6. **Design-loop integration.** Use R5 in the plateau-design optimisation (notes/9) instead of k = 3.5 and compare designs; then verify the optimised design with an FHS solid.
7. **Fit through the solver** (derivative of the reaction w.r.t. the 6 parameters) instead of the gradient-free fit, to tighten the parameters (low priority).

## Run list 2026-10-17: B10, B11, B9d and the impact solids in the CURRENT FHS geometry (F2 / H2)

Why (notes/18 section 13): the selected rule R5 has no static C4 data (the design lattice), no unequal-radius data (the design variables), and every F / H impact solid is old geometry
(BCC 11-13 % too soft), so there is no valid impact validation. All inputs are written (`abaqus/inp/`), every plate end audited (`cases._check_plate_ends`), blind predictions recorded
(`results/B10_B11_summary.md`, `results/F2H2_summary.md`). Copy back as before: the history CSV into `abaqus/csv/`, `.sta` / `.dat` / `.msg` into `abaqus/logs/` (no `_field_U` files needed).
Then `PYTHONPATH=src:tests python abaqus/bx_predict.py` / `abaqus/impact_predict.py` and `abaqus/compare.py <case>`. Compare linear solids by the HOLD-WINDOW MEAN.

| group | case (abaqus/inp) | what | elements | est. wall time |
|---|---|---|---|---|
| A (minutes) | B11a_T45_col10_obl06_linE0 | T-joint 45 deg, column r 1.0, oblique r 0.6 | 13.1k | ~5 min |
| | B11b_T45_col06_obl10_linE0 | 45 deg, column 0.6, oblique 1.0 | 11.5k | ~5 min |
| | B11c_T30_col10_obl06_linE0 | 30 deg, column 1.0, oblique 0.6 | 14.8k | ~5 min |
| | B11d_T30_col06_obl10_linE0 | 30 deg, column 0.6, oblique 1.0 | 9.7k | ~5 min |
| | F2e_unit_cell_BCC_FHS_solid_r100_h050 | BCC r 1.0 impact (linear E = 40), 0.5 ms | 42.9k | ~3-5 min |
| | F2f_unit_cell_BCC_FHS_solid_r150_h075 | BCC r 1.5 impact | ~27k | ~2 min |
| | F2g_unit_cell_octet_FHS_solid_r100_h050 | octet r 1.0 impact (first FHS octet impact solid) | ~55k | ~5 min |
| | F2h_unit_cell_octet_FHS_solid_r150_h075 | octet r 1.5 impact | ~46k | ~5 min |
| | H2e_unit_cell_BCC_FHS_EPU45_solid_r100_h050 | BCC r 1.0 impact, EPU 45 | 42.9k | ~5 min |
| B (~1-1.5 h) | B10a_C4_r08_linE0 | C4 cell r 0.8, prescribed motion, hold 0.5 ms | 45.4k | ~50 min |
| | B10b_C4_r06_linE0 | C4 cell r 0.6, hold 0.5 ms | 58.9k | ~65 min |
| | B10c_C4_graded_top10_bot06_linE0 | graded C4 (top 1.0, bottom 0.6, mid 0.8), hold 0.3 ms | 102.9k | ~80 min |
| | B10d_C4_graded_top06_bot10_linE0 | graded C4 (top 0.6, bottom 1.0), hold 0.3 ms | 98.3k | ~80 min |
| C (long) | B9d_C12_r05_linE0_longhold | C12 as B9c with a 1.2 ms hold (static value) | 102.6k | ~1.6 h |
| | F2d_unit_cell_C12_FHS_solid_r050_h025 | C12 impact, 3 ms | 102.6k | ~1.8 h |
| | H2d_unit_cell_C12_FHS_EPU45_solid_r050_h025 | C12 impact, EPU 45, 4 ms | 102.6k | ~2 h |
| | F2a_unit_cell_C4_FHS_solid_r080_h040 | C4 r 0.8 impact (the design lattice), 3 ms | 45.4k | ~4.7 h |
| | H2a_unit_cell_C4_FHS_EPU45_solid_r080_h040 | C4 r 0.8 impact, EPU 45, 4 ms | 45.4k | ~4-5 h |

Run order: A, then B10a / B10b, then B10c / d, then B9d and F2d, then F2a (and the EPU C and H cases last). If time is short: A + B10a + B10b + B9d already answer the static C4, radius and C12 questions.

**Blind predictions, prescribed motion** (hold value / bare beams; JOINT_FHS | zone k = 3.5 | rule R5):

| case | JOINT_FHS | zone k = 3.5 | rule R5 | what decides |
|---|---|---|---|---|
| B10a C4 r 0.8 | 1.582 | 1.536 | 1.514 | static C4: models within 5 %; a solid outside 1.45-1.65 means the C4 joints (valence 3, L 2.6-6.4) are not captured |
| B10b C4 r 0.6 | 1.418 | 1.321 | 1.306 | thinner struts: models differ by 8 % |
| B10c graded 1.0 / 0.6 | 1.575 | 1.479 | 1.574 | unequal radii at every joint; R5 = JOINT_FHS here |
| B10d graded 0.6 / 1.0 | 1.577 | 1.483 | 1.576 | the reverse orientation |
| B11a 45 deg, col 1.0 / obl 0.6 | 1.068 | 1.117 | 1.101 | radius-ratio dependence (r_b / r_a)^2 |
| B11b 45 deg, col 0.6 / obl 1.0 | 1.198 | 1.179 | **1.313** | R5 +10 % above JOINT_FHS: the thin column feels the thick oblique strut |
| B11c 30 deg, col 1.0 / obl 0.6 | 1.108 | 1.188 | 1.192 | |
| B11d 30 deg, col 0.6 / obl 1.0 | 1.334 | 1.258 | **1.474** | R5 +10 % above JOINT_FHS |
| B9d C12 (long hold) | 1.367 | 1.437 | 1.472 | replaces B9c's uncertain static value |

If B11b / B11d come out near 1.2 / 1.33 the neighbour sum should use a weaker radius ratio than (r_b / r_a)^2 (this directly changes the design gradient d k / d r); near 1.3 / 1.47, the quadratic ratio is right.

**Blind predictions, impact** (max stroke [mm] / peak force [N]; bare | JOINT_FHS | rule R5; ours energy as in the input; full table with times and energies in `results/F2H2_summary.md`):

| case | bare | JOINT_FHS | R5 |
|---|---|---|---|
| F2e BCC r 1.0 | 0.843 / 57.9 | 0.753 / 67.5 | 0.776 / 68.4 |
| F2f BCC r 1.5 | 0.834 / 134.9 | 0.706 / 152.1 | 0.735 / 157.9 |
| F2g octet r 1.0 | 0.887 / 36.7 | 0.747 / 48.5 | 0.700 / 51.0 |
| F2h octet r 1.5 | 1.002 / 97.8 | 0.727 / 151.2 | 0.639 / 174.9 |
| H2e BCC r 1.0 EPU | 1.023 / 122.3 | 0.869 / 156.5 | 0.906 / 146.8 |
| F2d C12 r 0.5 | 9.529 / 4.8 | 8.183 / 5.7 | 7.851 / 7.7 |
| F2a C4 r 0.8 | 10.232 / 16.7 | 8.110 / 19.6 | 8.285 / 21.6 |
| H2d C12 r 0.5 EPU | 9.425 / 3.5 | 7.666 / 4.2 | 7.239 / 4.0 |
| H2a C4 r 0.8 EPU | 10.483 / 12.4 | 7.760 / 17.5 | 8.099 / 17.4 |

**Bookkeeping added the same day:** `CLAUDE.md` (repo rules: every background / long job reports to `run_status.log`), `tests/run_logged.py` (wrapper / watcher for any command), run_status reporting inside every long script.

### Results 2026-10-19: B10 (C4), B11 (unequal radii), B9d (long hold), F2 / H2 (impact in the CURRENT FHS geometry), and what they do to the joint model

**Static specimens (hold-window mean, error = ours / solid - 1; `results/B10_B11_summary.md`; compare.py had a name clash B10 / B11 vs B1 that hid the solid rows, fixed):**

| case | solid / bare | zone k = 3.5 | JOINT_FHS (lambda = r) | rule R5 (set-3 fit) |
|---|---|---|---|---|
| **B10a C4 r 0.8 (design lattice)** | **1.975** | -22.2 % | -19.9 % | **-23.3 %** |
| B10b C4 r 0.6 | 1.462 | -9.6 % | -3.0 % | -10.6 % |
| B10c graded 1.0 / 0.6 | 1.882 | -21.4 % | -16.3 % | -16.3 % |
| B10d graded 0.6 / 1.0 | 1.871 | -20.8 % | -15.7 % | -15.7 % |
| B11a T 45 deg, col 1.0 / obl 0.6 | 1.167 | -4.3 % | -8.5 % | -5.7 % |
| B11b T 45 deg, col 0.6 / obl 1.0 | 1.269 | -7.1 % | -5.6 % | **+3.5 %** |
| B11c T 30 deg, col 1.0 / obl 0.6 | 1.222 | -2.8 % | -9.3 % | **-2.5 %** |
| B11d T 30 deg, col 0.6 / obl 1.0 | 1.460 | -13.8 % | -8.6 % | **+0.9 %** |
| B9d C12 (1.2 ms hold) | 1.409 | +2.0 % | -3.0 % | +4.5 % |

- **Unequal radii: the quadratic radius ratio (r_b / r_a)^2 of the rule is confirmed** (B11b / d solid 1.269 / 1.460 vs R5 1.313 / 1.474, vs the predicted decision band "near 1.3 / 1.47 = quadratic is right"); R5 is within +-6 % on all four, JOINT_FHS under-predicts by 6-9 %.
  This is what the design gradient dk / dr relies on.
- **The C4 design lattice is under-predicted by every model by 16-23 %** (B10a, c, d; r 0.6 by 3-11 %). The solid windows are stable (B10a mean 21.38 N, halves 21.45 / 21.32, ringing 7 %), so it is not a measurement artefact. The solid's stiffening
  rises steeply with r (1.46 at r 0.6 -> 1.98 at r 0.8); no model follows. C4 nodes: valence 3, angles 90 / 120 / 120 deg, strut lengths 6.35 / 6.35 / 2.63 mm (L/r = 3.3 for the short strut at r 0.8).
- **Impact in the current geometry (valid solids; `results/F2H2_summary.md`), stroke error JOINT_FHS / R5:** F2a C4 +11.6 / +14.0 %, F2d C12 -0.6 / -4.6, F2e BCC r 1.0 +2.0 / +5.1, F2f BCC r 1.5 +3.4 / +7.6, F2g octet r 1.0 +6.6 / -0.1, F2h octet r 1.5 +15.2 / +1.3,
  H2e BCC EPU -5.7 / -1.7, H2d C12 EPU -0.4 / -6.0, H2a C4 EPU +18.6 / +23.8; mean |error| 7.1 % for both. R5 is much better on the octet, worse on C12 and C4; the largest errors of both models are the C4 cells, consistent with the static C4 miss (our C4 is too soft, so the stroke is too long).
  Peak force R5: F2d +42 % (the C12 peak), F2h +15 %, F2e +18 %.

**Diagnostics on why the C4 cell is missed (offline, no new Abaqus runs):**
- Refit of the rule on all 22 specimens (set 4, 6 and 7 parameters; `results/rule_fit4/`): in-sample rms 7.0 % / 7.4 %, max 17 % (B10a): the rule family cannot represent C4 and C12 / octet / BCC at once (the rate of change with the zone length scale differs: zl = 1.2 puts B10a at -0.5 % but B9d at +40 %). Neither the zone floor `lmin` nor the profile taper (`taper` 1 / 2) nor a plate-node scale helps (plate nodes change B10a by < 7 points).
- **The slicing model with a weaker participation decay is better than the 6-parameter rule on most cells** (`results/slice_decay.md`, no fitted constant except lambda): at lambda = infinity it is within +-3.5 % on B5a/c/d, B9d, B10b, B11b, B11d, B8, B7 but over-predicts the BCC cell B6a by +18.5 % and still misses C4 r 0.8 (-13.5 %) and the graded C4 (-10 %);
  rms over the 22 specimens 11.6 % (lambda = r, the recorded model), 9.2 % (2 r), 8.1 % (4 r), 7.7 % (8 r and none). The recorded lambda = r was chosen before any evaluation and never tuned; the static data prefer no decay (the impact data of 2026-10-05 had preferred lambda = r).
  Neither the slicing model nor the rule is below ~7.7 % rms / 17-18 % worst case on the 22 static specimens: the remaining errors are structural: C4 r 0.8 under, BCC cell over, for both model families.
- **Not yet checked:** whether the C4 r 0.8 solid itself is mesh-converged (h/r = 0.5; the stiffening of 1.98 may contain locking in the short necks); the short-strut Timoshenko limit (L/r = 3.3); the overlap of the two joints of the 2.63 mm strut at r 0.8.

## Run list 2026-10-19: mesh-convergence study of the C4 r 0.8 solid (B10e, B10f)

Why: B10a (h 0.4, 45.4k C3D10M) gives a stiffening of 1.975 x bare beams, 16-23 % above every model (notes/18 section 14). The short C4 strut is 2.63 mm = 3.3 r with nu = 0.49: before any
joint-model change we must know whether that solid value is mesh-converged. Same case and loading as B10a (r 0.8, 0.5 mm in 0.3 ms, hold 0.5 ms, linear E0), only the tet size changes; plate-end audit passed.

| case (abaqus/inp) | tet size h | elements | est. wall time |
|---|---|---|---|
| B10f_C4_r08_h055_linE0 (coarse) | 0.55 | 21.4k | ~6 min |
| B10a_C4_r08_linE0 (done) | 0.40 | 45.4k | 14 min (done) |
| B10e_C4_r08_h028_linE0 (fine) | 0.28 | 110.7k | ~50-60 min |

Run B10f first, then B10e. Copy back the history CSV(s) and `.sta` / `.dat` / `.msg` as before. Then `PYTHONPATH=src:tests python abaqus/bx_predict.py B10a B10e B10f` (the beam-model predictions are mesh independent) and compare the hold-window means.

**Decision rule** (hold mean of the sum reaction, B10a = 21.38 N, stiffening 1.975): the three meshes give a convergence trend (refinement ratio ~1.4); extrapolate with the observed order.
- fine within 3 % of B10a and the trend flattening: the solid is converged, the C4 under-prediction is real and needs a structural joint-model change (short strut / merged joints);
- fine more than ~8 % below B10a (stiffening below ~1.82): part of the 1.975 is locking of the short necks; use the extrapolated value, re-score B10a-d, F2a and H2a (same mesh size) and weight them down accordingly;
- in between: the extrapolated value is the reference and the remaining model error is the one reported.

### Results 2026-10-19: mesh-convergence study of the C4 r 0.8 solid (B10f / B10a / B10e) and what it means

| case | h [mm] | elements | hold mean [N] | stiffening vs bare (10.82 N) | JOINT_FHS error | rule R5 error |
|---|---|---|---|---|---|---|
| B10f | 0.55 | 21.4k | 21.010 | 1.942 | -18.5 % | -22.0 % |
| B10a | 0.40 | 45.4k | 21.383 | 1.976 | -19.9 % | -23.3 % |
| B10e | 0.28 | 110.7k | 21.576 | 1.994 | -20.6 % | -24.0 % |
| Richardson (observed order 1.93) | 0 | - | 21.77 | **2.012** | -21.4 % | -24.7 % |

- **The solid is mesh-converged to < 1 % at h 0.28 and its stiffness INCREASES with refinement** (the opposite of locking; coarse meshes are slightly soft). The C4 under-prediction is therefore real physics, not a solid artefact:
  the models miss the converged C4 r 0.8 stiffness by 21-25 %. (The h = r/2 meshes used for all other solids are ~ 1-2 % soft by the same trend; this does not change any conclusion.)
- **Diagnosis so far (offline):** on the C4 cell the slicing model and the rule R5 both leave the MIDDLE of the short 2.63 mm strut (L/r = 3.3 at r 0.8, 4.4 at r 0.6) unstiffened (element factors 1.01 / 1.0, ends 1.4-1.8), although the two valence-3 joints at its ends must merge along it.
  The long-strut end elements have mean factors 5.0-5.4 (slicing) and 3.1-3.3 (rule); neither reaches the solid. One-at-a-time changes of the rule (zone scale 1.2: B10a -0.5 % but C12 B9d +40 %; short-strut zone growth: B10a -13.7 % at 0.5 but C12 +12 %, B10b unchanged; zone floor, taper 1 / 2, plate-node scale, gain, saturation)
  never satisfy C4 and C12 together. Both have struts of ~ 2.6 mm but r 0.8 vs 0.5; the data need a mechanism that switches on between r / L = 0.19 (C12, fits) and 0.30 (C4 r 0.8; r 0.6 at 0.23 is -10 %).
- **Proposed next specimens (B12, "short-strut series", not yet built):** a symmetric Y-joint pair: two valence-3 nodes (all angles 120 deg) joined by a strut of length L_s, with four 7.3 mm legs up to the plates, r 0.8, L_s = 2.63 / 4.5 / 7.0 mm (L_s / r = 3.3 / 5.6 / 8.8), plus r 0.6 at L_s = 2.63 (small meshes, minutes each).
  This isolates the merged-joint regime at fixed node topology and gives the missing L / r dependence directly; the C4 cell and C12 are then explained (or not) by the same function.

## Run list 2026-10-19: B12 (short-strut Y-joint series) and what the offline tests say about it (notes/18 section 17)

Built as proposed: two valence-3 Y nodes (all angles 120 deg) at mid height joined by a strut of length Ls, two legs per node (30 deg to the vertical) ending on the plates, H 12.7, linear E0,
top plate down 0.5 mm in 0.3 ms, hold 0.5 ms, current FHS geometry, solid h = r / 2 (10-16k C3D10M each, a few minutes of Abaqus each), plate-end check passed.

| case (abaqus/inp) | r | Ls [mm] | Ls / r | elements |
|---|---|---|---|---|
| B12a_Y_r080_Ls263_linE0 | 0.8 | 2.63 | 3.3 | 10.0k |
| B12b_Y_r080_Ls450_linE0 | 0.8 | 4.5 | 5.6 | 10.3k |
| B12c_Y_r080_Ls700_linE0 | 0.8 | 7.0 | 8.8 | 10.0k |
| B12d_Y_r060_Ls263_linE0 | 0.6 | 2.63 | 4.4 | 12.5k |
| B12e_Y_r050_Ls263_linE0 | 0.5 | 2.63 | 5.3 | 15.5k |

Blind predictions (`results/B12_summary.md`; hold value / bare beams): every model gives only 1.03-1.08 (B12a: bare 32.1 N, JOINT_FHS 1.042-1.047, rule R5 1.075-1.080; B12c 1.04-1.07; B12e 1.03-1.05).
**B12 is a weak probe:** a RIGID short strut raises the predicted reaction by only 5-9 % (B12a 34.7 -> 37.0 N, B12e 12.9 -> 14.0 N) because the inclined legs dominate the load path; so a solid more than ~10 % above 1.08 would be news,
anything below tells little. Run only if cheap (all five are a few minutes); it does not replace B13.

**Where the C4 stiffness sits (offline sensitivity of the B10a beam model, rule R5; solid target 21.77 N vs R5 16.39 N = +33 %):** short struts x3 -> 21.08 N (+28.6 %); short struts rigid -> 31.43 N (+92 %); end elements of the long struts x2 / x4 -> +10 / +16 %;
middle elements of the long struts x1.3 -> +15 %; elements within 0.9 mm of the plates x2 -> +8 %. **The C4 cell is dominated by its short struts, and the solid behaves as if they were ~3 x stiffer than the models make them (their middle element stays at factor 1.0).**

**Offline tests that did NOT reconcile C4 (needs +25 %) with C12 (needs 0) / octet / BCC, all within the rule family:** zone scale; zone floor; short-strut zone growth (L / r < 5); linear / exponential / Gaussian tails of height 0.3-2 and length 1-2.5 r from each joint (additive and multiplicative combination of the two ends:
C4 -> +-3 % at tail 0.3 / 1.5 r, but C12 B9d +37 %, B9c +69 %, octet B9b +19 %); plate-node scale; gain / saturation; the angle weight sin^p and ((1 + cos)/2)^p (refit R7 on 22 specimens: rms 7.6 %, max 18.8 %, the fit sets p = 0).
The slicing model (real FHS joint shape) without participation decay matches C12 (+2 %) and C4 r 0.6 (+2 %) but not C4 r 0.8 (-13.5 %). C4 r 0.6 -> 0.8 changes the stiffening from 1.46 to 1.98, which suggests a threshold: the two joint fillets (~ 1.7 r each) merge along the 2.63 mm strut at r ~ 0.77.

**Next specimens (B13, proposed, not built): the C4 cell with DECOUPLED radii of its short (2.63 mm) and long (6.35 mm) struts**, (r_long, r_short) = (0.6, 0.8), (0.6, 1.0), (0.8, 0.6), (0.8, 1.0) plus the existing B10b (0.6, 0.6) and B10a (0.8, 0.8): the short struts are in the main load path
there, so the solid reaction versus r_short and r_long gives the effective short-strut stiffness (and the radius-ratio dependence that the design gradient needs) directly. 45-100k elements each, 15-35 min of Abaqus each.

## Run list 2026-10-19: B13 (C4 with decoupled short / long strut radii), the last open experiment of the joint-model work

Why (notes/18 sections 16-18): the converged C4 r 0.8 solid is 21-25 % stiffer than every model; the C4 reaction is dominated by its 12 short (2.63 mm) struts (short struts x3 = +29 %); the solids make them ~2.1 x (r 0.6) and ~3.2 x (r 0.8)
stiffer than rule R5 (`results/short_strut_calibration.md`). B13 varies the short and the long struts' radii separately (B10b (0.6, 0.6) and B10a (0.8, 0.8) are the equal-radius members), 0.5 mm in 0.3 ms, hold 0.4 ms, linear E0,
current FHS geometry. All four inputs are audited (every plate end has a full face; 12 short and 24 long struts with the right radii).

| order | case (abaqus/inp) | r long | r short | L_short / r_short | elements | est. wall time |
|---|---|---|---|---|---|---|
| 1 | B13d_C4_rl080_rs100_linE0 | 0.8 | 1.0 | 2.6 | 46.4k | ~15-20 min |
| 2 | B13a_C4_rl060_rs080_linE0 | 0.6 | 0.8 | 3.3 | 67.7k | ~30 min |
| 3 | B13b_C4_rl060_rs100_linE0 | 0.6 | 1.0 | 2.6 | 81.4k | ~35 min |
| 4 | B13c_C4_rl080_rs060_linE0 | 0.8 | 0.6 | 4.4 | 88.2k | ~40 min |
| optional | B12a-e (short-strut Y pairs) | | | | 10-16k | ~5 min each (weak probe, see above) |

Then `PYTHONPATH=src:tests python abaqus/b13_predict.py` and `python abaqus/short_strut_calibration.py` (the multiplier m on the short struts that reproduces each solid, on top of R5).

**Blind predictions** (hold value / bare beams; `results/B13_summary.md`):

| case | JOINT_FHS (no decay) | rule R5 | R5 + short-strut law (provisional stopgap) | solid |
|---|---|---|---|---|
| B10b (0.6, 0.6) | 1.492 | 1.306 | 1.325 | 1.462 |
| B13a (0.6, 0.8) | 1.555 | 1.363 | **1.716** | |
| B13b (0.6, 1.0) | 1.964 | 1.844 | **2.318** | |
| B10a (0.8, 0.8) | 1.708 | 1.514 | 1.942 | 1.975 |
| B13c (0.8, 0.6) | 2.044 | 1.814 | **1.914** | |
| B13d (0.8, 1.0) | 1.794 | 1.524 | **2.066** | |

**Decision rules:** (1) the stopgap (gain 1.5, onset L / r = 4.6, calibrated on the four C4 solids B10a-d) is validated for the C4 family if it is within +-8 % on all four B13 solids; (2) if the multiplier m of B13a (r_short 0.8, r_long 0.6) equals that of B10a and m of B13c equals that of B10b, m depends on the
short strut's own radius only (law in L / r_short; refit gain and onset on the six points); (3) if m also depends on r_long (the merged node blob is set by the neighbours), the law must use the node's local radii and the design gradient changes accordingly.

## Joint model: state at wrap-up (2026-10-19)

**What is configured** (`src/jointRule.py`, `jointRule.PRESETS`; use `joint=jointRule.PRESETS[name]` in `drop_rig` / the prescribed-motion pipeline; exact design gradients, tested adjoint = AD 1e-9, = finite differences 1e-6):

| preset | what | when |
|---|---|---|
| `R5` | closed-form rigid-like end zones (0.77 x the geometric zone length, neighbour-weighted, valence sum saturating at 2.5), fitted on 13 linear specimens | long-strut lattices: simple cubic, BCC, octet, C12, T-joints, unequal radii |
| `R5_C4` (provisional) | R5 + short-strut law (gain 1.5, onset L / r 4.6) | the C4 design lattice, until B13 confirms or replaces it |
| `R5_all22` | R5 refitted on all 22 specimens | only if one compromise model is wanted (rms 7.0 %, worst 17 %) |

**Accuracy on the 22 prescribed-motion specimens (hold-window mean; `results/joint_model_summary.md`)**, rms / worst case [%]: bare beams 32 / 70; zone k = 3.5 11.3 / 24; JOINT_FHS slicing (lambda = r) 11.6 / 30; slicing without decay 7.7 / 18.5 (not differentiable);
**R5 7.8 / 23 (3.0 / 5.7 without the C4 cells; held out on 13 specimens 5.0 / 9.7); R5 + short-strut law 3.6 / 9.4**; R5 refit 7.0 / 17.
Unequal radii (B11, the design variables): R5 within +-6 % (JOINT_FHS 6-9 % low); the quadratic radius ratio is confirmed. C4 solid mesh-converged (stiffening 1.942 / 1.976 / 1.994, extrapolated 2.012).
Impact (current-geometry solids F2 / H2), stroke error R5 / R5_C4: C4 F2a +14.0 / -3.3 %, H2a +23.8 / +0.2 %; C12 -4.6 / -4.7, -6.0 / -6.0; octet (F2g, F2h) -0.1, +1.3 (R5) vs JOINT_FHS +6.6, +15.2; BCC +5.1 .. +7.6 (R5).

**Known limits (stated up front):** (a) the short-strut law is a four-point calibration whose onset sits exactly between the C4 (L / r <= 4.4) and C12 (L / r >= 4.6) struts, and it worsens the C4 impact PEAK force (+49 % / +25 %); (b) the EPU 45 stiffening
of the fast prescribed ramps is 6-17 % lower than the linear one and no constant transfers to impact (mu = 1 is used); (c) section shape (VAE, non-circular) is not represented: the rule uses the radius only; (d) the explicit step cost with these stiffer ends is 1.5-2.9 x bare beams;
(e) all fits are gradient-free and the 6-7 fitted numbers are not tightly identified (predictions are stable, parameters are not); (f) the fits use linear-elastic prescribed motion; impact validation exists for C4, C12, BCC and octet only.

**Lessons that changed the work (kept in the memory and CLAUDE.md):** compare linear solids by the hold-window mean (they ring); audit that plate faces reach the plates (B7); the first solids of the F / H series were built with an old FHS geometry (11-13 % too soft in BCC);
mesh-converge a solid before blaming a model (C4); impact stroke is a weak test of joint stiffness (JOINT_FHS looks right in impact yet is 21-30 % too soft on the static B9b / B9c).

**What is left for the joint configuration:** B13 (above); then either adopt the law (rule 1) or replace it (rules 2-3); re-run the C4 impact solids F2a / H2a with the final model and look at the peak force; use the preset in the plateau-design optimisation (notes/9, k = 3.5 before) and verify one optimised design with an FHS solid.

## Differentiability audit and the final presets (2026-10-19; B13 results not yet received)

**Status of B13:** the four inputs are built and the blind predictions recorded (sections above); the Abaqus results are NOT in the repo yet, so the short-strut law is still the provisional four-point stopgap (`PRESETS['R5_C4']`).
The calibration script (`short_strut_calibration.py`) and `b13_predict.py` are ready to run on the B13 CSVs; the decision rules are written down.

**Differentiability, for use in gradient-based design (`src/jointRule.py`, tests `tests/test_joint_rule_smooth.py`, `tests/test_joint_rule_adjoint.py`):**
- The rule is a torch expression of the element radii and of every parameter: autograd gives dk/dr (and the gradient w.r.t. the rule parameters, for fitting through the solver). Checked against central finite differences.
- **Kinks removed.** The hard rule has kinks where AD and finite differences disagree: the clamp of the pair zone length, `max(W - 2, 0)`, the zone cap at L / 2, the zone edge crossing an element boundary (piecewise-linear overlap), and the short-strut onset.
  `smooth = 0.05` (a width in units of the own radius) replaces each by a smooth version. A first version blended the zone edge in the stiffness domain and shifted the effective edge outward by ~ w ln k (+27 % on the short C4 struts); the final version blends the
  COMPLIANCE, 1/beta = 1 - (1 - 1/k0) phi0 - (1 - 1/k1) phi1, which reproduces the hard rule to 0.2 % (mean factor ratio 1.000, max 0.001-0.002 at r 0.5 / 0.7 / 0.9).
  The short-strut onset is smoothed over 0.03 in L / r only (a wider one leaked into C12's L / r = 4.64 struts and raised the C12 error from 3.2 % to 12.1 %).
- **Results:** autograd = finite differences at random per-element radii (180 elements, 12 random draws, relative error < 1e-5); the factors are continuous along a 1500-step radius scan (no jumps > 5 % per 0.4 % radius step); the parameter gradients (a, zl, b, p, wsat, short_gain) are exact;
  the checkpointed adjoint through the whole explicit drop equals unrolled AD (1e-9) and finite differences (1e-6) for the shipped smooth `R5_C4` preset (`tests/test_joint_rule_adjoint.py`).
- **Cost per design evaluation:** 0.02 s (forward) and 0.14 s (backward) for the 3 x 3 C4 lattice (276 struts, 1332 elements), negligible against the 10 min value + gradient of the drop.
- **Accuracy unchanged by smoothing** (`results/preset_check.md`, 22 specimens, rms / worst): `R5` 7.8 / 23.5 %; **`R5_C4` 3.6 / 9.7 %** (C4 7.0, graded C4 3.6, C12 3.6, others as R5); `R5_all22` 7.1 / 18.0 %.

**How to use it in the optimiser:** `rig = drop_rig(lat, radii(x, ...), law, mat=mat, joint=jointRule.PRESETS['R5_C4'])`; the gradient w.r.t. the design `x` then includes how the radii move the joint stiffening. Use `PRESETS['R5']` for lattices without short (L / r < 4.6) struts.
The presets are the smooth form; do not use `smooth = 0` in an optimisation (kinks).

### Results 2026-10-19: B13 (C4 with decoupled short / long strut radii; all four runs completed; solid hold-window means, ringing 7-11 %, halves agree < 1 %)

| case | r long | r short | solid [N] | solid / bare | JOINT_FHS (lambda = r) | JOINT_FHS (no decay) | rule R5 | R5 refit (22) | **R5 + short-strut law (blind)** |
|---|---|---|---|---|---|---|---|---|---|
| B10b | 0.6 | 0.6 | 6.111 | 1.462 | -3.0 % | +2.0 % | -10.6 % | -6.2 % | -9.4 % (calibration set) |
| B13a | 0.6 | 0.8 | 8.639 | 1.682 | -12.2 % | -7.6 % | -19.0 % | -16.9 % | **+2.1 %** |
| B13b | 0.6 | 1.0 | 12.033 | 2.155 | -14.0 % | -8.9 % | -14.4 % | -12.4 % | **+7.5 %** |
| B10a | 0.8 | 0.8 | 21.383 | 1.975 | -19.9 % | -13.5 % | -23.3 % | -17.2 % | -1.7 % (calibration set) |
| B13c | 0.8 | 0.6 | 17.067 | 2.094 | -11.4 % | -2.4 % | -13.4 % | -9.2 % | **-8.6 %** |
| B13d | 0.8 | 1.0 | 27.708 | 2.101 | -20.6 % | -14.6 % | -27.5 % | -22.0 % | **-1.7 %** |

- **The stopgap passes its blind test:** the short-strut law (gain 1.5, onset L / r = 4.6) was calibrated on B10a-d only; on the four B13 solids it is within +7.5 / -8.6 % (rms 5.9 %), against 13-28 % low for R5 alone. Decision rule 1 (within +-8 %) is met on 3 of 4 and missed by 0.6 point on B13c.
  The only competitor that comes close is the geometry-based slicing model without participation decay (-2.4 ... -14.6 %), which is not differentiable.
- **Law refit on all eight C4 solids** (`results/short_law_fit.md`, grid over gain and onset, controls = C12 B9c / B9d, unequal-radius T-joints B11a-d, octet B9b): the optimum is flat; best (gain 1.75, onset 4.6) rms 5.9 % vs the shipped (1.5, 4.6) 6.0 %, so the preset is kept. The onset is at the edge of the allowed range (4.6 protects C12, whose shortest struts are L / r = 4.64); controls stay at 3.1-3.2 % rms
  for every gain up to 2.0. Leave-one-out over the eight C4 solids: **6.1 % rms, 10.5 % worst** (B13b +10.5, B13c -9.3, B10b -9.2).
- **What the multiplier m (short struts, on top of R5; `results/short_strut_calibration.md`) says about the physics:** it depends mainly on the short strut's own radius, not on L alone:
  r_short 0.6: m = 2.09 (r_long 0.6) and 2.04 (r_long 0.8), i.e. independent of the long struts; r_short 0.8: 2.73 (r_long 0.6) and 3.16 (r_long 0.8); r_short 1.0: 2.42 (r_long 0.6) and 4.40 (r_long 0.8). The coupling to the long struts' radius appears only for the thick short struts;
  m is a lumped correction on top of R5 (whose own end zones are too soft for thick struts), so this is a residual of the model form, not a separate law. It is the origin of the +-10 % that the law leaves.
- **Solid-only observation (no model needed):** raising r_short by 0.2 raises the C4 reaction by x1.40 / x1.39 at r_long 0.6 (0.6 -> 0.8 -> 1.0) and by x1.25 / x1.30 at r_long 0.8; raising r_long 0.6 -> 0.8 raises it by x2.8 / x2.5 / x2.3 for r_short 0.6 / 0.8 / 1.0.

**Conclusion on the C4 correction:** `PRESETS['R5_C4']` is validated for C4-type lattices (valence-3 nodes with 90 / 120 / 120 deg angles and 2.6 mm short struts) to about +-10 % in statics (blind 5.9 % rms), and leaves every other topology unchanged. It is NOT validated outside that family; its onset sits at the C12 boundary; the impact peak force of C4 gets worse (+49 % / +25 %); B12 (Y-pair series) remains an optional extra.
