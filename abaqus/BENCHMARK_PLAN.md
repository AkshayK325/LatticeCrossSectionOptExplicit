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
| B2i top 1.0, bottom 0.6, mid 0.8, 45 N mm | 🟦 | | | `inp/B2i_unit_cell_C4_graded_top100_bot060.inp` (71.6k el., min alt. 0.040 mm) | **fused+SV: 8.2312 mm @ 2.589 ms, peak 10.02 N**; fused 7.9766 mm @ 2.559 ms; k 3.5: 7.9422 mm @ 2.543 ms; bare 9.8966 mm @ 2.988 ms | — |
| B2j top 0.6, bottom 1.0, mid 0.8, 42 N mm | 🟦 | | | `inp/B2j_unit_cell_C4_graded_top060_bot100.inp` (72.8k el., 0.035 mm) | **fused+SV: 8.0857 mm @ 2.489 ms, peak 15.76 N**; fused 7.8407 mm @ 2.412 ms; k 3.5: 7.7812 mm @ 2.353 ms; bare 9.7447 mm @ 2.908 ms | — |
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
| F1a_unit_cell_C4_FHS_solid_r080_h040 | B2a | C4 r 0.8, 42.8k C3D10M (h 0.4, 289 boundary slivers removed), fixed mass scaling BELOW MIN to B2a's 3.37e-8 s (263 el, est. +0.06 % mass), 4 ms | 🟦 inp ready: **run first** |
| F1c / F1d / F1e-h / F1i-k | B2c / B2d / B2e-h / B2i-k | the same cells with FHS joints | ⬜ written after F1a has run |

Beam prediction for F1a: provisional, the fused-section joint model of the sphere joint (`joint_shape='fhs'` not
yet implemented, plan step 2).

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
