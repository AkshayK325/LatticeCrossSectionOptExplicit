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
| **A2** single strut L = 6.4, r = 0.8, 8 B31, first-mode imperfection 0.05 mm (x); clamped base; top node guided (only z free, rotations fixed) with mass M = 10 g at 6 m/s (180 N mm), 2 ms | 🟦 | geometric nonlinearity (P–δ), buckling, axial wave speed at E₀ | peak root reaction ≤ 2%; RF3(t) correlation ≥ 0.98; buckling time (mid \|U1\| > 0.2 mm) ≤ 3%; max stroke ≤ 2% | `inp/A2_strut_impact.inp` | stroke 2.521 mm, mid U1 2.00 mm (buckled), root RF3 −201…+98 N, energy 180.01 ± 0.01 | — |
| **A3** crossed beams: A (r 0.8, along x, L 10, 10 B31) clamped at both ends; free B (r 0.5, along y) 0.3 mm above, moving down at 2 m/s; hits A at mid-span, rebounds; frictionless; 2 ms. **A3b** = same, Abaqus penalty × 10 (brackets Abaqus's own contact stiffness) | 🟦 | beam-radius contact (onset at r_A + r_B), contact impulse, rebound | impulse on B ≤ 5%; B rebound momentum ratio ≤ 5%; B / A mid-span U3 histories (shape; not graded); A3 vs A3b spread = Abaqus contact uncertainty | `inp/A3_crossed_beams.inp`, `inp/A3b_crossed_beams_penaltyx10.inp`. **Check** that contact engaged (B must not pass through A); if not, the beam edges need explicit inclusion | rebound momentum ratio 0.810, B mid U3 min −0.304 mm, energy incl. contact conserved 1e-4 | — |
| **A4** 3×3×1 C4 lattice drop, 1 mm elements (1332 B31), r = 0.8, 0.75 J, 6 m/s; top nodes in a `*RIGID BODY` (rotations fixed) whose reference node carries impactor + plate; bottom clamped; **no contact**; 4 ms | 🟦 | the whole assembly: BCs, rigid plate, impactor mass, many-element dynamics | max stroke ≤ 2%; plate U3 nRMSE ≤ 2%; bottom-reaction nRMSE ≤ 5%, peak ≤ 5% | `inp/A4_lattice_drop_nocontact.inp` | stroke 2.314 mm at 0.649 ms (stiff: linear E₀, so no densification and no contact needed), bottom reaction −965…+650 N, energy 750.00 → 750.08 | — |
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
