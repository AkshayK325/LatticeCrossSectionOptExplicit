# Abaqus/Explicit benchmarks

Checks of the explicit beam solver against Abaqus 2024/Explicit. The cases and the pass criteria are in
`BENCHMARK_PLAN.md`, which tracks done and left.

| Path | Content |
|---|---|
| `cases.py` | One definition per case: writes the `.inp` and runs our solver on the same model |
| `inp/` | Generated Abaqus input files (run these). Only `.inp` is tracked here |
| `logs/` | Useful Abaqus job logs per run: `.sta` (energy table, increments, timing), `.dat` (input-processing warnings/errors, model size), `.msg` when it has warnings. `.env`, `.com`, `.prt`, … are not kept |
| `odb/` | Put the `.odb` files here (git-ignored, large) |
| `extract_odb.py` | Abaqus Python script: odb → CSV |
| `csv/` | Extracted Abaqus histories `<job>_history*.csv` (tracked). The `<job>_field_U*.csv` frame dumps (7 GB, nothing reads them) are git-ignored and were removed from the tree 2026-10-16; they stay in git history and can be re-extracted from the odb |
| `ours/` | Our histories for the same case `<name>.csv` (tracked) |
| `compare.py` | Overlay, metrics, figure (`ResultsFig/abaqus/<n>_<name>.png`) and `results/<name>.md` |
| `results/` | One report per case |

## Round trip for one case
1. Write the input (here): `PYTHONPATH=src python abaqus/cases.py A1 A2 ...` → `abaqus/inp/<name>.inp`.
2. Run Abaqus (your machine): `abaqus job=A1_cantilever input=A1_cantilever.inp double=both interactive`.
   Use double precision: the comparison is at the 1% level.
3. Extract (on the Abaqus machine, Abaqus Python): `abaqus python extract_odb.py A1_cantilever.odb`.
   This writes `A1_cantilever_history.csv` (and the unused `A1_cantilever_field_U.csv`, do not copy it).
4. Copy the history CSV into `abaqus/csv/`, the `.sta` and `.dat` (and `.msg` if it has warnings) into `abaqus/logs/`,
   optionally the odb into `abaqus/odb/` (git-ignored).
5. Compare (here): `PYTHONPATH=src python abaqus/compare.py A1 A2 ...` → figure + `abaqus/results/A1_cantilever.md`.
   Then update the status in `BENCHMARK_PLAN.md`.

**Why step 3.** The odb format can only be read with Abaqus's own Python (`odbAccess`), and this machine has no
Abaqus. If only the odb is available, step 3 has to be run where Abaqus is installed.
