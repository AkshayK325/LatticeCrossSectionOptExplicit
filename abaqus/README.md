# Abaqus/Explicit benchmarks

Checks of the explicit beam solver against Abaqus 2024/Explicit. The cases and the pass criteria are in
`BENCHMARK_PLAN.md`, which tracks done and left.

| Path | Content |
|---|---|
| `cases.py` | One definition per case: writes the `.inp` and runs our solver on the same model |
| `inp/` | Generated Abaqus input files (run these) |
| `odb/` | Put the `.odb` files here (git-ignored, large) |
| `extract_odb.py` | Abaqus Python script: odb → CSV |
| `csv/` | Extracted Abaqus histories `<job>_history.csv`, `<job>_field_U.csv` (tracked) |
| `ours/` | Our histories for the same case `<name>.csv` (tracked) |
| `compare.py` | Overlay, metrics, figure (`ResultsFig/abaqus/<n>_<name>.png`) and `results/<name>.md` |
| `results/` | One report per case |

## Round trip for one case
1. Write the input (here): `PYTHONPATH=src python abaqus/cases.py A1 A2 ...` → `abaqus/inp/<name>.inp`.
2. Run Abaqus (your machine): `abaqus job=A1_cantilever input=A1_cantilever.inp double=both interactive`.
   Use double precision: the comparison is at the 1% level.
3. Extract (on the Abaqus machine, Abaqus Python): `abaqus python extract_odb.py A1_cantilever.odb`.
   This writes `A1_cantilever_history.csv` and `A1_cantilever_field_U.csv`.
4. Copy the two CSVs into `abaqus/csv/`, plus optionally the odb into `abaqus/odb/`, and the `.sta` / `.msg` if
   Abaqus warned about anything.
5. Compare (here): `PYTHONPATH=src python abaqus/compare.py A1 A2 ...` → figure + `abaqus/results/A1_cantilever.md`.
   Then update the status in `BENCHMARK_PLAN.md`.

**Why step 3.** The odb format can only be read with Abaqus's own Python (`odbAccess`), and this machine has no
Abaqus. If only the odb is available, step 3 has to be run where Abaqus is installed.
