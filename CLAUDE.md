# Rules for Claude in this repo

## 1. Every background or long-running job reports to `run_status.log` (important, always)

`run_status.log` (git-ignored, repo root) is the live status board of everything that runs in the background. The user reads it, and so must you when asked to "check the run".

- **Scripts you write or edit:** call `runStatus` (in `tests/`, run with `PYTHONPATH=src:tests`) in `main()`:
  `import runStatus as rs; rs.start('what this run is')`, then `rs.update('current item', i, n)` per item (gives % done and time left).
  Finished / failed / killed is written automatically.
- **Commands that do not call runStatus** (third-party tools, one-liners, `pytest`, a `for` loop in the shell): start them through the wrapper
  `PYTHONPATH=src:tests python tests/run_logged.py --name "..." [--total N --count REGEX] -- <command>`.
- **A job that is already running:** attach a watcher
  `python tests/run_logged.py --watch PID --name "..." [--file LOG --count REGEX --total N]`.
- **Ladders of many small jobs** (fits, scans): one status line for the ladder as a whole (`abaqus/ladder_status.py`) plus each job's own entry.
- **When you launch something in the background:** within a minute confirm the entry is in `run_status.log` (read the file), and tell the user it is there.
  Never leave a job running that is not in the log. Output piped through `grep` is block-buffered and looks like a hang: use `python -u` and write the log
  file directly, and treat the result files / `run_status.log` as the source of truth for progress.
- **"Check the run" / "check results":** read `run_status.log` first, then the process list, then the result files.
- A job on another machine (Abaqus) cannot be seen from here: say so, and write its expected outputs and run order in `abaqus/BENCHMARK_PLAN.md`.

## 1b. Which runs are needed (read `notes/0_1_GlobalRunPolicy.md` before planning or launching any run)
Run only what can change a decision: six-question gate, verification ladder (analytic/own code -> cheap Abaqus -> one solid cell -> patch), tolerance budget fixed up front, no tuning on the validation case, one start per optimisation, mesh studies only on cheap shell cases. Register of planned runs and the "not run now" list are in that file; update it at every launch and result.

## 2. Other standing rules (details in the auto-memory; repeated here because they are costly to forget)
- Git: commit locally and often, never push, no Claude attribution (user is the sole author). `notes/` and `summary/` are git-ignored: never `git add -f` them.
- Every `notes/*.md` starts with the link to `notes/0_0_GlobalMasterPrompt.md`.
- Cap heavy jobs' memory (`RLIMIT_AS`) and mesh size.
- Before trusting an Abaqus solid: plate faces reach the plates (`cases._check_plate_ends`), compare linear solids by the hold-window mean (they ring).
- The joint-stiffness model must keep exact design gradients (adjoint vs finite differences) — see `notes/18_JointStiffness_MethodStudy.md`.
