# results

This directory stores Hydra experiment outputs, logs, checkpoints, summaries, and plots.

Hydra writes each run under `results/<exp_name>/` and changes the process cwd to that directory.
Solver checkpoints are therefore usually saved under `results/<exp_name>/checkpoints/`.

Policy:
- Keep this directory in git.
- Do not track generated result files in git.
