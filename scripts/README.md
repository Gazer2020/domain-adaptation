# scripts

This directory stores local run scripts and helper utilities.

Policy:
- Keep this directory in git.
- By default, do not track script files.
- Track only this README.md unless policy changes.

Current launcher:
- `run_experiment_suite.py`: generic batch experiment runner with JSON spec input, `screen` support, resume support, summaries, and optional auto shutdown.

Example:
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json`
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json --groups core,graph --screen`
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json --resume --shutdown`
