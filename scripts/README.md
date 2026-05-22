# scripts

This directory stores local run scripts and helper utilities.

Policy:
- Keep this directory in git.
- By default, do not track local script files.
- Track this README and the generic suite launcher.
- Keep one-off local scripts, generated specs, and cache files ignored unless they become shared infrastructure.

Current launcher:
- `run_experiment_suite.py`: generic batch experiment runner with JSON spec input, `screen` support, resume support, summaries, suite-level and per-run Feishu webhook notifications, and optional auto shutdown.

Auto shutdown is only for runs where the user explicitly requested it; pass `--shutdown` only after that confirmation.
When both Feishu notification and shutdown are enabled, the launcher sends the notification before calling the shutdown command.
Shutdown is automatically skipped if the suite is interrupted (KeyboardInterrupt or SIGINT/SIGTERM/SIGHUP/SIGQUIT), so a manual stop does not power off the machine.

Feishu notification:
- Store `FEISHU_WEBHOOK_URL=...` in the repository-root `.env` file.
- Pass `--notify-feishu` to send a result card containing the `summary.md` path and Markdown summary text.
- Pass `--notify-each-run` to send a success card after each completed experiment, including group, run id, best/last accuracy, runtime minutes, and summary path.
- Webhook custom bots do not attach local files directly; the local `summary.md` path is included in the message.
- Feishu webhook requests bypass `HTTP_PROXY`/`HTTPS_PROXY` explicitly so a stopped local proxy cannot block notifications.
- To resend a completed suite without rerunning experiments, use `--resume --notify-feishu` with the same spec and groups.

Example:
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json`
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json --groups core,graph --screen`
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json --notify-feishu`
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json --notify-each-run`
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json --resume --notify-feishu`
- `python scripts/run_experiment_suite.py --spec /path/to/suite.json --resume --shutdown`
