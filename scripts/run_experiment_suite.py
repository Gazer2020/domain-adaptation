#!/usr/bin/env python3
"""
Generic experiment-suite runner.

Spec format (JSON):
{
  "suite_prefix": "prc_oh_cpr2a_mainline",
  "summary_lines": [
    "- dataset: Office-Home",
    "- task: cpr2a (sources=[Clipart, Product, Real World] -> target=Art)"
  ],
  "common": [
    "src/main.py",
    "seed=42",
    "device=cuda",
    "batch_size=64",
    "num_workers=8",
    "dataset=office-home",
    "dataset.target=Art",
    "dataset.sources=[\"Clipart\",\"Product\",\"Real World\"]",
    "method=prc",
    "method.epochs=20"
  ],
  "groups": {
    "main": [
      {
        "id": "m1",
        "name": "baseline",
        "purpose": "base run",
        "overrides": []
      }
    ]
  }
}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACC_RE = re.compile(r"Acc=([0-9]+(?:\.[0-9]+)?)% \(best=([0-9]+(?:\.[0-9]+)?)%\)")
MANUAL_STOP_SIGNALS = {
    signal.SIGINT,
    signal.SIGTERM,
    signal.SIGHUP,
    signal.SIGQUIT,
}


class RunFailedError(RuntimeError):
    def __init__(self, exp_name: str, returncode: int):
        super().__init__(f"Run failed: {exp_name} rc={returncode}")
        self.exp_name = exp_name
        self.returncode = returncode

    @property
    def stop_signal(self) -> signal.Signals | None:
        if self.returncode >= 0:
            return None
        try:
            return signal.Signals(-self.returncode)
        except ValueError:
            return None

    @property
    def is_manual_stop(self) -> bool:
        stop_signal = self.stop_signal
        return stop_signal in MANUAL_STOP_SIGNALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a JSON-defined experiment suite.")
    parser.add_argument("--spec", required=True, help="Path to suite JSON spec.")
    parser.add_argument("--groups", default="all", help="Comma-separated groups or 'all'.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for src/main.py.")
    parser.add_argument("--screen", action="store_true", help="Launch this suite in a detached screen session.")
    parser.add_argument("--session", default="", help="Optional screen session name.")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiment ids from existing summary.csv.")
    parser.add_argument(
        "--notify-feishu",
        action="store_true",
        help="Send a Feishu webhook notification after the suite finishes or fails. Reads URL from .env.",
    )
    parser.add_argument(
        "--notify-title",
        default="",
        help="Optional notification title. Defaults to the suite prefix.",
    )
    parser.add_argument(
        "--notify-max-chars",
        type=int,
        default=3500,
        help="Maximum Markdown characters to include in the Feishu card.",
    )
    parser.add_argument("--shutdown", action="store_true", help="Power off after all requested runs finish.")
    parser.add_argument(
        "--shutdown-cmd",
        default="shutdown now",
        help="Shell command used when --shutdown is set. Run through bash -lc.",
    )
    return parser.parse_args()


def load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if "suite_prefix" not in spec:
        raise ValueError("Spec must define 'suite_prefix'.")
    if "common" not in spec or not isinstance(spec["common"], list):
        raise ValueError("Spec must define list field 'common'.")
    if "groups" not in spec or not isinstance(spec["groups"], dict) or not spec["groups"]:
        raise ValueError("Spec must define non-empty dict field 'groups'.")
    return spec


def select_groups(spec: dict, groups_arg: str) -> list[str]:
    value = str(groups_arg).strip().lower()
    if value in {"", "all"}:
        return list(spec["groups"].keys())
    groups = [item.strip() for item in groups_arg.split(",") if item.strip()]
    unknown = [group for group in groups if group not in spec["groups"]]
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}; available={list(spec['groups'])}")
    return groups


def read_rows(summary_csv: Path) -> list[dict]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(suite_name: str, out_dir: Path, rows: list[dict], summary_lines: list[str]) -> None:
    fields = ["suite", "group", "id", "name", "purpose", "exp_name", "best_acc", "last_acc", "minutes"]
    summary_csv = out_dir / "summary.csv"
    summary_md = out_dir / "summary.md"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with summary_md.open("w", encoding="utf-8") as f:
        f.write(f"# {suite_name}\n\n")
        for line in summary_lines:
            f.write(f"{line}\n")
        if summary_lines:
            f.write("\n")
        f.write("| id | name | best_acc | last_acc | minutes |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['id']} | {row['name']} | {row['best_acc']} | "
                f"{row['last_acc']} | {row['minutes']} |\n"
            )


def run_one(
    suite_name: str,
    group_name: str,
    exp: dict,
    common: list[str],
    python_executable: str,
    suite_log: Path,
) -> dict:
    exp_name = f"{suite_name}/{exp['id']}_{exp['name']}"
    cmd = [python_executable, *common, *list(exp.get("overrides", [])), f"exp_name={exp_name}"]
    last_acc = None
    best_acc = None
    start = time.time()

    with suite_log.open("a", encoding="utf-8") as log:
        header = (
            "\n" + "=" * 96 + "\n"
            f"RUN {suite_name} | {exp['id']} | {exp['name']}\n"
            f"Purpose: {exp.get('purpose', '')}\n"
            f"Command: {shlex.join(cmd)}\n"
            + "=" * 96 + "\n"
        )
        print(header, flush=True)
        log.write(header)
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            match = ACC_RE.search(line)
            if match:
                last_acc = float(match.group(1))
                best_acc = float(match.group(2))

        rc = proc.wait()
        if rc != 0:
            raise RunFailedError(exp_name, rc)

    return {
        "suite": suite_name,
        "group": group_name,
        "id": exp["id"],
        "name": exp["name"],
        "purpose": exp.get("purpose", ""),
        "exp_name": exp_name,
        "best_acc": "" if best_acc is None else str(best_acc),
        "last_acc": "" if last_acc is None else str(last_acc),
        "minutes": str(round((time.time() - start) / 60.0, 2)),
    }


def launch_in_screen(args: argparse.Namespace, spec: dict) -> None:
    if shutil.which("screen") is None:
        raise RuntimeError("screen is not installed or not on PATH.")

    suite_prefix = spec["suite_prefix"]
    session = args.session or f"{suite_prefix}_screen"
    launcher_dir = ROOT / "results" / f"{suite_prefix}_launcher"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = launcher_dir / f"{session}.log"

    existing = subprocess.run(
        ["screen", "-list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if f".{session}" in existing.stdout:
        raise RuntimeError(f"Screen session already exists: {session}")

    rerun = [
        args.python,
        "scripts/run_experiment_suite.py",
        "--spec",
        str(Path(args.spec).resolve()),
        "--groups",
        args.groups,
    ]
    if args.resume:
        rerun.append("--resume")
    if args.notify_feishu:
        rerun.append("--notify-feishu")
    if args.notify_title:
        rerun.extend(["--notify-title", args.notify_title])
    if args.notify_max_chars != 3500:
        rerun.extend(["--notify-max-chars", str(args.notify_max_chars)])
    if args.shutdown:
        rerun.extend(["--shutdown", "--shutdown-cmd", args.shutdown_cmd])

    shell_cmd = f"cd {shlex.quote(str(ROOT))} && {shlex.join(rerun)} 2>&1 | tee {shlex.quote(str(launcher_log))}"
    subprocess.run(["screen", "-dmS", session, "bash", "-lc", shell_cmd], check=True, cwd=ROOT)

    print(f"Started screen session: {session}")
    print(f"Launcher log: {launcher_log}")
    print(f"Attach: screen -r {session}")


def find_summary_files(suite_prefix: str, groups: list[str]) -> list[Path]:
    paths = []
    for group in groups:
        summary_md = ROOT / "results" / f"{suite_prefix}_{group}" / "summary.md"
        if summary_md.exists():
            paths.append(summary_md)
    return paths


def load_dotenv(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def load_feishu_webhook() -> str:
    env_path = ROOT / ".env"
    webhook = load_dotenv(env_path).get("FEISHU_WEBHOOK_URL", "")
    if not webhook:
        raise ValueError(f"Missing FEISHU_WEBHOOK_URL in {env_path}.")
    return webhook


def build_notification_markdown(
    suite_prefix: str,
    groups: list[str],
    status: str,
    started_at: float,
    error: str | None,
    max_chars: int,
) -> str:
    elapsed = round((time.time() - started_at) / 60.0, 2)
    summary_files = find_summary_files(suite_prefix, groups)
    lines = [
        f"**Status:** {status}",
        f"**Suite:** `{suite_prefix}`",
        f"**Groups:** `{','.join(groups)}`",
        f"**Elapsed:** {elapsed} min",
    ]
    if error:
        lines.append(f"**Error:** `{error}`")
    if summary_files:
        lines.append("")
        lines.append("**Summary files:**")
        lines.extend(f"- `{path}`" for path in summary_files)
        lines.append("")
        lines.append("**Result summary:**")
        for path in summary_files:
            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                content = f"Failed to read {path}: {exc}"
            lines.append("")
            lines.append(content)
    else:
        lines.append("")
        lines.append("No `summary.md` file was written.")

    text = "\n".join(lines)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n...(truncated; see summary files on the machine)"
    return text


def send_feishu_notification(
    webhook: str,
    title: str,
    markdown: str,
    status: str,
    suite_log: Path | None = None,
) -> None:
    header_template = "green" if status == "success" else "red"
    body = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": header_template,
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": markdown,
                }
            ],
        },
    }
    try:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        # Feishu webhook calls must ignore machine-wide proxy env vars; this
        # server often has HTTP(S)_PROXY pointed at a local proxy that may be off.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=20) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise RuntimeError(f"Feishu webhook HTTP {response.status}: {response_body}")
            print("Feishu notification sent.", flush=True)
            if suite_log is not None:
                with suite_log.open("a", encoding="utf-8") as log:
                    log.write("\nFeishu notification sent.\n")
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        print(f"Feishu notification failed: {exc}", flush=True)
        if suite_log is not None:
            with suite_log.open("a", encoding="utf-8") as log:
                log.write(f"\nFeishu notification failed: {exc}\n")


def maybe_notify_feishu(
    args: argparse.Namespace,
    suite_prefix: str,
    groups: list[str],
    status: str,
    started_at: float,
    error: str | None,
) -> None:
    if not args.notify_feishu:
        return
    try:
        webhook = load_feishu_webhook()
    except ValueError as exc:
        print(f"Skip Feishu notification: {exc}", flush=True)
        return
    title = args.notify_title or f"{suite_prefix}: {status}"
    markdown = build_notification_markdown(
        suite_prefix=suite_prefix,
        groups=groups,
        status=status,
        started_at=started_at,
        error=error,
        max_chars=args.notify_max_chars,
    )
    summary_files = find_summary_files(suite_prefix, groups)
    suite_log = summary_files[0].parent / "suite.log" if summary_files else None
    send_feishu_notification(
        webhook=webhook,
        title=title,
        markdown=markdown,
        status=status,
        suite_log=suite_log,
    )


def run_suite_body(args: argparse.Namespace, spec: dict, groups: list[str]) -> None:
    suite_prefix = spec["suite_prefix"]
    summary_lines = list(spec.get("summary_lines", []))
    common = list(spec["common"])

    print(f"Start suite prefix={suite_prefix} groups={groups}", flush=True)
    for group in groups:
        suite_name = f"{suite_prefix}_{group}"
        out_dir = ROOT / "results" / suite_name
        out_dir.mkdir(parents=True, exist_ok=True)
        suite_log = out_dir / "suite.log"
        rows = read_rows(out_dir / "summary.csv") if args.resume else []
        completed = {row["id"] for row in rows}

        for exp in spec["groups"][group]:
            if args.resume and exp["id"] in completed:
                print(f"Skip completed: {suite_name} {exp['id']} {exp['name']}", flush=True)
                continue
            row = run_one(
                suite_name=suite_name,
                group_name=group,
                exp=exp,
                common=common,
                python_executable=args.python,
                suite_log=suite_log,
            )
            rows.append(row)
            write_summary(suite_name, out_dir, rows, summary_lines)


def run_suite(args: argparse.Namespace, spec: dict) -> None:
    groups = select_groups(spec, args.groups)
    suite_prefix = spec["suite_prefix"]
    started_at = time.time()
    status = "success"
    error = None
    caught_exc = None
    should_shutdown = args.shutdown
    try:
        run_suite_body(args, spec, groups)
    except KeyboardInterrupt as exc:
        status = "interrupted"
        error = "KeyboardInterrupt"
        caught_exc = exc
        should_shutdown = False
    except Exception as exc:
        caught_exc = exc
        if isinstance(exc, RunFailedError) and exc.is_manual_stop:
            status = "interrupted"
            stop_signal = exc.stop_signal
            signal_name = stop_signal.name if stop_signal is not None else f"signal {-exc.returncode}"
            error = f"{exc} ({signal_name}; treated as manual stop)"
            should_shutdown = False
        else:
            status = "failed"
            error = str(exc)
            should_shutdown = args.shutdown
    finally:
        maybe_notify_feishu(
            args=args,
            suite_prefix=suite_prefix,
            groups=groups,
            status=status,
            started_at=started_at,
            error=error,
        )
    if should_shutdown:
        print(f"All requested runs finished. Calling shutdown via shell: {args.shutdown_cmd}", flush=True)
        subprocess.run(["bash", "-lc", args.shutdown_cmd], check=False, cwd=ROOT)
    else:
        print(f"Shutdown skipped for suite status: {status}", flush=True)
    if caught_exc is not None:
        raise caught_exc


def main() -> None:
    args = parse_args()
    if args.notify_feishu:
        load_feishu_webhook()
    spec = load_spec(Path(args.spec))
    if args.screen:
        launch_in_screen(args, spec)
        return
    run_suite(args, spec)


if __name__ == "__main__":
    main()
