from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_experiment_suite.py"


def _load_suite_module():
    spec = importlib.util.spec_from_file_location("run_experiment_suite_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def test_collect_and_write_provenance(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "tests@example.com")
    _git(tmp_path, "config", "user.name", "Tests")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "initial")

    spec_path = tmp_path / "suite.json"
    spec_path.write_text('{"suite_prefix":"demo","common":[],"groups":{"main":[]}}\n')

    module = _load_suite_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    provenance = module.collect_provenance(spec_path, ["src/main.py", "seed=42"])

    assert provenance["git_commit"] == _git(tmp_path, "rev-parse", "HEAD")
    assert provenance["git_dirty"] is True
    assert len(provenance["spec_sha256"]) == 64
    assert provenance["collected_at_utc"].endswith("+00:00")
    assert provenance["uv_lock_sha256"] is None
    assert provenance["common_overrides"] == ["src/main.py", "seed=42"]

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    module.write_provenance(out_dir, provenance)
    saved = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert saved["spec_sha256"] == provenance["spec_sha256"]

    module.write_summary("demo", out_dir, [], ["- purpose: test"], provenance)
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Provenance" in summary
    assert provenance["git_commit"] in summary
