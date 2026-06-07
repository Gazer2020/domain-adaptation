from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from methods import list_solvers
from utils import register_resolvers, validate_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "configs"


def _compose_method(method_name: str):
    register_resolvers(src_dir=CONFIG_DIR.parent)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=[f"method={method_name}"])


@pytest.mark.parametrize(
    "method_name",
    sorted(path.stem for path in (CONFIG_DIR / "method").glob("*.yaml")),
)
def test_all_method_configs_compose_and_validate(method_name: str):
    cfg = _compose_method(method_name)
    validate_config(cfg, available_solvers=list_solvers())


@pytest.mark.parametrize(
    ("override", "expected_path"),
    [
        ("batch_size=0", "batch_size"),
        ("performance.amp.dtype=fp8", "performance.amp.dtype"),
        ("performance.compile.dynamic=sometimes", "performance.compile.dynamic"),
        ("performance.dataloader.prefetch_factor=0", "prefetch_factor"),
        ("distributed.backend=mpi", "distributed.backend"),
        ("resume.save_every_epochs=-1", "resume.save_every_epochs"),
    ],
)
def test_invalid_shared_config_fails_fast(override: str, expected_path: str):
    cfg = _compose_method("ros")
    OmegaConf.update(cfg, *override.split("=", 1), merge=False)

    with pytest.raises(ValueError, match=expected_path):
        validate_config(cfg, available_solvers=list_solvers())
