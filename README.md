# Domain Adaptation

A Hydra-driven research repository for closed-set, open-set, universal, partial, and multi-source domain adaptation experiments.

The project is organized around a few simple ideas:
- datasets are described declaratively in `src/configs/dataset`
- methods are registered lazily from `src/methods`
- runtime and performance defaults live in one shared config
- experiment outputs are isolated under `results/`

## Highlights

- Multiple DA settings: `csda`, `osda`, `pda`, `unida`, `msda`
- Method registry with lazy imports
- Shared `BaseSolver` for runtime, evaluation, checkpointing, and device handling
- Hydra config composition for datasets, methods, and performance options
- Local-data friendly layout: datasets and LMDB caches are symlinked under `data/`
- Full training-state resume with optimizer, scheduler, AMP, RNG, and method state
- Multi-process distributed data parallel training for compatible solvers

## Repository Layout

```text
src/
  main.py                 # training entrypoint
  configs/
    config.yaml           # shared defaults
    dataset/              # dataset definitions and class splits
    method/               # method-specific hyperparameters
  datasets/
    storage.py            # files/LMDB datasets and LMDB environment ownership
    samplers.py           # multi-source batch samplers
    transforms.py         # reusable augmentation components
    loader.py             # dataloader construction and orchestration
  methods/
    registry.py           # solver registration and lookup
    base_solver.py        # shared solver contract
    components.py         # focused cross-method training components
    *.py                  # concrete methods
  models/
    backbones.py          # torchvision backbone registry
    heads.py              # shared heads/modules
  utils/
    config.py             # config parsing and OmegaConf resolver helpers
    distributed.py        # torch.distributed process and gradient synchronization
    runtime.py            # runtime setup, seeding, logging helpers
    validation.py         # startup configuration validation
    utils.py              # lightweight generic helpers

data/                     # local dataset symlinks and LMDB cache symlinks
results/                  # Hydra run outputs
scripts/                  # local helper scripts
```

Registered solvers are `cad`, `cosda`, `dare`, `dcfm`, `dcpr`, `dcpr_alt`, `factda`, `mic`, `ros`, `rtda`, `rvtc`, and `sourceonly`.

## Setup

Python `>=3.10` is required.

Install dependencies from [pyproject.toml](pyproject.toml). The most direct option in this repo is:

```bash
uv sync
```

If you do not use `uv`, create your environment as usual and install the dependencies listed in `pyproject.toml`.

## Data Layout

This repository does not track datasets in git.

Expected workflow:
- symlink each dataset under `data/`
- keep the symlink names aligned with `src/configs/dataset/*.yaml`
- optionally symlink `data/lmdb-cache` to a faster local disk

Examples already used by this repo:
- `data/domainnet -> /root/autodl-tmp/DomainNet`
- `data/image-clef -> /root/autodl-tmp/image_CLEF`
- `data/office-31 -> /root/autodl-tmp/Office-31`
- `data/office-home -> /root/autodl-tmp/OfficeHome`
- `data/pacs -> /root/autodl-tmp/PACS`
- `data/visda-2017 -> /root/autodl-tmp/Visda-2017`

See [data/README.md](data/README.md) for the local-data policy.

## Running Experiments

The entrypoint is [main.py](src/main.py).

Example:

```bash
python src/main.py dataset=office-31 method=mic exp_name=mic_a2w
```

MSDA example:

```bash
python src/main.py dataset=image-clef method=dcpr dataset.sources='[b,c,i]' dataset.target=p exp_name=dcpr_imageclef_bci_to_p
```

Useful override examples:

```bash
python src/main.py method=dcpr performance.compile.enabled=false
python src/main.py method=dcpr performance.augmentation.target_tensor_v2=auto
python src/main.py num_workers=8 performance.dataloader.num_workers_source=4
```

Save a full training-state checkpoint every epoch and resume it later:

```bash
python src/main.py method=mic exp_name=mic_resume resume.save_every_epochs=1
python src/main.py method=mic exp_name=mic_resume \
  resume.path=checkpoints/mic_resume.resume.pth
```

The resume checkpoint is separate from the lightweight best-model checkpoint.
It includes model, optimizer/scheduler, AMP scaler, epoch/global step, RNG state,
and registered method-specific state.

Compatible solvers can run with multiple processes:

```bash
torchrun --standalone --nproc-per-node=2 src/main.py \
  method=mic exp_name=mic_2gpu distributed.enabled=true
```

Distributed training currently supports `sourceonly`, `mic`, `rvtc`, `factda`,
`dcfm`, `ros`, and `cad`. Methods with global prototype or memory state
(`cosda`, `rtda`, `dare`, `dcpr`, `dcpr_alt`) fail fast instead of silently
changing their algorithm semantics.

Hydra writes each run under:

```text
results/<exp_name>/
```

Because Hydra changes into that run directory, the default best checkpoint path is:

```text
results/<exp_name>/checkpoints/<exp_name>.pth
```

## Batch Experiment Suites

For multi-task sweeps, use [run_experiment_suite.py](scripts/run_experiment_suite.py) with a JSON spec:

```bash
python scripts/run_experiment_suite.py --spec scripts/specs/<suite>.json --groups main
```

Useful launcher options:
- `--screen` starts a detached `screen` session for long runs.
- `--resume` skips completed experiment ids from an existing `summary.csv`.
- `--continue-on-error` continues later experiments in the same suite when one run fails instead of aborting.
- `--notify-feishu` sends the generated `summary.md` to the Feishu webhook stored as `FEISHU_WEBHOOK_URL` in the repository-root `.env`.
- `--notify-each-run` sends a Feishu success card after each completed experiment, using the same webhook.
- `--shutdown` powers off after all selected runs finish and notifications are attempted. Shutdown is skipped if the suite is interrupted (KeyboardInterrupt or manual-stop signals); only use it when that behavior is intended.

Feishu webhook requests bypass machine-wide proxy environment variables so a stopped local proxy does not block result delivery.

## Adding a New Method

1. Create a solver in `src/methods/<name>.py`
2. Decorate the solver class with `@register_solver("<name>")`
3. Add a config file in `src/configs/method/<name>.yaml`
4. Add the module path to `_SOLVER_MODULES` in [src/methods/__init__.py](src/methods/__init__.py)

The recommended path is to inherit from [base_solver.py](src/methods/base_solver.py) and reuse:
- device transfer helpers
- autocast/grad-scaler utilities
- evaluation/checkpoint helpers
- resumable optimizer/scheduler registration
- common runtime flags

Method-level structure, logging, and checkpoint conventions are documented in [src/methods/README.md](src/methods/README.md).

## Adding a New Dataset

1. Add a dataset config under `src/configs/dataset`
2. Point `root:` to the symlink name under `data/`
3. Define class splits for the settings you want to support

Most datasets can reuse the generic logic in [loader.py](src/datasets/loader.py) without extra code.

## Project Conventions

- Keep method-specific logic in `src/methods/<method>.py`
- Keep cross-method infrastructure in `src/models`, `src/datasets`, or `src/utils`
- Prefer shared config/runtime helpers over ad-hoc parsing in individual files
- Treat `data/`, `results/`, and `scripts/` as local working directories, not source code

## Current Defaults

The shared defaults in [config.yaml](src/configs/config.yaml) are intended to be safe, practical baselines rather than one-size-fits-all maxima.

Notable examples:
- `performance.compile.enabled=false`
- `performance.channels_last=false`
- `performance.faiss_threads=1` for FAISS-based methods
- `performance.augmentation.target_tensor_v2=auto`
- dataloader workers default to `4/4/2` for source/target/test
- `resume.save_every_epochs=0` keeps periodic resume checkpoints opt-in
- `distributed.enabled=auto` activates only under `torchrun`

These are easy to override per run through Hydra.

## Development Checks

Install the development dependency group and run the regression suite:

```bash
uv sync --group dev
uv run --group dev pytest
```

The tests cover Hydra composition, runtime validation, DataLoader options,
LMDB ownership, distributed samplers, augmentation components, full-state
resume, and experiment provenance output.
