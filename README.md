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

## Repository Layout

```text
src/
  main.py                 # training entrypoint
  configs/
    config.yaml           # shared defaults
    dataset/              # dataset definitions and class splits
    method/               # method-specific hyperparameters
  datasets/
    loader.py             # dataset construction, transforms, dataloaders
  methods/
    registry.py           # solver registration and lookup
    base_solver.py        # shared solver contract
    *.py                  # concrete methods
  models/
    backbones.py          # torchvision backbone registry
    heads.py              # shared heads/modules
  utils/
    config.py             # config parsing and OmegaConf resolver helpers
    runtime.py            # runtime setup, seeding, logging helpers
    utils.py              # lightweight generic helpers

data/                     # local dataset symlinks and LMDB cache symlinks
results/                  # Hydra run outputs
scripts/                  # local helper scripts
```

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
- `data/office-31 -> /root/autodl-tmp/Office-31`
- `data/office-home -> /root/autodl-tmp/OfficeHome`
- `data/image-clef -> /root/autodl-tmp/image_CLEF`

See [data/README.md](data/README.md) for the local-data policy.

## Running Experiments

The entrypoint is [main.py](src/main.py).

Example:

```bash
python src/main.py dataset=office-31 method=mic exp_name=mic_a2w
```

MSDA example:

```bash
python src/main.py dataset=image-clef method=prc dataset.sources='[b,c,i]' dataset.target=p exp_name=prc_imageclef_bci_to_p
```

Useful override examples:

```bash
python src/main.py method=prc performance.compile.enabled=false
python src/main.py method=prc performance.augmentation.target_tensor_v2=auto
python src/main.py num_workers=8 performance.dataloader.num_workers_source=4
```

Hydra writes each run under:

```text
results/<exp_name>/
```

## Adding a New Method

1. Create a solver in `src/methods/<name>.py`
2. Decorate the solver class with `@register_solver("<name>")`
3. Add a config file in `src/configs/method/<name>.yaml`
4. Add the module path to `_SOLVER_MODULES` in [src/methods/__init__.py](src/methods/__init__.py)

The recommended path is to inherit from [base_solver.py](src/methods/base_solver.py) and reuse:
- device transfer helpers
- autocast/grad-scaler utilities
- evaluation/checkpoint helpers
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
- `performance.augmentation.target_tensor_v2=auto`
- dataloader workers default to `4/4/2` for source/target/test

These are easy to override per run through Hydra.
