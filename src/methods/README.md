# Methods Conventions

This directory holds concrete solver implementations. The goal is to keep every solver file readable in the same way, even when the algorithm itself is very different.

## Recommended File Skeleton

Use this order when adding or refactoring a solver:
- module docstring with paper/method summary
- imports
- module-level helpers that are local to the method only
- `@register_solver(...)` class definition
- `build_model()`
- optimizer/scheduler helpers
- train/eval mode helpers
- forward/evaluation helpers
- algorithm-specific training utilities
- `train()`
- checkpoint overrides last

## Shared BaseSolver Facilities

Prefer reusing `BaseSolver` helpers instead of re-implementing infrastructure in each method.

Useful helpers:
- `_to_device()`, `_auto_cast()`, `_optimizer_step_with_optional_clip()`
- `_log_epoch_summary()` for consistent epoch-end logging
- `_log_best_checkpoint_loaded()` and `_log_training_complete()` for train-end summaries
- `_save_named_modules_checkpoint()` and `_load_named_modules_checkpoint()` for multi-component checkpoints
- `_maybe_save_best()` and `_load_best_checkpoint_if_available()` for best-model handling

## Logging Style

Keep epoch logs compact and field-oriented:
- prefix with the solver or stage name
- emit scalar metrics as `key=value`
- keep evaluation metric naming explicit: `Acc` for closed-set accuracy, `Score` when the method returns H-score or another composite metric
- prefer one summary line per epoch rather than multiple fragmented lines

Examples:
- `RGR 3/20 | src=1.2043 rnode=0.0812 total=1.2368 | rmp=0.50 crmp=1.00 | Acc=72.40% (best=73.10%)`
- `DCFM Warmup 2/5 | task=0.9123 dom=0.3881 total=1.3004 | Acc=64.20% (best=64.20%)`

## Checkpoint Conventions

Checkpoint files should:
- include a top-level `method` field
- use stable keys for each learnable component, for example `model`, `student`, `ema`, `feature_extractor`
- keep method-specific tensors or thresholds as plain top-level values
- load older single-model checkpoints when practical to preserve backward compatibility

Preferred pattern:

```python
self._save_named_modules_checkpoint(
    path,
    modules={"student": self.net, "ema": self.ema_net},
    extra_state={"threshold": self.threshold},
)
```

## Boundary Rules

Keep method files focused on method logic:
- method-specific augmentation, losses, and pseudo-label logic stay here
- generic config parsing belongs in `src/utils/config.py`
- runtime setup and logging summaries belong in `src/utils/runtime.py`
- reusable neural modules belong in `src/models/`
- dataset/build-loader behavior belongs in `src/datasets/`

When a helper starts being reused by multiple solvers, move it out of the method file.
