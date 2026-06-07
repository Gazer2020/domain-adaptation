from __future__ import annotations

from datasets.loader import _build_loader_kwargs


def test_loader_kwargs_disable_worker_only_options_for_single_process():
    kwargs = _build_loader_kwargs(
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=object(),
    )

    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_loader_kwargs_keep_prefetch_settings_with_workers():
    worker_init = object()
    kwargs = _build_loader_kwargs(
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=5,
        worker_init_fn=worker_init,
    )

    assert kwargs == {
        "num_workers": 3,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 5,
        "worker_init_fn": worker_init,
    }
