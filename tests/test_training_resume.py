from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from methods.base_solver import BaseSolver


class _ResumeSolver(BaseSolver):
    def build_model(self):
        self.net = torch.nn.Linear(3, 2).to(self.device)

    def train(self):
        pass


def _config(resume_path=None):
    return OmegaConf.create(
        {
            "device": "cpu",
            "exp_name": "resume-test",
            "dataset": {"num_classes": 2},
            "method": {
                "name": "resume-test",
                "setting": "csda",
                "save_start_epoch": 1,
            },
            "performance": {
                "non_blocking_transfer": True,
                "zero_grad_set_to_none": True,
                "channels_last": False,
                "amp": {"enabled": False, "dtype": "bf16"},
                "compile": {
                    "enabled": False,
                    "backend": "inductor",
                    "mode": "default",
                    "dynamic": "auto",
                    "fullgraph": False,
                },
            },
            "resume": {
                "path": resume_path,
                "save_path": "auto",
                "save_every_epochs": 0,
            },
        }
    )


def _loaders():
    dataset = TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=2)
    return loader, loader, loader


def test_training_checkpoint_restores_full_runtime_state(tmp_path):
    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)
    solver = _ResumeSolver(_config(), _loaders())
    optimizer = torch.optim.SGD(
        solver.net.parameters(), lr=0.2, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    solver.register_training_state(optimizer=optimizer, scheduler=scheduler)

    inputs, labels = next(iter(solver.source_loader))
    loss = torch.nn.functional.cross_entropy(solver.net(inputs), labels)
    solver._zero_grad(optimizer)
    solver._optimizer_step_with_optional_clip(loss, optimizer)
    scheduler.step()
    solver._best_metric = 73.5

    checkpoint_path = tmp_path / "training.resume.pth"
    solver.save_training_checkpoint(checkpoint_path, epoch=2)
    expected_torch_random = torch.rand(3)
    expected_numpy_random = np.random.rand(3)
    expected_python_random = [random.random() for _ in range(3)]
    expected_model = {
        name: value.detach().clone()
        for name, value in solver.net.state_dict().items()
    }

    resumed = _ResumeSolver(
        _config(str(checkpoint_path)),
        _loaders(),
    )
    resumed_optimizer = torch.optim.SGD(
        resumed.net.parameters(), lr=0.2, momentum=0.9
    )
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=1
    )
    resumed.register_training_state(
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )

    assert resumed._resume_epoch == 2
    assert resumed._training_global_step == 1
    assert resumed._best_metric == 73.5
    assert list(resumed._epoch_range(5)) == [2, 3, 4]
    for name, value in resumed.net.state_dict().items():
        assert torch.equal(value, expected_model[name])
    assert resumed_optimizer.state_dict()["state"]
    assert resumed_scheduler.state_dict()["last_epoch"] == 1
    assert torch.equal(torch.rand(3), expected_torch_random)
    assert np.array_equal(np.random.rand(3), expected_numpy_random)
    assert [random.random() for _ in range(3)] == expected_python_random
