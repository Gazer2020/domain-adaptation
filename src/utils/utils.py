import torch


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class GpuLossAccumulator:
    """Accumulate per-batch scalar losses/metrics on GPU, sync only at epoch end.

    Replaces the pattern ``AverageMeter().update(loss.item())`` which forces a
    CPU-GPU synchronisation on every training step.  Instead, keep running sums
    as GPU tensors and call ``.item()`` once per key when ``compute()`` is
    called.

    Usage::

        acc = GpuLossAccumulator(device=self.device)
        for batch in loader:
            ...
            acc.update("task", loss_task)      # GPU scalar tensor
            acc.update("total", loss)
            acc.update("beta", beta_value)      # or plain float
            acc.step()
        metrics = acc.compute()  # -> {"task": 1.23, "total": 4.56, "beta": 0.01}
    """

    def __init__(self, *, device=None):
        self._sums: dict = {}
        self._device = device
        self._steps = 0

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            v = value.detach().float()
            if self._device is not None:
                v = v.to(self._device)
            return v
        t = torch.tensor(float(value), dtype=torch.float32)
        if self._device is not None:
            t = t.to(self._device)
        return t

    def update(self, key, value):
        v = self._to_tensor(value)
        if key not in self._sums:
            self._sums[key] = v
        else:
            self._sums[key] += v

    def step(self):
        self._steps += 1

    def compute(self):
        scale = 1.0 / max(1, self._steps)
        return {k: (v * scale).item() for k, v in self._sums.items()}


def cycle(iterable):
    """
    Infinitely cycle through an iterable.
    
    Useful for iterating over a smaller dataset indefinitely
    to match the length of a larger dataset.
    
    Args:
        iterable: An iterable to cycle through
        
    Yields:
        Items from the iterable, repeating indefinitely
    """
    while True:
        for x in iterable:
            yield x


def get_device(device_str: str = "auto") -> str:
    """
    Get the appropriate device string.
    
    Args:
        device_str: Device specification. 'auto' will detect available device.
        
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    import torch
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str
