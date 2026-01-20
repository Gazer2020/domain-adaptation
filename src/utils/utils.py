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
