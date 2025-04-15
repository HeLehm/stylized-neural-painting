import torch


def auto_device():
    """
    Automatically selects the device for PyTorch tensors.
    Returns 'cuda' if a GPU is available,
    Returns 'mps' if a Mac with Apple silicon is available,
    otherwise returns 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
