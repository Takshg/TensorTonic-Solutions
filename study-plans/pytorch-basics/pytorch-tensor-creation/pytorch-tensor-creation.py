import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    if method == "zeros": 
        t = torch.zeros(shape)
    elif method == "ones": 
        t = torch.ones(shape)
    else:
        t = torch.full(size = shape, fill_value = value)
    t = t.to(torch.float32)
    return t