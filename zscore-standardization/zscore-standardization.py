import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """

    x = np.array(X, dtype=float)
    mu = np.mean(x, keepdims=True, axis=axis)     
    std = np.std(x, keepdims=True, axis=axis)
    std = np.where(std > eps, std, 1.0)
    z = ((x-mu)/std)
    return z