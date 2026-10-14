import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """

    K = num_classes if (num_classes != None) else (np.max(y) + 1)
    one_hot = np.zeros((len(y), K), dtype=float)
    one_hot[np.arange(np.size(y)), y] = 1.0
    return one_hot 
    