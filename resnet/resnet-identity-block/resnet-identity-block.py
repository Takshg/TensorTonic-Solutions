import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)

    # y=ReLU(ReLU(xW1^T)W2^T+x)
    input = x.copy()
    out = input @ W1.T  # xW_1^T
    out = np.maximum(0, out) @ W2.T  #ReLU(xW1^T)W2^T
    y = np.maximum(0, (out + x))
    
    return y