import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """

    param = np.array(param, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)

    # Step 1: Update first moment 
    m_t = (beta1 * m) + ((1-beta1) * grad)

    # Step 2: Update second moment
    v_t = (beta2 * v) + ((1-beta2) * (grad**2))

    # Step 3: Bias Correction
    beta1_t = beta1**t # First moment decay
    beta2_t = beta2**t # Second moment decay
    
    m_hat = m_t / (1-beta1_t)
    v_hat = v_t / (1-beta2_t)

    # Step 4: Parameter Update
    param_new = param - lr * (m_hat / (np.sqrt(v_hat) + eps))
    
    return (param_new, m_t, v_t)