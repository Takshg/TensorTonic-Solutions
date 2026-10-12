import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # 1. Initialize the weights 
    w = np.zeros(X.shape[1]) # num of weights = num of features in design matrix 
    b = 0.0
    
    for i in range(steps): # Repeat until convergence 
        
        # 2. Calculate predictions 
        z = X @ w + b
        p = _sigmoid(z)
        
        # 3. Calculate loss (BCE)
        total_loss = np.sum(np.multiply(y, np.log(p)) + np.multiply((1-y), np.log(1-p)))
        avg_loss = np.multiply((-1/len(X)),total_loss)
        
        # 4. Compute the gradients Loss w.r.t weights and Loss w.r.t bias
        grad_w = X.T @ (p-y) / len(y)
        grad_b = np.mean(p-y)
    
        # 5. Update parameters 
        w = w - np.multiply(lr, grad_w)
        b = b - np.multiply(lr, grad_b)

        i += 1

    return (w,b)

        