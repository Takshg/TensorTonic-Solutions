import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    k = len(predictions)
    q = predictions.copy()
    for i in range(len(predictions)): 
        if i == target: 
            q[i] = ((1-epsilon) + epsilon/k)
        else: 
            q[i] = epsilon/k
    
    q_ln_p = []
    
    for i in range(len(predictions)): 
        q_ln_p.append(q[i] * math.log(predictions[i]))
        
    loss = -1 * sum(q_ln_p)

    return loss 