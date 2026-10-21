def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Initialize weights 
    x = x0
    
    for i in range(steps):
        # Gradient 
        grad = (2*a*x) + b

        # Update 
        x = x - (lr * grad)
    return x 