import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    sampling_dist = []
    rng = np.random.default_rng(seed)
    
    # Draw bootstrap sample of size n w/ replacement 
    for i in range(n_bootstrap): 
        x_samples = rng.choice(x, size=len(x), replace=True)
        sampling_dist.append(np.mean(x_samples))
        i += 1

    # Compute statistic for bootstrap sample
    bootstrap_mean = np.mean(sampling_dist)

    # CI 
    lower_confidence_level = (1-ci)/2
    upper_confidence_level = ci + (1-ci)/2
    lower_CI = np.quantile(sampling_dist, lower_confidence_level )
    upper_CI = np.quantile(sampling_dist, upper_confidence_level)

    result = {"bootstrap_mean": bootstrap_mean, "lower": lower_CI, "upper": upper_CI}
    return result