import numpy as np


def initialize_weights(n_in, n_out=1, method="random_normal", scale=0.1, seed=42):
    """Return weight vector of size n_in + 1 (index 0 is bias)."""
    rng = np.random.default_rng(seed)
    if method == "random_normal":
        return rng.normal(0.0, scale, size=n_in + 1)
    elif method == "xavier":
        limit = np.sqrt(6.0 / (n_in + n_out))
        w = rng.uniform(-limit, limit, size=n_in)
        return np.concatenate([[0.0], w])
    elif method == "he":
        std = np.sqrt(2.0 / n_in)
        w = rng.normal(0.0, std, size=n_in)
        return np.concatenate([[0.0], w])
    raise ValueError(f"Unknown initializer: {method}")
