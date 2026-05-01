import numpy as np


def initialize_layer(n_in, n_out=1, method="random_normal", scale=0.1, seed=42):
    """Return (W, b) for a single layer.

    Layout:
      n_out == 1:  W shape (n_in,)        b is a Python float (scalar)
      n_out > 1 :  W shape (n_out, n_in)  b shape (n_out,)

    The random draws are arranged so that, for n_out=1, the resulting (W, b)
    is identical (modulo reshaping) to the legacy single-vector initializer
    which produced ``rng.normal(0, scale, size=n_in+1)`` with index 0 = bias.
    """
    rng = np.random.default_rng(seed)

    if method == "random_normal":
        if n_out == 1:
            W = rng.normal(0.0, scale, size=n_in)
            return W, 0.0
        W = rng.normal(0.0, scale, size=(n_out, n_in))
        b = np.zeros(n_out)
        return W, b

    elif method == "xavier":
        limit = np.sqrt(6.0 / (n_in + n_out))
        if n_out == 1:
            W = rng.uniform(-limit, limit, size=n_in)
            return W, 0.0
        W = rng.uniform(-limit, limit, size=(n_out, n_in))
        b = np.zeros(n_out)
        return W, b

    elif method == "he":
        std = np.sqrt(2.0 / n_in)
        if n_out == 1:
            W = rng.normal(0.0, std, size=n_in)
            return W, 0.0
        W = rng.normal(0.0, std, size=(n_out, n_in))
        b = np.zeros(n_out)
        return W, b

    raise ValueError(f"Unknown initializer: {method}")


def initialize_layers(architecture, method="random_normal", scale=0.1, seed=42):
    """Initialize weights and biases for all layers of an MLP.

    architecture: [n_in, h1, h2, ..., n_out]
    Returns: list of (W, b) tuples, one per layer.
    Each layer gets a different seed derived from the base seed.
    """
    n_layers = len(architecture) - 1
    seed_seq = np.random.SeedSequence(seed)
    layer_seeds = seed_seq.spawn(n_layers)

    params = []
    for i in range(n_layers):
        W, b = initialize_layer(
            architecture[i], n_out=architecture[i + 1],
            method=method, scale=scale, seed=layer_seeds[i],
        )
        params.append((W, b))
    return params