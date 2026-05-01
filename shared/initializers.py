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

    if method != "random_normal":
        raise ValueError(f"Unknown initializer: {method}")

    if n_out == 1:
        arr = rng.normal(0.0, scale, size=n_in + 1)
        return arr[1:].copy(), float(arr[0])
    arr = rng.normal(0.0, scale, size=n_out * (n_in + 1))
    b = arr[:n_out].copy()
    W = arr[n_out:].reshape(n_out, n_in).copy()
    return W, b


def initialize_layers(architecture, method="random_normal", scale=0.1, seed=42):
    """Initialize weights and biases for all layers of an MLP.

    architecture: [n_in, h1, h2, ..., n_out]
    Returns: list of (W, b) tuples, one per layer.
    Each layer gets a different seed derived from the base seed.
    """
    params = []
    for i in range(len(architecture) - 1):
        layer_seed = seed + i
        W, b = initialize_layer(
            architecture[i], n_out=architecture[i + 1],
            method=method, scale=scale, seed=layer_seed,
        )
        params.append((W, b))
    return params
