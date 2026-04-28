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
            arr = rng.normal(0.0, scale, size=n_in + 1)
            return arr[1:].copy(), float(arr[0])
        arr = rng.normal(0.0, scale, size=n_out * (n_in + 1))
        b = arr[:n_out].copy()
        W = arr[n_out:].reshape(n_out, n_in).copy()
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
