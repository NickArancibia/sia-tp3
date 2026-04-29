import numpy as np


def activate(h, name, beta=1.0):
    if name == "identity":
        return h.copy()
    elif name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(beta * h, -500, 500)))
    elif name == "tanh":
        return np.tanh(beta * h)
    elif name == "step":
        return np.where(h > 0, 1.0, -1.0)
    elif name == "relu":
        return np.maximum(0.0, h)
    raise ValueError(f"Unknown activation: {name}")


def activate_deriv(O, name, beta=1.0):
    """Derivative of activation w.r.t. pre-activation h, expressed via output O.

    For 'step', returns a pseudo-derivative (identity) since the true
    derivative is 0 almost everywhere and undefined at 0.
    For 'relu', O is the post-activation output so relu'(h) = (O > 0).
    """
    if name == "identity":
        return np.ones_like(O)
    elif name == "logistic":
        return beta * O * (1.0 - O)
    elif name == "tanh":
        return beta * (1.0 - O ** 2)
    elif name == "step":
        return np.ones_like(O)
    elif name == "relu":
        return np.where(O > 0, 1.0, 0.0)
    raise ValueError(f"Unknown activation: {name}")
