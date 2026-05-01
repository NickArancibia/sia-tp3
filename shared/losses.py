import numpy as np


def mse(y_true, y_pred):
    diff = y_true - y_pred
    if diff.ndim == 1:
        return 0.5 * float(np.mean(diff ** 2))
    sum_axes = tuple(range(1, diff.ndim))
    return 0.5 * float(np.mean(np.sum(diff ** 2, axis=sum_axes)))


def mse_deriv(y_true, y_pred):
    return y_pred - y_true


def cross_entropy(y_true, y_pred):
    eps = 1e-12
    return -float(np.mean(np.sum(y_true * np.log(y_pred + eps), axis=-1)))


def cross_entropy_deriv(y_true, y_pred):
    return y_pred - y_true


def build_loss(name):
    if name == "mse":
        return mse, mse_deriv
    elif name == "cross_entropy":
        return cross_entropy, cross_entropy_deriv
    raise ValueError(f"Unknown loss: {name}")