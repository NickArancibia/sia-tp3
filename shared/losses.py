import numpy as np


def mse(y_true, y_pred):
    diff = y_true - y_pred
    if diff.ndim == 1:
        return 0.5 * float(np.mean(diff ** 2))
    sum_axes = tuple(range(1, diff.ndim))
    return 0.5 * float(np.mean(np.sum(diff ** 2, axis=sum_axes)))


def mse_deriv(y_true, y_pred):
    return y_pred - y_true


def build_loss(name):
    """Devuelve (loss_fn, loss_deriv_fn) para el nombre dado.

    Hoy sólo soporta "mse". Para agregar una nueva loss:
      1. Definir `nombre_loss(y_true, y_pred) -> escalar`
      2. Definir `nombre_loss_deriv(y_true, y_pred) -> ndarray per-sample`
      3. Agregar una rama acá: `elif name == "nombre_loss": return ...`
    """
    if name == "mse":
        return mse, mse_deriv
    raise ValueError(f"Unknown loss: {name}")
