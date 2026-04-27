import numpy as np


def mse(y_true, y_pred):
    return 0.5 * float(np.mean((y_true - y_pred) ** 2))
