import copy

import numpy as np


class EarlyStopping:
    """Stop when the monitored loss stops improving.

    ``best_params`` stores a deep copy of the parameter list at the best epoch,
    so it works equally for a single ndarray (perceptron simple) or a list of
    matrices/vectors (MLP).
    """

    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self._best = np.inf
        self._counter = 0
        self.best_params = None

    def __call__(self, val_loss, params):
        """Return True if training should stop."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            self.best_params = copy.deepcopy(params)
            return False
        self._counter += 1
        return self._counter >= self.patience


def l2_penalty(weights, lam):
    """Compute L2 regularization term: (lam / 2) * sum of ||W^(m)||^2 for all layers.

    weights: list of weight matrices (NOT biases — biases are not regularized).
    """
    if lam == 0:
        return 0.0
    return 0.5 * lam * sum(float(np.sum(W ** 2)) for W in weights)


def l2_gradient(weights, lam):
    """Compute L2 regularization gradient: lam * W for each weight matrix.

    Returns list of gradient matrices (same shapes as weights).
    Biases are not regularized, so they are excluded from this list.
    """
    return [lam * W for W in weights]