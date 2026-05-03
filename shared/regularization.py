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
        self.best_epoch = 0

    def __call__(self, val_loss, params, epoch=None):
        """Return True if training should stop."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            self.best_params = copy.deepcopy(params)
            self.best_epoch = 0 if epoch is None else int(epoch)
            return False
        self._counter += 1
        return self._counter >= self.patience

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self._best,
            "counter": self._counter,
            "best_params": copy.deepcopy(self.best_params),
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state):
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self._best = float(state.get("best", self._best))
        self._counter = int(state.get("counter", self._counter))
        best_params = state.get("best_params")
        self.best_params = None if best_params is None else copy.deepcopy(best_params)
        self.best_epoch = int(state.get("best_epoch", self.best_epoch))


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
