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
        """Return True if training should stop.

        ``params`` may be a single ndarray or a list of arrays/scalars.
        """
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            self.best_params = copy.deepcopy(params)
            return False
        self._counter += 1
        return self._counter >= self.patience
