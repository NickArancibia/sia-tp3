import numpy as np


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self._best = np.inf
        self._counter = 0
        self.best_weights = None

    def __call__(self, val_loss, weights):
        """Return True if training should stop."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            self.best_weights = weights.copy()
            return False
        self._counter += 1
        return self._counter >= self.patience
