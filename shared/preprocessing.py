import numpy as np


class ZScoreScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def stratified_split(y, val_frac, test_frac, seed=42):
    """Return (train_idx, val_idx, test_idx) stratified by class y.

    A fraction of 0 yields an empty split (no minimum). A non-zero fraction
    ensures at least 1 sample per class in that split (avoids empty splits
    when the fraction is small but non-zero — relevant for rare classes).

    Use ``test_frac=0`` when the test set is provided externally (e.g. EJ2's
    ``digits_test.csv``) and you only want to split into train/val.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * test_frac)) if test_frac > 0 else 0
        n_val  = max(1, int(n * val_frac))  if val_frac  > 0 else 0
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test:n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def build_scaler(name):
    if name == "z-score":
        return ZScoreScaler()
    elif name == "min-max":
        return MinMaxScaler()
    elif name in (None, "none"):
        return None
    raise ValueError(f"Unknown scaler: {name}")
