import numpy as np

from shared.activations import activate, activate_deriv
from shared.initializers import initialize_layer
from shared.losses import mse


class SimplePerceptron:


    def __init__(
        self,
        n_inputs,
        activation="logistic",
        beta=1.0,
        initializer="random_normal",
        init_scale=0.1,
        seed=42,
        weight_decay=0.0,
    ):
        if n_inputs <= 0:
            raise ValueError(f"n_inputs must be positive, got {n_inputs}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        self.W, self.b = initialize_layer(
            n_inputs, n_out=1, method=initializer, scale=init_scale, seed=seed
        )
        self.activation = activation
        self.beta = beta
        self.weight_decay = weight_decay

    def _forward(self, X):
        """X: (N, n_features). Returns (O, h)."""
        h = X @ self.W + self.b
        O = activate(h, self.activation, self.beta)
        return O, h

    def pre_activation(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        O, _ = self._forward(X)
        return O

    def get_params(self):
        """Return parameters as a list, in the order the optimizer expects."""
        return [self.W, self.b]

    def set_params(self, params):
        self.W, self.b = params[0], params[1]

    def train_epoch(self, X, t, optimizer, batch_size=0, shuffle=True, rng=None):
        """Run one full training epoch.

        batch_size=0  → full-batch gradient descent
        batch_size=1  → online (per-sample)
        batch_size=N  → mini-batch of size N
        """
        N = X.shape[0]
        if batch_size == 0:
            batch_size = N

        indices = np.arange(N)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(indices)

        batch_losses = []

        for start in range(0, N, batch_size):
            idx = indices[start:start + batch_size]
            X_b, t_b = X[idx], t[idx]

            O, _ = self._forward(X_b)
            deriv = activate_deriv(O, self.activation, self.beta)

            # Gradient of 0.5*mean((t-O)^2) w.r.t. parameters
            delta = (t_b - O) * deriv                  # (batch,)
            grad_b = -float(np.mean(delta))             # scalar
            grad_W = -np.mean(delta[:, None] * X_b, axis=0)  # (n_in,)

            if self.weight_decay > 0:
                # Weight decay applies to W only — bias is left untouched.
                grad_W = grad_W + self.weight_decay * self.W

            self.W, self.b = optimizer.step([self.W, self.b], [grad_W, grad_b])

            batch_losses.append(mse(t_b, O))

        return float(np.mean(batch_losses)), batch_losses

    def save(self, path):
        np.savez(
            path,
            W=self.W,
            b=np.asarray([self.b], dtype=np.float64),
            activation=self.activation,
            beta=float(self.beta),
            weight_decay=float(self.weight_decay),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        W = np.asarray(data["W"], dtype=np.float64)
        model = cls(
            n_inputs=W.shape[0],
            activation=str(data["activation"]),
            beta=float(data["beta"]),
            weight_decay=float(data["weight_decay"]),
        )
        model.W = W.copy()
        model.b = float(np.asarray(data["b"], dtype=np.float64).reshape(-1)[0])
        return model
