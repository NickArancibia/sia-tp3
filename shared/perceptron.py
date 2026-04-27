import numpy as np

from shared.activations import activate, activate_deriv
from shared.initializers import initialize_weights
from shared.losses import mse


class SimplePerceptron:
    """Simple perceptron supporting identity (linear) and sigmoid/tanh (non-linear) activations.

    Weight vector layout: w[0] = bias, w[1:] = feature weights.
    """

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
        self.w = initialize_weights(n_inputs, n_out=1,
                                    method=initializer, scale=init_scale, seed=seed)
        self.activation = activation
        self.beta = beta
        self.weight_decay = weight_decay

    def _forward(self, X):
        """X: (N, n_features). Returns (O, h)."""
        h = X @ self.w[1:] + self.w[0]
        O = activate(h, self.activation, self.beta)
        return O, h

    def predict(self, X):
        O, _ = self._forward(X)
        return O

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
        if shuffle and rng is not None:
            rng.shuffle(indices)

        batch_losses = []

        for start in range(0, N, batch_size):
            idx = indices[start:start + batch_size]
            X_b, t_b = X[idx], t[idx]

            O, _ = self._forward(X_b)
            deriv = activate_deriv(O, self.activation, self.beta)

            # Gradient of 0.5*mean((t-O)^2) w.r.t. weights
            delta = (t_b - O) * deriv            # (batch,)
            grad_bias = -np.mean(delta)
            grad_w = -np.mean(delta[:, None] * X_b, axis=0)

            if self.weight_decay > 0:
                grad_w = grad_w + self.weight_decay * self.w[1:]

            grads = np.concatenate([[grad_bias], grad_w])
            self.w = optimizer.step(self.w, grads)

            batch_losses.append(mse(t_b, O))

        return float(np.mean(batch_losses)), batch_losses
