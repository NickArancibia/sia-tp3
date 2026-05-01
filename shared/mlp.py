import copy

import numpy as np

from shared.activations import activate, activate_deriv
from shared.initializers import initialize_layers
from shared.losses import build_loss


class MLP:
    """Multi-Layer Perceptron with configurable architecture and activation.

    architecture: [n_in, h1, h2, ..., n_out]
        e.g. [784, 40, 20, 10]
    hidden_activation: activation for all hidden layers ("tanh", "logistic", "relu")
    output_activation: activation for the output layer ("logistic", "identity", "tanh")
    beta: sigmoid steepness parameter (used with tanh/logistic)
    initializer: weight initialization method ("random_normal")
    init_scale: std for random_normal initializer
    seed: base random seed (each layer uses an independent sub-stream)
    weight_decay: L2 regularization lambda (0 = no regularization)
    loss_name: loss function (sólo "mse" por ahora; agregar nuevas en shared/losses.py)
    """

    def __init__(
        self,
        architecture,
        hidden_activation="tanh",
        output_activation="logistic",
        beta=1.0,
        initializer="random_normal",
        init_scale=0.1,
        seed=42,
        weight_decay=0.0,
        loss_name="mse",
    ):
        architecture = list(architecture)
        if len(architecture) < 2:
            raise ValueError(
                f"architecture must have at least 2 elements (n_in, n_out); "
                f"got {architecture}"
            )
        if any(int(n) <= 0 for n in architecture):
            raise ValueError(
                f"all layer sizes must be positive ints; got {architecture}"
            )
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        self.architecture = architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.beta = float(beta)
        self.weight_decay = float(weight_decay)
        self.loss_name = loss_name
        self._loss_fn, self._loss_deriv_fn = build_loss(loss_name)
        self.n_layers = len(architecture) - 1

        self._activations = []
        for i in range(self.n_layers):
            if i < self.n_layers - 1:
                self._activations.append(hidden_activation)
            else:
                self._activations.append(output_activation)

        params = initialize_layers(
            architecture, method=initializer, scale=init_scale, seed=seed,
        )
        self.weights = []
        self.biases = []
        for W, b in params:
            W = np.atleast_2d(W.copy())
            if np.isscalar(b):
                b = np.array([b], dtype=np.float64)
            else:
                b = b.copy().astype(np.float64)
            self.weights.append(W)
            self.biases.append(b)

    def _forward(self, X):
        """Forward pass. Returns (output, cache) where cache stores
        pre-activations h and post-activations V for each layer."""
        V = X
        h_list = []
        V_list = [V]
        for i in range(self.n_layers):
            h = V @ self.weights[i].T + self.biases[i]
            h_list.append(h)
            V = activate(h, self._activations[i], self.beta)
            V_list.append(V)
        return V, (h_list, V_list)

    def predict(self, X):
        output, _ = self._forward(X)
        return output

    def _backward(self, t, cache):
        """Backpropagation. Returns list of (grad_W, grad_b) per layer.

        t: target output, shape (batch, n_out)
        cache: (h_list, V_list) from forward pass
        """
        h_list, V_list = cache
        batch_size = t.shape[0]

        grad_W_list = []
        grad_b_list = []

        deltas = [None] * self.n_layers

        output = V_list[-1]
        delta = (output - t) * activate_deriv(output, self._activations[-1], self.beta)
        deltas[-1] = delta

        for i in range(self.n_layers - 2, -1, -1):
            delta_next = deltas[i + 1]
            W_next = self.weights[i + 1]
            h_i = h_list[i]
            V_i = V_list[i + 1]
            delta_i = activate_deriv(V_i, self._activations[i], self.beta) * (delta_next @ W_next)
            deltas[i] = delta_i

        for i in range(self.n_layers):
            V_prev = V_list[i]
            delta = deltas[i]

            grad_W = delta.T @ V_prev / batch_size
            grad_b = np.mean(delta, axis=0)

            if self.weight_decay > 0:
                grad_W = grad_W + self.weight_decay * self.weights[i]

            grad_W_list.append(grad_W)
            grad_b_list.append(grad_b)

        return grad_W_list, grad_b_list

    def get_params(self):
        """Return all parameters as a flat list [W0, b0, W1, b1, ...]."""
        params = []
        for i in range(self.n_layers):
            params.append(self.weights[i])
            params.append(self.biases[i])
        return params

    def set_params(self, params):
        """Set all parameters from a flat list [W0, b0, W1, b1, ...]."""
        for i in range(self.n_layers):
            self.weights[i] = params[2 * i]
            self.biases[i] = params[2 * i + 1]

    def train_epoch(self, X, t, optimizer, batch_size=0, shuffle=True, rng=None):
        """Run one full training epoch.

        X: input data, shape (N, n_in)
        t: target data, shape (N, n_out)
        optimizer: optimizer instance with step(params, grads) method
        batch_size: 0 = full batch, 1 = online, N = mini-batch
        shuffle: whether to shuffle data before each epoch
        rng: numpy random generator for shuffling
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

            output, cache = self._forward(X_b)
            loss_val = self._loss_fn(t_b, output)

            grad_W_list, grad_b_list = self._backward(t_b, cache)

            params = self.get_params()
            grads = []
            for i in range(self.n_layers):
                grads.append(grad_W_list[i])
                grads.append(grad_b_list[i])

            new_params = optimizer.step(params, grads)
            self.set_params(new_params)

            batch_losses.append(loss_val)

        return float(np.mean(batch_losses)), batch_losses

    def save(self, path):
        """Save model weights and config to a .npz file."""
        save_dict = {
            "architecture": np.array(self.architecture),
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "beta": self.beta,
            "weight_decay": self.weight_decay,
            "loss_name": self.loss_name,
        }
        for i in range(self.n_layers):
            save_dict[f"W{i}"] = self.weights[i]
            save_dict[f"b{i}"] = self.biases[i]
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path):
        """Load a model from a .npz file."""
        data = np.load(path, allow_pickle=True)
        architecture = list(data["architecture"])
        model = cls(
            architecture=architecture,
            hidden_activation=str(data["hidden_activation"]),
            output_activation=str(data["output_activation"]),
            beta=float(data["beta"]),
            weight_decay=float(data["weight_decay"]),
            loss_name=str(data["loss_name"]),
        )
        for i in range(model.n_layers):
            model.weights[i] = data[f"W{i}"]
            model.biases[i] = data[f"b{i}"]
        return model
