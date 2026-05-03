import numpy as np


def _to_list(x):
    """Wrap a single ndarray/scalar as a one-element list, leave lists as-is."""
    if isinstance(x, list):
        return x
    return [x]


def _zeros_like_each(params_list):
    return [np.zeros_like(np.asarray(p, dtype=float)) for p in params_list]


class GradientDescent:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        params = _to_list(params)
        grads = _to_list(grads)
        return [p - self.lr * g for p, g in zip(params, grads)]

    def reset(self):
        pass

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state):
        self.lr = state.get("lr", self.lr)


class Momentum:
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self._velocity = None

    def step(self, params, grads):
        params = _to_list(params)
        grads = _to_list(grads)
        if self._velocity is None:
            self._velocity = _zeros_like_each(params)
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self._velocity[i] = self.momentum * self._velocity[i] - self.lr * g
            new_params.append(p + self._velocity[i])
        return new_params

    def reset(self):
        self._velocity = None

    def state_dict(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "velocity": None if self._velocity is None else [v.copy() for v in self._velocity],
        }

    def load_state_dict(self, state):
        self.lr = state.get("lr", self.lr)
        self.momentum = state.get("momentum", self.momentum)
        velocity = state.get("velocity")
        self._velocity = None if velocity is None else [np.asarray(v, dtype=float).copy() for v in velocity]


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params, grads):
        params = _to_list(params)
        grads = _to_list(grads)
        if self._m is None:
            self._m = _zeros_like_each(params)
            self._v = _zeros_like_each(params)
        self._t += 1
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g ** 2
            m_hat = self._m[i] / (1.0 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1.0 - self.beta2 ** self._t)
            new_params.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return new_params

    def reset(self):
        self._m = None
        self._v = None
        self._t = 0

    def state_dict(self):
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "m": None if self._m is None else [m.copy() for m in self._m],
            "v": None if self._v is None else [v.copy() for v in self._v],
            "t": self._t,
        }

    def load_state_dict(self, state):
        self.lr = state.get("lr", self.lr)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.eps = state.get("eps", self.eps)
        m_state = state.get("m")
        v_state = state.get("v")
        self._m = None if m_state is None else [np.asarray(m, dtype=float).copy() for m in m_state]
        self._v = None if v_state is None else [np.asarray(v, dtype=float).copy() for v in v_state]
        self._t = int(state.get("t", self._t))


def build_optimizer(cfg):
    name = cfg.get("optimizer", "gd").lower()
    lr = cfg.get("learning_rate", 0.01)
    if name in ("gd", "sgd"):
        return GradientDescent(lr)
    elif name == "momentum":
        return Momentum(lr, cfg.get("momentum", 0.9))
    elif name == "adam":
        betas = cfg.get("adam_betas", [0.9, 0.999])
        return Adam(lr, betas[0], betas[1], cfg.get("adam_eps", 1e-8))
    raise ValueError(f"Unknown optimizer: {name}")
