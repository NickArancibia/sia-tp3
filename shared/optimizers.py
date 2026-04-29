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
