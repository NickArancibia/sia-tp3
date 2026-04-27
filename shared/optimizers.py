import numpy as np


class GradientDescent:
    def __init__(self, lr):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads

    def reset(self):
        pass


class Momentum:
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self._velocity = None

    def step(self, params, grads):
        if self._velocity is None:
            self._velocity = np.zeros_like(params)
        self._velocity = self.momentum * self._velocity - self.lr * grads
        return params + self._velocity

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
        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)
        self._t += 1
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * grads
        self._v = self.beta2 * self._v + (1.0 - self.beta2) * grads ** 2
        m_hat = self._m / (1.0 - self.beta1 ** self._t)
        v_hat = self._v / (1.0 - self.beta2 ** self._t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

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
