"""Tests del MLP con la arquitectura decoupled.

Cobertura:
  A. Numerical gradient check (dL/dh y dL/dW) para MSE + cada activación
     element-wise. Esto valida que la regla de la cadena
     δ = (∂L/∂y) ⊙ f'(h) está bien implementada en el backward.
  B. Convergencia: validaciones del enunciado del TP3 (AND con escalón,
     y=x lineal, y=tanh(x) no lineal, XOR multicapa).
  C. Validaciones de configuración inválida.
  D. Save / load preserva config y predicciones.
  E. Regresión: scripts existentes (EJ1) siguen andando.

Uso:
    python3 tests/test_mlp_backward.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from shared.activations import activate, activate_deriv
from shared.losses import mse, mse_deriv
from shared.mlp import MLP
from shared.optimizers import Adam, GradientDescent
from shared.perceptron import SimplePerceptron


# =====================================================================
# Helpers
# =====================================================================


def _print_section(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def _ok(msg):
    print(f"  [OK] {msg}")


def _numeric_grad_dL_dW(model, X, t, layer_idx, eps=1e-6):
    """Finite differences para dL/dW de la capa `layer_idx`."""
    W = model.weights[layer_idx]
    grad = np.zeros_like(W)
    original = W.copy()
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = original[i, j] + eps
            L_plus = model._loss_fn(t, model.predict(X))
            W[i, j] = original[i, j] - eps
            L_minus = model._loss_fn(t, model.predict(X))
            W[i, j] = original[i, j]
            grad[i, j] = (L_plus - L_minus) / (2 * eps)
    return grad


def _numeric_grad_dL_db(model, X, t, layer_idx, eps=1e-6):
    """Finite differences para dL/db de la capa `layer_idx`."""
    b = model.biases[layer_idx]
    grad = np.zeros_like(b)
    original = b.copy()
    for i in range(b.shape[0]):
        b[i] = original[i] + eps
        L_plus = model._loss_fn(t, model.predict(X))
        b[i] = original[i] - eps
        L_minus = model._loss_fn(t, model.predict(X))
        b[i] = original[i]
        grad[i] = (L_plus - L_minus) / (2 * eps)
    return grad


# =====================================================================
# Test A — Gradient check decoupled chain rule
# =====================================================================


def test_A_gradient_check():
    _print_section("Test A — Gradient check para MSE + cada activación")
    rng = np.random.default_rng(0)

    # Probamos con todas las activaciones que pueden ir en la salida.
    # `step` queda fuera porque su derivada es pseudo (no matchea
    # finite differences de la salida real).
    output_activations = ["identity", "logistic", "tanh", "relu"]

    for out_act in output_activations:
        arch = [5, 8, 6, 3]
        model = MLP(
            architecture=arch,
            hidden_activation="tanh",
            output_activation=out_act,
            initializer="random_normal",
            seed=42,
            loss_name="mse",
        )
        X = rng.normal(0, 1, size=(7, 5))
        # Targets en rango compatible con la activación de salida
        if out_act == "logistic":
            t = rng.random(size=(7, 3))  # ∈ (0, 1)
        elif out_act == "tanh":
            t = rng.uniform(-1, 1, size=(7, 3))
        else:
            t = rng.normal(0, 1, size=(7, 3))

        _, cache = model._forward(X)
        grad_W_list, grad_b_list = model._backward(t, cache)

        # Chequeo dL/dW y dL/db por capa
        max_err_W = 0.0
        max_err_b = 0.0
        for layer_idx in range(model.n_layers):
            gW_numeric = _numeric_grad_dL_dW(model, X, t, layer_idx)
            err_W = np.max(np.abs(grad_W_list[layer_idx] - gW_numeric))
            max_err_W = max(max_err_W, err_W)

            gb_numeric = _numeric_grad_dL_db(model, X, t, layer_idx)
            err_b = np.max(np.abs(grad_b_list[layer_idx] - gb_numeric))
            max_err_b = max(max_err_b, err_b)

        assert max_err_W < 1e-4, (
            f"MSE+{out_act}: max abs error en dL/dW = {max_err_W:.2e}"
        )
        assert max_err_b < 1e-4, (
            f"MSE+{out_act}: max abs error en dL/db = {max_err_b:.2e}"
        )
        _ok(f"MSE + {out_act:10s}: max err dL/dW={max_err_W:.2e}, dL/db={max_err_b:.2e}")


# =====================================================================
# Test B — Validaciones del enunciado del TP3
# =====================================================================


def test_B_enunciado_validations():
    _print_section("Test B — Validaciones del enunciado (AND, y=x, y=tanh, XOR)")

    # B1) AND con perceptrón escalón
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
    t = np.array([-1, -1, -1, 1], dtype=float)
    p = SimplePerceptron(n_inputs=2, activation="step", initializer="random_normal",
                         init_scale=0.5, seed=7)
    opt = GradientDescent(lr=0.1)
    rng = np.random.default_rng(0)
    for _ in range(50):
        p.train_epoch(X, t, opt, batch_size=1, shuffle=True, rng=rng)
    preds = np.sign(p.predict(X))
    assert (preds == t).all(), f"AND no convergió: preds={preds}, target={t}"
    _ok(f"AND escalón: preds={preds.astype(int).tolist()} ✓")

    # B2) y = x con perceptrón lineal
    rng = np.random.default_rng(3)
    X_lin = rng.uniform(-1, 1, size=(50, 1))
    t_lin = X_lin[:, 0]
    p = SimplePerceptron(n_inputs=1, activation="identity", initializer="random_normal",
                         seed=11)
    opt = GradientDescent(lr=0.1)
    for _ in range(200):
        p.train_epoch(X_lin, t_lin, opt, batch_size=10, shuffle=True, rng=rng)
    err = float(np.mean((t_lin - p.predict(X_lin)) ** 2))
    assert err < 1e-5, f"y=x lineal: MSE={err:.2e} > 1e-5"
    _ok(f"y=x lineal: MSE={err:.2e}, W={p.W[0]:.4f} ✓")

    # B3) y = tanh(x) con perceptrón no-lineal
    rng = np.random.default_rng(4)
    X_t = rng.uniform(-2, 2, size=(50, 1))
    t_t = np.tanh(X_t[:, 0])
    p = SimplePerceptron(n_inputs=1, activation="tanh", initializer="random_normal", seed=13)
    opt = Adam(lr=0.05)
    for _ in range(500):
        p.train_epoch(X_t, t_t, opt, batch_size=10, shuffle=True, rng=rng)
    err = float(np.mean((t_t - p.predict(X_t)) ** 2))
    assert err < 1e-3, f"y=tanh no-lineal: MSE={err:.2e}"
    _ok(f"y=tanh no-lineal: MSE={err:.2e}, W={p.W[0]:.4f} ✓")

    # B4) XOR con MLP [2, 3, 2, 1]
    X_xor = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
    t_xor = np.array([[1], [1], [-1], [-1]], dtype=float)
    m = MLP(architecture=[2, 3, 2, 1], hidden_activation="tanh",
            output_activation="tanh", initializer="random_normal", seed=22, loss_name="mse")
    opt = Adam(lr=0.05)
    rng = np.random.default_rng(0)
    for _ in range(800):
        m.train_epoch(X_xor, t_xor, opt, batch_size=4, shuffle=True, rng=rng)
    preds = np.sign(m.predict(X_xor)).flatten()
    assert (preds == t_xor.flatten()).all(), (
        f"XOR no convergió: preds={preds}, target={t_xor.flatten()}"
    )
    _ok(f"XOR MLP [2,3,2,1]: preds={preds.astype(int).tolist()} ✓")


# =====================================================================
# Test C — Validaciones de configuración inválida
# =====================================================================


def test_C_invalid_configs():
    _print_section("Test C — Validaciones de configuración inválida")

    cases = [
        ("arch len < 2", dict(architecture=[2])),
        ("layer size 0", dict(architecture=[2, 0, 1])),
        ("layer size -1", dict(architecture=[2, -1, 1])),
        ("weight_decay < 0", dict(architecture=[2, 4, 4], weight_decay=-0.1)),
        ("loss desconocida", dict(architecture=[2, 4, 4], loss_name="foo")),
        ("activación desconocida", dict(architecture=[2, 4, 4], output_activation="foo")),
    ]
    for name, kwargs in cases:
        try:
            m = MLP(**kwargs)
            # Si no levantó en __init__, intentar un forward (algunas
            # activaciones inválidas se descubren en activate()).
            if "output_activation" in kwargs:
                m.predict(np.zeros((1, 2)))
            assert False, f"{name}: NO levantó error"
        except (ValueError, TypeError) as e:
            _ok(f"{name}: raise {type(e).__name__} ✓")


# =====================================================================
# Test D — Save / load
# =====================================================================


def test_D_save_load():
    _print_section("Test D — Save / load preserva config y predicciones")
    rng = np.random.default_rng(7)

    X = rng.normal(0, 1, size=(20, 4))
    t = np.tanh(X @ rng.normal(0, 0.3, size=(4, 2)))
    m = MLP(architecture=[4, 6, 2], hidden_activation="tanh",
            output_activation="tanh", initializer="random_normal",
            seed=33, loss_name="mse")
    opt = Adam(lr=0.01)
    rng_train = np.random.default_rng(0)
    for _ in range(20):
        m.train_epoch(X, t, opt, batch_size=8, shuffle=True, rng=rng_train)

    pred_before = m.predict(X)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        m.save(path)
        m2 = MLP.load(path)
        pred_after = m2.predict(X)
        assert np.allclose(pred_before, pred_after), (
            "Predicciones no coinciden tras save/load"
        )
        assert m2.architecture == m.architecture
        assert m2.hidden_activation == m.hidden_activation
        assert m2.output_activation == m.output_activation
        assert m2.loss_name == m.loss_name
        _ok("Save/load preserva predicciones y config ✓")
    finally:
        os.unlink(path)


# =====================================================================
# Test E — Regresión: SimplePerceptron import + smoke test
# =====================================================================


def test_E_regression_smoke():
    _print_section("Test E — Smoke test: SimplePerceptron entrena con MSE")
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, size=(80, 4))
    Y = np.tanh(X @ rng.normal(0, 0.3, size=(4, 2)))

    m = MLP(architecture=[4, 8, 2], hidden_activation="tanh",
            output_activation="tanh", initializer="random_normal",
            seed=99, loss_name="mse")
    opt = Adam(lr=0.01)
    rng_train = np.random.default_rng(0)
    initial_loss = mse(Y, m.predict(X))
    for _ in range(50):
        m.train_epoch(X, Y, opt, batch_size=16, shuffle=True, rng=rng_train)
    final_loss = mse(Y, m.predict(X))
    assert final_loss < initial_loss / 5, (
        f"MLP+MSE no convergió: {initial_loss:.4f} → {final_loss:.4f}"
    )
    _ok(f"MLP+MSE: loss {initial_loss:.4f} → {final_loss:.4f} ✓")


# =====================================================================
# Main
# =====================================================================


def main():
    print("Tests del MLP — arquitectura decoupled (loss_deriv * activate_deriv)")
    tests = [
        test_A_gradient_check,
        test_B_enunciado_validations,
        test_C_invalid_configs,
        test_D_save_load,
        test_E_regression_smoke,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"\n[FAIL] {t.__name__}:\n  {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"\n[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed.append(t.__name__)

    print()
    print("=" * 64)
    if failed:
        print(f"FALLARON {len(failed)} de {len(tests)} tests:")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    else:
        print(f"=== TODOS LOS TESTS PASARON ({len(tests)}/{len(tests)}) ===")


if __name__ == "__main__":
    main()
