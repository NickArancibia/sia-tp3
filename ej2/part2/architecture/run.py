"""EJ2 part2 — Variantes de arquitectura.

Barre 5 arquitecturas distintas con Adam lr=1e-3, 3 seeds.
Cubre: regresión logística (sin hidden), 1 capa oculta (32, 64), 2 capas
ocultas (40-20, 64-32). El sweep `learning_rate/` ya validó que lr=1e-3 está
en la ventana óptima de Adam.

Nota: usamos batch=64 y max_epochs=50 (en lugar de batch=32 / 100 ep del
sweep de lr) por presupuesto de cómputo. Eso significa que los números
absolutos no son directamente comparables con `learning_rate/`, pero sí lo
son entre arquitecturas (comparación interna válida).

Output: results.pkl con lista de dicts.
"""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam

ARCHS = [
    [784, 10],                    # logistic regression (sin hidden)
    [784, 32, 10],                # 1 capa oculta chica
    [784, 64, 10],                # 1 capa oculta mediana
    [784, 40, 20, 10],            # baseline (2 capas)
    [784, 64, 32, 10],            # 2 capas grandes
]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10

OUT_DIR = os.path.join(RESULTS_PART2, "architecture")


def arch_label(arch):
    return "-".join(str(n) for n in arch)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape} "
          f"test={data['X_test'].shape}")

    results = []
    for arch in ARCHS:
        for seed in SEEDS:
            label = arch_label(arch)
            print(f"\narch={label} seed={seed}")
            model = build_mlp(arch, seed=seed, init_scale=0.1)
            opt = Adam(lr=LR)
            hist = train_model(model, data, opt,
                               max_epochs=MAX_EPOCHS,
                               batch_size=BATCH_SIZE,
                               early_stopping_patience=PATIENCE,
                               verbose=False, seed=seed)
            ev = evaluate_on_test(model, data)
            n_params = sum(W.size + b.size for W, b in zip(model.weights,
                                                           model.biases))
            row = {"arch": arch, "arch_label": label, "n_params": n_params,
                   "seed": seed, **hist, **ev}
            results.append(row)
            print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
                  f"params={n_params:,} elapsed={hist['elapsed']:.1f}s")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
