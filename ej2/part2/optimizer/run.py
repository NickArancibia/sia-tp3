"""EJ2 part2 — Variantes de mecanismo de optimización.

Compara 4 setups: GD (full batch), SGD mini-batch, Momentum, Adam.
SGD online (batch=1) se omite por costo computacional (cubierto en
`batch_lr/` con timeout). Para cada optimizador se elige un lr razonable
según el material (Clase 9 — Optimización); la justificación numérica
del lr por optimizador se hace en el sweep `optimizer_lr/`.
"""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam, GradientDescent, Momentum

# Para cada optimizador, lr y batch_size apropiados:
# - GD full batch: lr más grande (gradiente promedio, baja varianza)
# - SGD online: lr más chico (alta varianza por sample)
# - Momentum mini-batch: lr intermedio
# - Adam mini-batch: lr 1e-3 estándar
OPTIMIZERS = [
    {"name": "gd_full",     "ctor": lambda: GradientDescent(lr=0.1),     "batch": 0},
    {"name": "sgd_mini",    "ctor": lambda: GradientDescent(lr=0.05),    "batch": 32},
    {"name": "momentum",    "ctor": lambda: Momentum(lr=0.05, momentum=0.9), "batch": 32},
    {"name": "adam",        "ctor": lambda: Adam(lr=1e-3),               "batch": 32},
]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 60
ARCH = [784, 40, 20, 10]
PATIENCE = 12

OUT_DIR = os.path.join(RESULTS_PART2, "optimizer")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape}")

    results = []
    for opt_spec in OPTIMIZERS:
        for seed in SEEDS:
            print(f"\nopt={opt_spec['name']} batch={opt_spec['batch']} seed={seed}")
            model = build_mlp(ARCH, seed=seed, init_scale=0.1)
            opt = opt_spec["ctor"]()
            hist = train_model(model, data, opt,
                               max_epochs=MAX_EPOCHS,
                               batch_size=opt_spec["batch"],
                               early_stopping_patience=PATIENCE,
                               verbose=False, seed=seed)
            ev = evaluate_on_test(model, data)
            row = {"opt_name": opt_spec["name"],
                   "batch_size": opt_spec["batch"],
                   "seed": seed, **hist, **ev}
            results.append(row)
            print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
                  f"stopped={hist['stopped_at']} elapsed={hist['elapsed']:.1f}s")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
