"""EJ2 part2 — Sweep 2D optimizer × learning rate (heatmap).

Refina el experimento `optimizer/`: en vez de fijar un lr "razonable" por
optimizador, hace un sweep 2D para mostrar que cada uno tiene su lr óptimo
distinto. Esto justifica visualmente la observación de Nick: Adam puede usar
lr más chico que SGD/Momentum.

Sin SGD online (batch=1) — ya cubierto en `optimizer/` antiguo y muy lento.
"""
import os
import pickle
import sys
import time

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam, GradientDescent, Momentum

OPTIMIZERS = ["sgd_mini", "momentum", "adam"]
LRS = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 50
PATIENCE = 10
ARCH = [784, 64, 32, 10]
BATCH_SIZE = 32

OUT_DIR = os.path.join(RESULTS_PART2, "optimizer_lr")


def make_opt(name, lr):
    # "sgd_mini" = GradientDescent con BATCH_SIZE=32 (i.e. SGD mini-batch).
    # No es GD full-batch — usar el experimento `optimizer/` para esa comparación.
    if name == "sgd_mini":
        return GradientDescent(lr=lr)
    if name == "momentum":
        return Momentum(lr=lr, momentum=0.9)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(name)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape}")

    results = []
    for opt_name in OPTIMIZERS:
        for lr in LRS:
            for seed in SEEDS:
                label = f"opt={opt_name}, lr={lr:.0e}, seed={seed}"
                print(f"\n{label}")
                model = build_mlp(ARCH, seed=seed, init_scale=0.1)
                opt = make_opt(opt_name, lr)
                t0 = time.time()
                hist = train_model(model, data, opt,
                                   max_epochs=MAX_EPOCHS,
                                   batch_size=BATCH_SIZE,
                                   early_stopping_patience=PATIENCE,
                                   verbose=False, seed=seed)
                ev = evaluate_on_test(model, data)
                elapsed = time.time() - t0
                row = {"opt_name": opt_name, "lr": lr, "seed": seed,
                       **hist, **ev}
                results.append(row)
                print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
                      f"stopped={hist['stopped_at']} elapsed={elapsed:.1f}s")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
