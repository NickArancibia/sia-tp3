"""EJ3 part2 — Modelo seleccionado para alcanzar accuracy ≥ 98%.

Hiperparámetros más generosos que EJ2 (arquitectura más grande, más epochs,
weight decay liviano para regularizar). Multi-seed para estabilidad.
"""
import os
import pickle
import sys

EJ3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ3_DIR)
sys.path.insert(0, EJ3_DIR)
sys.path.insert(0, REPO_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam

# Combo más grande para alcanzar 98%. Arquitectura wider+deeper, más epochs,
# weight decay para regularizar.
SELECTED = {
    "arch": [784, 256, 128, 10],
    "lr": 5e-4,
    "batch_size": 64,
    "init_scale": 0.05,
    "weight_decay": 1e-5,
    "max_epochs": 150,
    "patience": 20,
}
SEEDS = [0, 1, 2]

OUT_DIR = os.path.join(RESULTS_PART2, "selected_model")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2,
                        scaler="z-score", seed=42)
    print(f"Modelo seleccionado: {SELECTED}")
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape} "
          f"test={data['X_test'].shape}")

    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        model = build_mlp(SELECTED["arch"], seed=seed,
                          init_scale=SELECTED["init_scale"],
                          weight_decay=SELECTED["weight_decay"])
        opt = Adam(lr=SELECTED["lr"])
        hist = train_model(model, data, opt,
                           max_epochs=SELECTED["max_epochs"],
                           batch_size=SELECTED["batch_size"],
                           early_stopping_patience=SELECTED["patience"],
                           verbose=True, seed=seed)
        ev = evaluate_on_test(model, data)
        n_params = sum(W.size + b.size for W, b in zip(model.weights,
                                                       model.biases))
        row = {"seed": seed, "n_params": n_params,
               "config": SELECTED, **hist, **ev}
        results.append(row)
        print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f}")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
