"""EJ2 part2 — Variantes de tasa de aprendizaje.

Barre lr ∈ {1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4} con Adam (optimizador
default), arquitectura baseline [784,40,20,10], 3 seeds.

Output: results.pkl con lista de dicts {lr, seed, train_losses, val_losses,
train_accs, val_accs, test_acc, val_acc, train_acc, test_cm, ...}.
"""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam

LRS = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 100
BATCH_SIZE = 32
ARCH = [784, 40, 20, 10]
PATIENCE = 15

OUT_DIR = os.path.join(RESULTS_PART2, "learning_rate")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape} "
          f"test={data['X_test'].shape} n_classes={data['n_classes']}")

    results = []
    for lr in LRS:
        for seed in SEEDS:
            print(f"\nlr={lr:.0e} seed={seed}")
            model = build_mlp(ARCH, seed=seed, init_scale=0.1)
            opt = Adam(lr=lr)
            hist = train_model(model, data, opt,
                               max_epochs=MAX_EPOCHS,
                               batch_size=BATCH_SIZE,
                               early_stopping_patience=PATIENCE,
                               verbose=False, seed=seed)
            ev = evaluate_on_test(model, data)
            row = {"lr": lr, "seed": seed, **hist, **ev}
            results.append(row)
            print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
                  f"stopped={hist['stopped_at']} elapsed={hist['elapsed']:.1f}s")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
