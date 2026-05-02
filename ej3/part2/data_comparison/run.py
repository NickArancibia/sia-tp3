"""EJ3 part2 — Comparación digits.csv vs more_digits.csv.

Entrena la MISMA arquitectura con dos datasets distintos:
- digits.csv (12449 samples, 9 clases — falta el 8)
- more_digits.csv (15741 samples, 10 clases)

Evalúa ambos sobre el MISMO test (digits_test.csv, 10 clases).
Esto aísla el efecto de "más datos + clase 8 incluida" vs el modelo en sí.
"""
import os
import pickle
import sys

EJ3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ3_DIR)
sys.path.insert(0, EJ3_DIR)
sys.path.insert(0, REPO_DIR)

from common import (RESULTS_PART2, TEST_CSV, build_mlp, evaluate_on_test,
                    prepare_data, train_model)
from shared.optimizers import Adam

# Dos datasets de entrenamiento; mismo test.
EJ2_DATA_DIR = os.path.join(REPO_DIR, "ej2", "data")
EJ3_DATA_DIR = os.path.join(EJ3_DIR, "data")
DATASETS = [
    {"name": "digits", "path": os.path.join(EJ2_DATA_DIR, "digits.csv")},
    {"name": "more_digits", "path": os.path.join(EJ3_DATA_DIR, "more_digits.csv")},
]
SEEDS = [0, 1, 2]
ARCH = [784, 128, 64, 10]
LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 20

OUT_DIR = os.path.join(RESULTS_PART2, "data_comparison")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for ds in DATASETS:
        for seed in SEEDS:
            print(f"\ndataset={ds['name']} seed={seed}")
            data = prepare_data(train_csv=ds["path"], test_csv=TEST_CSV,
                                val_frac=0.2, scaler="z-score", seed=42)
            print(f"  train={data['X_train'].shape} val={data['X_val'].shape} "
                  f"test={data['X_test'].shape} n_classes={data['n_classes']}")
            model = build_mlp(ARCH, seed=seed, init_scale=0.1)
            opt = Adam(lr=LR)
            hist = train_model(model, data, opt,
                               max_epochs=MAX_EPOCHS,
                               batch_size=BATCH_SIZE,
                               early_stopping_patience=PATIENCE,
                               verbose=False, seed=seed)
            ev = evaluate_on_test(model, data)
            row = {"dataset": ds["name"], "seed": seed, **hist, **ev}
            results.append(row)
            print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
                  f"stopped={hist['stopped_at']}")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
