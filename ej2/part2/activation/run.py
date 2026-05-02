"""EJ2 part2 — Variar función de activación intermedia (tanh vs logistic vs relu).

Mantiene fijo: arch [784, 64, 32, 10], Adam lr=1e-3, batch=32, output_act=logistic.
Comparación de las 3 activaciones intermedias mencionadas en el material
(Clase 10.1 escalón / Clase 10.2 sigmoide y tanh / extras: relu).
"""
import os
import pickle
import sys
import time

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam

ACTIVATIONS = ["tanh", "logistic", "relu"]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 60
PATIENCE = 12
ARCH = [784, 64, 32, 10]
LR = 1e-3
BATCH_SIZE = 32

OUT_DIR = os.path.join(RESULTS_PART2, "activation")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape}")

    results = []
    for act in ACTIVATIONS:
        for seed in SEEDS:
            print(f"\nact={act} seed={seed}")
            model = build_mlp(ARCH, hidden_act=act, seed=seed, init_scale=0.1)
            opt = Adam(lr=LR)
            t0 = time.time()
            hist = train_model(model, data, opt,
                               max_epochs=MAX_EPOCHS,
                               batch_size=BATCH_SIZE,
                               early_stopping_patience=PATIENCE,
                               verbose=False, seed=seed)
            ev = evaluate_on_test(model, data)
            elapsed = time.time() - t0
            row = {"activation": act, "seed": seed, **hist, **ev}
            results.append(row)
            print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
                  f"stopped={hist['stopped_at']} elapsed={elapsed:.1f}s")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
