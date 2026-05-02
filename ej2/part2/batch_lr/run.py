"""EJ2 part2 — Sweep 2D batch_size × learning rate (heatmap).

Replica el patrón de ej1/part2/batch_lr/. Para cada combinación batch × lr,
entrena con Adam y arquitectura ganadora del sweep de architecture
([784, 64, 32, 10]). 3 seeds por combo.

Output: results.pkl con lista de dicts {lr, batch_size, seed, train_losses,
val_accs, test_acc, val_acc, train_acc, ...}.

Nota: batch=1 (online) es el más lento. Si en runtime tarda > 60s/seed lo
saltamos y reportamos en summary.
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

LRS = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
# 0 = full batch en train_epoch; usamos -1 como tag y luego mapeamos
BATCH_SIZES = [1, 8, 32, 128, 0]   # 0 = full batch
SEEDS = [0, 1, 2]
MAX_EPOCHS = 50
PATIENCE = 10
ARCH = [784, 64, 32, 10]
ONLINE_TIMEOUT_S = 60.0   # si seed=0 con batch=1 tarda más, salteamos batch=1

OUT_DIR = os.path.join(RESULTS_PART2, "batch_lr")


def batch_label(b):
    if b == 0:
        return "full"
    if b == 1:
        return "online"
    return str(b)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape}")

    skip_online = False
    results = []
    for batch in BATCH_SIZES:
        if batch == 1 and skip_online:
            print(f"\nSALTANDO batch=1 (anterior seed pasó {ONLINE_TIMEOUT_S}s)")
            continue
        for lr in LRS:
            for seed in SEEDS:
                label = f"batch={batch_label(batch)}, lr={lr:.0e}, seed={seed}"
                print(f"\n{label}")
                model = build_mlp(ARCH, seed=seed, init_scale=0.1)
                opt = Adam(lr=lr)
                t0 = time.time()
                hist = train_model(model, data, opt,
                                   max_epochs=MAX_EPOCHS,
                                   batch_size=batch,
                                   early_stopping_patience=PATIENCE,
                                   verbose=False, seed=seed)
                ev = evaluate_on_test(model, data)
                elapsed = time.time() - t0
                row = {"lr": lr, "batch_size": batch,
                       "batch_label": batch_label(batch),
                       "seed": seed, **hist, **ev}
                results.append(row)
                print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
                      f"stopped={hist['stopped_at']} elapsed={elapsed:.1f}s")
                if batch == 1 and seed == 0 and elapsed > ONLINE_TIMEOUT_S:
                    print(f"  WARNING: batch=1 muy lento — salteo el resto de batch=1")
                    skip_online = True
                    break
            if skip_online and batch == 1:
                break

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
