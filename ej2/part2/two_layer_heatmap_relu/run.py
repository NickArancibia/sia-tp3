"""EJ2 part2 — Two-layer heatmap: barrer (n1, n2) en {0,8,16,32,64,128,256,512}^2.

n=0 significa "esa capa no existe". Como (0, n) y (n, 0) producen el
mismo MLP ([784, n, 10]), esos pares se calculan UNA sola vez y se
comparten entre las dos celdas del heatmap (esto se resuelve en plot.py).

Configs únicas (con SIZES de 8 elementos incluyendo 0):
- (0, 0)               → [784, 10]                — 1 run
- (0, n) para n>0      → [784, n, 10]             — 7 runs
- (m, n) para m,n>0    → [784, m, n, 10]          — 49 runs
Total: 57 configs × 5 seeds = 285 entrenamientos.

Es INCREMENTAL: si results.pkl existe, salta las (n1, n2, seed) que
ya están adentro y solo corre las nuevas. Guarda después de CADA
config para que sea seguro interrumpir.

Hiperparámetros alineados con width_sweep, depth_sweep y
ej2/part/architecture (Nick): Adam lr=1e-3, batch=32, max_epochs=500,
patience=50, tanh, init_scale=0.1.

Output: results.pkl (lista de dicts).
"""
import os
import pickle
import sys
import time

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV,
                    evaluate_on_test, prepare_data, train_model)
from shared.mlp import MLP
from shared.optimizers import Adam

SIZES = [0, 8, 16, 32, 64, 128, 256, 512]
SEEDS = [0]               # SOLO 1 seed (para velocidad; std se rellena en plot)
MAX_EPOCHS = 500
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 50

OUT_DIR = os.path.join(RESULTS_PART2, "two_layer_heatmap_relu")


def make_arch(n1, n2):
    arch = [784]
    if n1 > 0:
        arch.append(n1)
    if n2 > 0:
        arch.append(n2)
    arch.append(10)
    return arch


def get_unique_configs():
    """(0,0) + 5x (0,n) + 25x (m,n) = 31 configs únicas."""
    configs = [(0, 0)]
    for n in SIZES[1:]:
        configs.append((0, n))
    for m in SIZES[1:]:
        for n in SIZES[1:]:
            configs.append((m, n))
    return configs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.pkl")

    # Resume: cargar resultados existentes si los hay
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            results = pickle.load(f)
        done = {(r["n1"], r["n2"], r["seed"]) for r in results}
        print(f"Resumiendo: {len(results)} corridas ya en results.pkl "
              f"({len(done)} (n1,n2,seed) únicas).")
    else:
        results = []
        done = set()

    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape} "
          f"test={data['X_test'].shape}")

    configs = get_unique_configs()
    pending = [(n1, n2, seed) for n1, n2 in configs for seed in SEEDS
               if (n1, n2, seed) not in done]
    total = len(pending)
    print(f"Configs únicas: {len(configs)}, seeds: {len(SEEDS)} → "
          f"{len(configs)*len(SEEDS)} totales, {total} pendientes.")

    if total == 0:
        print("Nada que hacer, todos los runs ya están.")
        return

    t_start = time.perf_counter()
    for run_idx, (n1, n2, seed) in enumerate(pending, 1):
        arch = make_arch(n1, n2)
        label = "-".join(str(n) for n in arch)
        elapsed_total = time.perf_counter() - t_start
        print(f"\n[{run_idx}/{total}] n1={n1} n2={n2} arch={label} seed={seed} "
              f"(elapsed={elapsed_total:.0f}s)")
        model = MLP(architecture=arch,
                    hidden_activation="relu",
                    output_activation="logistic",
                    initializer="he_normal",
                    init_scale=0.1,
                    seed=seed)
        opt = Adam(lr=LR)
        hist = train_model(model, data, opt,
                           max_epochs=MAX_EPOCHS,
                           batch_size=BATCH_SIZE,
                           early_stopping_patience=PATIENCE,
                           verbose=False, seed=seed)
        ev = evaluate_on_test(model, data)
        n_params = sum(W.size + b.size for W, b in zip(model.weights,
                                                       model.biases))
        time_per_epoch = hist["elapsed"] / max(1, hist["stopped_at"])
        row = {"n1": n1, "n2": n2, "arch": arch, "arch_label": label,
               "n_params": n_params, "seed": seed,
               "time_per_epoch": time_per_epoch,
               **hist, **ev}
        results.append(row)
        print(f"  test_acc={ev['test_acc']:.4f} "
              f"val_acc={ev['val_acc']:.4f} "
              f"train_acc={ev['train_acc']:.4f} "
              f"params={n_params:,} stopped@{hist['stopped_at']} "
              f"elapsed={hist['elapsed']:.1f}s ({time_per_epoch:.2f}s/ep)")

        # Guardar después de CADA run para que interrumpir no pierda nada
        with open(out_path, "wb") as f:
            pickle.dump(results, f)

    print(f"\nGuardado en {out_path}")
    print(f"Tiempo total esta corrida: {(time.perf_counter() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
