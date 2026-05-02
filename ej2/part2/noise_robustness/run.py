"""EJ2 part2 — Robustez al ruido (opcional del enunciado).

Toma el modelo seleccionado, agrega ruido gaussiano N(0, σ²) al test set para
varios σ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} y mide la accuracy.
Usa varias seeds para el ruido para tener varianza.
"""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

import numpy as np

from common import RESULTS_PART2, TEST_CSV, TRAIN_CSV, prepare_data
from shared.metrics import accuracy, classify_from_output
from shared.mlp import MLP
from shared.preprocessing import ZScoreScaler

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
NOISE_SEEDS = [0, 1, 2, 3, 4]

OUT_DIR = os.path.join(RESULTS_PART2, "noise_robustness")
SELECTED_DIR = os.path.join(RESULTS_PART2, "selected_model")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model_path = os.path.join(SELECTED_DIR, "model.npz")
    if not os.path.exists(model_path):
        print(f"ERROR: corré primero ej2/part2/selected_model/run.py "
              f"(falta {model_path})")
        sys.exit(1)

    model = MLP.load(model_path)
    print(f"Modelo cargado. arch={model.architecture}")

    # Cargar test SIN scaling para agregar ruido en el espacio original [0,1]
    # y luego scalear con el scaler entrenado.
    data_raw = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2,
                            scaler=None, seed=42)
    scaler_path = os.path.join(SELECTED_DIR, "scaler.npz")
    scaler = ZScoreScaler.load(scaler_path)
    X_test_raw = data_raw["X_test"]
    y_test = data_raw["y_test"]
    print(f"Test: {X_test_raw.shape}, ruido aplicado en espacio [0,1] antes de scalear")

    results = []
    for sigma in NOISE_LEVELS:
        accs = []
        for seed in NOISE_SEEDS:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, sigma, size=X_test_raw.shape)
            X_noisy = np.clip(X_test_raw + noise, 0.0, 1.0)
            X_scaled = scaler.transform(X_noisy)
            pred = classify_from_output(model.predict(X_scaled))
            accs.append(accuracy(y_test, pred))
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        results.append({"sigma": sigma, "mean_acc": mean_acc,
                        "std_acc": std_acc, "all_accs": accs})
        print(f"σ={sigma:.2f} → acc = {mean_acc:.4f} ± {std_acc:.4f}")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
