"""EJ2 part2 — Modelo seleccionado [784, 512, 512, 10] con ReLU + He.

Basado en los resultados del two_layer_heatmap (ganador: n1=512, n2=512)
y de la conclusión de Exp4 (ReLU > tanh > logistic).

Hiperparámetros:
- Arch: [784, 512, 512, 10]
- Activación oculta: ReLU
- Init: He normal (recomendado para ReLU)
- Optimizer: Adam lr=1e-3 (consistente con el sweep batch_lr/optimizer_lr)
- Batch: 32, max_epochs=150, patience=20
- 5 seeds para mean ± std
"""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV,
                    evaluate_on_test, prepare_data, train_model)
from shared.mlp import MLP
from shared.optimizers import Adam

SELECTED = {
    "arch": [784, 512, 10],
    "hidden_act": "relu",
    "lr": 1e-3,
    "optimizer": "adam",
    "batch_size": 32,
    "init_scale": 0.1,         # ignorado por he_normal
    "initializer": "he_normal",
    "max_epochs": 150,
    "patience": 20,
}
SEEDS = [0, 1, 2, 3, 4]

OUT_DIR = os.path.join(RESULTS_PART2, "selected_model_512_1layer")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score",
                        stratify=True, seed=42)
    print(f"Modelo seleccionado: {SELECTED}")
    print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape} "
          f"test={data['X_test'].shape}")

    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        model = MLP(architecture=SELECTED["arch"],
                    hidden_activation=SELECTED["hidden_act"],
                    output_activation="logistic",
                    initializer=SELECTED["initializer"],
                    init_scale=SELECTED["init_scale"],
                    seed=seed)
        opt = Adam(lr=SELECTED["lr"])
        hist = train_model(model, data, opt,
                           max_epochs=SELECTED["max_epochs"],
                           batch_size=SELECTED["batch_size"],
                           early_stopping_patience=SELECTED["patience"],
                           verbose=False, seed=seed)
        ev = evaluate_on_test(model, data)
        n_params = sum(W.size + b.size for W, b in zip(model.weights,
                                                       model.biases))
        row = {"seed": seed, "n_params": n_params,
               "config": SELECTED, **hist, **ev}
        results.append(row)
        print(f"  test_acc={ev['test_acc']:.4f} val_acc={ev['val_acc']:.4f} "
              f"train_acc={ev['train_acc']:.4f} stopped@{hist['stopped_at']}")

        if seed == SEEDS[0]:
            model.save(os.path.join(OUT_DIR, "model.npz"))
            if data["scaler"] is not None:
                data["scaler"].save(os.path.join(OUT_DIR, "scaler.npz"))

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
