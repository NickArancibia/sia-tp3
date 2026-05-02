"""EJ2 part2 — Modelo seleccionado, multi-seed, evaluación final.

Toma el mejor combo (best lr × best arch × best optimizer) según los sweeps
previos, lo entrena con 5 seeds y reporta:
- Test accuracy mean ± std
- Confusion matrix promedio (test)
- Per-class metrics (precision, recall, F1)
- Tiempo de entrenamiento
- Curvas de loss/acc

El "mejor combo" se hardcodea acá basándose en los summaries de las otras
carpetas; si los sweeps cambian, actualizar SELECTED.
"""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam

# Combo seleccionado tras analizar los sweeps de lr/arch/optimizer/optimizer_lr:
#   - arch ganadora del sweep `architecture/`: [784, 64, 32, 10]
#   - lr ganador del sweep `optimizer_lr/` para Adam: 5e-3 (val=0.948)
#     (idem `learning_rate/`: lr=5e-3 val=0.942)
#   - optimizer: Adam (gana en `optimizer/` y `optimizer_lr/`)
#   - batch_size=32: sweet spot del sweep `batch_lr/`
SELECTED = {
    "arch": [784, 64, 32, 10],
    "lr": 5e-3,
    "optimizer": "adam",
    "batch_size": 32,
    "init_scale": 0.1,
    "max_epochs": 150,
    "patience": 20,
}
SEEDS = [0, 1, 2, 3, 4]

OUT_DIR = os.path.join(RESULTS_PART2, "selected_model")


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
        model = build_mlp(SELECTED["arch"], seed=seed,
                          init_scale=SELECTED["init_scale"])
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

        # Guardar el modelo de la PRIMERA seed para que noise_robustness
        # use uno de los 5 modelos efectivamente reportados (no uno extra).
        if seed == SEEDS[0]:
            model.save(os.path.join(OUT_DIR, "model.npz"))
            if data["scaler"] is not None:
                data["scaler"].save(os.path.join(OUT_DIR, "scaler.npz"))
            print(f"  modelo guardado (seed {seed})")

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
