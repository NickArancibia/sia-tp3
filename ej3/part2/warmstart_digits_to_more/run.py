"""EJ3 part2 - Warm-start: pretrain en digits, finetune en more_digits.

Etapa 1: entrena en digits.csv (9 clases, falta el 8), guarda modelo + scaler.
Etapa 2: carga el modelo y continua entrenamiento con more_digits.csv.
El scaler se mantiene del pretrain para continuidad en la escala.
"""
import os
import pickle
import sys

EJ3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ3_DIR)
sys.path.insert(0, EJ3_DIR)
sys.path.insert(0, REPO_DIR)

import numpy as np

from common import (RESULTS_PART2, TEST_CSV, build_mlp,
                    evaluate_on_test, train_model)
from shared.digit_dataset_loader import load_dataset
from shared.mlp import MLP
from shared.optimizers import Adam
from shared.preprocessing import build_scaler, one_hot_encode, stratified_split

EJ2_DATA_DIR = os.path.join(REPO_DIR, "ej2", "data")
EJ3_DATA_DIR = os.path.join(EJ3_DIR, "data")
DIGITS_TRAIN = os.path.join(EJ2_DATA_DIR, "digits.csv")
MORE_TRAIN = os.path.join(EJ3_DATA_DIR, "more_digits.csv")

SELECTED = {
    "arch": [784, 128, 64, 10],
    "lr": 1e-3,
    "batch_size": 32,
    "init_scale": 0.1,
    "weight_decay": 0.0,
    "max_epochs": 150,
    "patience": 20,
}
PRETRAIN = {
    "max_epochs": 150,
    "patience": 20,
}
SEEDS = [0, 1, 2]

OUT_DIR = os.path.join(RESULTS_PART2, "warmstart_digits_to_more")


def _prepare_data_with_scaler(train_csv, test_csv, scaler, fit_scaler,
                              val_frac=0.2, seed=42):
    train_df = load_dataset(train_csv)
    test_df = load_dataset(test_csv)

    X_all = np.stack(train_df["image"].values)
    y_all = train_df["label"].values.astype(int)

    X_test_raw = np.stack(test_df["image"].values)
    y_test = test_df["label"].values.astype(int)

    train_idx, val_idx, _ = stratified_split(y_all, val_frac, 0.0, seed)

    if scaler is not None:
        if fit_scaler:
            scaler.fit(X_all[train_idx])
        X_train = scaler.transform(X_all[train_idx])
        X_val = scaler.transform(X_all[val_idx])
        X_test = scaler.transform(X_test_raw)
    else:
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_test_raw

    n_classes = max(int(y_all.max()), int(y_test.max())) + 1
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_all[train_idx],
        "y_val": y_all[val_idx],
        "y_test": y_test,
        "Y_train": one_hot_encode(y_all[train_idx], n_classes),
        "Y_val": one_hot_encode(y_all[val_idx], n_classes),
        "Y_test": one_hot_encode(y_test, n_classes),
        "scaler": scaler,
        "n_classes": n_classes,
    }
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        scaler = build_scaler("z-score")

        # Pretrain en digits
        data_pre = _prepare_data_with_scaler(
            train_csv=DIGITS_TRAIN,
            test_csv=TEST_CSV,
            scaler=scaler,
            fit_scaler=True,
            val_frac=0.2,
            seed=42,
        )
        print(f"Pretrain data: train={data_pre['X_train'].shape} "
              f"val={data_pre['X_val'].shape} test={data_pre['X_test'].shape}")

        model = build_mlp(
            SELECTED["arch"],
            seed=seed,
            init_scale=SELECTED["init_scale"],
            weight_decay=SELECTED["weight_decay"],
        )
        opt = Adam(lr=SELECTED["lr"])
        hist_pre = train_model(
            model,
            data_pre,
            opt,
            max_epochs=PRETRAIN["max_epochs"],
            batch_size=SELECTED["batch_size"],
            early_stopping_patience=PRETRAIN["patience"],
            verbose=True,
            seed=seed,
        )
        ev_pre = evaluate_on_test(model, data_pre)
        print(f"  pretrain test_acc={ev_pre['test_acc']:.4f} "
              f"val_acc={ev_pre['val_acc']:.4f}")

        pre_model_path = os.path.join(OUT_DIR, f"pretrain_model_seed{seed}.npz")
        model.save(pre_model_path)
        if seed == SEEDS[0] and data_pre["scaler"] is not None:
            data_pre["scaler"].save(os.path.join(OUT_DIR, "pretrain_scaler.npz"))

        # Fine-tune en more_digits usando el mismo scaler
        data_ft = _prepare_data_with_scaler(
            train_csv=MORE_TRAIN,
            test_csv=TEST_CSV,
            scaler=scaler,
            fit_scaler=False,
            val_frac=0.2,
            seed=42,
        )
        print(f"Finetune data: train={data_ft['X_train'].shape} "
              f"val={data_ft['X_val'].shape} test={data_ft['X_test'].shape}")

        model = MLP.load(pre_model_path)
        opt = Adam(lr=SELECTED["lr"])
        hist_ft = train_model(
            model,
            data_ft,
            opt,
            max_epochs=SELECTED["max_epochs"],
            batch_size=SELECTED["batch_size"],
            early_stopping_patience=SELECTED["patience"],
            verbose=True,
            seed=seed,
        )
        ev_ft = evaluate_on_test(model, data_ft)
        print(f"  finetune test_acc={ev_ft['test_acc']:.4f} "
              f"val_acc={ev_ft['val_acc']:.4f}")

        n_params = sum(W.size + b.size for W, b in zip(model.weights, model.biases))
        row = {
            "seed": seed,
            "n_params": n_params,
            "config": {
                "selected": SELECTED,
                "pretrain": PRETRAIN,
                "scaler": "z-score (digits)",
            },
            "pretrain_hist": hist_pre,
            "pretrain_eval": ev_pre,
            "finetune_hist": hist_ft,
            "finetune_eval": ev_ft,
        }
        results.append(row)

        if seed == SEEDS[0]:
            model.save(os.path.join(OUT_DIR, "finetuned_model.npz"))

    out_path = os.path.join(OUT_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
