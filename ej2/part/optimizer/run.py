"""Experimento 2: optimizador × learning rate con batch_size=32 fijo.

Produce:
  results/part/optimizer/raw.csv      — métricas finales por (config, seed)
  results/part/optimizer/curves.csv   — pérdida por (config, seed, epoca)
  results/part/optimizer/summary.csv  — media ± std por config
"""
import os
import sys

EJ2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ2_DIR)
PART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EJ2_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PART_DIR)

import pandas as pd

from _common import append_csv, load_raw, split_scale, summarize_group, train
from shared.config_loader import load_config

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "optimizer")
BATCH_SIZE = 32

OPTIMIZERS = [
    {"name": "gd",       "optimizer": "gd"},
    {"name": "momentum", "optimizer": "momentum"},
    {"name": "adam",     "optimizer": "adam"},
]

RAW_FIELDS = [
    "config_name", "optimizer", "learning_rate", "seed",
    "train_acc", "val_acc", "test_acc", "macro_f1", "epochs",
]
CURVES_FIELDS = ["config_name", "seed", "epoch", "train_loss", "val_loss"]
SUMMARY_FIELDS = [
    "config_name", "optimizer", "learning_rate",
    "mean_val_acc", "std_val_acc", "mean_test_acc", "std_test_acc",
    "mean_macro_f1", "std_macro_f1", "mean_epochs",
]


def main():
    cfg = load_config(os.path.join(EJ2_DIR, "config.yaml"))
    search = cfg["search"]
    seeds = [int(s) for s in search["seeds"]]
    lrs = [float(lr) for lr in search["learning_rates"]]

    full_cfg_arch = cfg["model"]["architecture"]
    hidden = full_cfg_arch[1:-1]
    hidden_act = cfg["model"].get("hidden_activation", "tanh")
    max_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping"].get("patience", 20)
    adam_betas = cfg["training"].get("adam_betas", [0.9, 0.999])
    adam_eps = cfg["training"].get("adam_eps", 1e-8)
    momentum = cfg["training"].get("momentum", 0.9)

    X_all, y_all, X_test_raw, y_test, n_classes = load_raw(cfg, EJ2_DIR)
    full_arch = [X_all.shape[1]] + hidden + [n_classes]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, "raw.csv")
    curves_path = os.path.join(RESULTS_DIR, "curves.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    for p in (raw_path, curves_path, summary_path):
        if os.path.exists(p):
            os.remove(p)

    summaries = []
    for opt in OPTIMIZERS:
        for lr in lrs:
            cname = f"{opt['name']}_lr{lr:.0e}"
            opt_cfg = {
                "optimizer": opt["optimizer"],
                "learning_rate": lr,
                "momentum": momentum,
                "adam_betas": adam_betas,
                "adam_eps": adam_eps,
            }

            print(f"\n{'='*50}")
            print(f"{cname}: opt={opt['name']}, batch=mini{BATCH_SIZE}, lr={lr:.0e}")
            print(f"{'='*50}")

            for seed in seeds:
                X_tr, y_tr, X_va, y_va, X_te = split_scale(X_all, y_all, X_test_raw, cfg, seed)
                result = train(X_tr, y_tr, X_va, y_va, X_te, y_test, n_classes,
                               full_arch, hidden_act, opt_cfg, BATCH_SIZE, seed, max_epochs, patience)

                append_csv(raw_path, [{
                    "config_name": cname, "optimizer": opt["name"],
                    "learning_rate": lr, "seed": seed,
                    "train_acc": result["train_acc"], "val_acc": result["val_acc"],
                    "test_acc": result["test_acc"], "macro_f1": result["macro_f1"],
                    "epochs": result["epochs"],
                }], RAW_FIELDS)

                curve_rows = [
                    {"config_name": cname, "seed": seed, "epoch": e + 1,
                     "train_loss": result["train_losses"][e],
                     "val_loss": result["val_losses"][e]}
                    for e in range(len(result["train_losses"]))
                ]
                append_csv(curves_path, curve_rows, CURVES_FIELDS)

                print(f"  seed={seed}: val_acc={result['val_acc']:.4f}  epochs={result['epochs']}")

            raw_df = pd.read_csv(raw_path)
            s = summarize_group(raw_df, cname, extra_fields=["optimizer", "learning_rate"])
            summaries = [x for x in summaries if x["config_name"] != cname]
            summaries.append(s)
            pd.DataFrame(summaries, columns=SUMMARY_FIELDS).to_csv(summary_path, index=False)
            print(f"  → val_acc = {s['mean_val_acc']:.4f} ± {s['std_val_acc']:.4f}")

    print(f"\nResultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
