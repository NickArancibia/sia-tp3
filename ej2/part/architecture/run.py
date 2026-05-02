"""Experimento 3: comparación de arquitecturas (profundidad y anchura).

Usa el mejor (batch_size, LR) global de batch_lr/summary.csv con Adam.
Produce:
  results/part/architecture/raw.csv      — métricas finales por (config, seed)
  results/part/architecture/summary.csv  — media ± std por config
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

from _common import (
    append_csv, load_raw, split_scale, summarize_group, train,
)
from shared.config_loader import load_config

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "architecture")
OPTIMIZER_SUMMARY = os.path.join(EJ2_DIR, "results", "part", "optimizer", "summary.csv")

RAW_FIELDS = [
    "config_name", "architecture", "n_params", "seed",
    "train_acc", "val_acc", "test_acc", "macro_f1", "epochs", "elapsed_s",
]
SUMMARY_FIELDS = [
    "config_name", "architecture", "n_params",
    "mean_val_acc", "std_val_acc", "mean_test_acc", "std_test_acc",
    "mean_macro_f1", "std_macro_f1", "mean_epochs", "mean_elapsed_s", "std_elapsed_s",
]


def _count_params(arch):
    total = 0
    for i in range(len(arch) - 1):
        total += arch[i] * arch[i + 1] + arch[i + 1]
    return total


def main():
    cfg = load_config(os.path.join(EJ2_DIR, "config.yaml"))
    search = cfg["search"]
    seeds = [int(s) for s in search["seeds"]]
    arch_list = search["architectures"]

    if not os.path.exists(OPTIMIZER_SUMMARY):
        print(f"ERROR: ejecutar optimizer/run.py primero (falta {OPTIMIZER_SUMMARY})")
        return
    opt_summary = pd.read_csv(OPTIMIZER_SUMMARY)

    # Mejor LR para Adam según optimizer/summary.csv
    adam_rows = opt_summary[opt_summary["optimizer"] == "adam"]
    lr = float(adam_rows.loc[adam_rows["mean_val_acc"].idxmax(), "learning_rate"])
    bs = 32
    hidden_act = cfg["model"].get("hidden_activation", "tanh")
    max_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping"].get("patience", 20)
    adam_betas = cfg["training"].get("adam_betas", [0.9, 0.999])
    adam_eps = cfg["training"].get("adam_eps", 1e-8)

    opt_cfg = {
        "optimizer": "adam",
        "learning_rate": lr,
        "adam_betas": adam_betas,
        "adam_eps": adam_eps,
    }

    X_all, y_all, X_test_raw, y_test, n_classes = load_raw(cfg, EJ2_DIR)
    n_features = X_all.shape[1]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, "raw.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    for p in (raw_path, summary_path):
        if os.path.exists(p):
            os.remove(p)

    print(f"Optimizer: Adam, batch_size={bs}, lr={lr:.0e}")

    summaries = []
    for arch_spec in arch_list:
        cname = arch_spec["name"]
        hidden = [int(h) for h in arch_spec["hidden"]]
        full_arch = [n_features] + hidden + [n_classes]
        n_params = _count_params(full_arch)

        print(f"\n{'='*50}")
        print(f"{cname}: {full_arch}  ({n_params:,} params)")
        print(f"{'='*50}")

        for seed in seeds:
            X_tr, y_tr, X_va, y_va, X_te = split_scale(X_all, y_all, X_test_raw, cfg, seed)
            result = train(X_tr, y_tr, X_va, y_va, X_te, y_test, n_classes,
                           full_arch, hidden_act, opt_cfg, bs, seed, max_epochs, patience)

            append_csv(raw_path, [{
                "config_name": cname,
                "architecture": str(full_arch),
                "n_params": n_params,
                "seed": seed,
                "train_acc": result["train_acc"],
                "val_acc": result["val_acc"],
                "test_acc": result["test_acc"],
                "macro_f1": result["macro_f1"],
                "epochs": result["epochs"],
                "elapsed_s": result["elapsed_s"],
            }], RAW_FIELDS)
            print(f"  seed={seed}: val_acc={result['val_acc']:.4f}  epochs={result['epochs']}")

        raw_df = pd.read_csv(raw_path)
        s = summarize_group(raw_df, cname, extra_fields=["architecture", "n_params"])
        summaries = [x for x in summaries if x["config_name"] != cname]
        summaries.append(s)
        pd.DataFrame(summaries, columns=SUMMARY_FIELDS).to_csv(summary_path, index=False)
        print(f"  → val_acc = {s['mean_val_acc']:.4f} ± {s['std_val_acc']:.4f}")

    print(f"\nResultados guardados en {RESULTS_DIR}")
    summary_df = pd.read_csv(summary_path).sort_values("mean_test_acc", ascending=False)
    print("\nTop 3 arquitecturas (test accuracy):")
    for _, row in summary_df.head(3).iterrows():
        print(f"  {row['config_name']:15s} {row['mean_test_acc']:.4f} ± {row['std_test_acc']:.4f}"
              f"  ({int(row['n_params']):,} params)")


if __name__ == "__main__":
    main()
