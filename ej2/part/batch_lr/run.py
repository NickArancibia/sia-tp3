"""Experimento 1: grid batch_size × learning_rate con Adam fijo.

Produce:
  results/part/batch_lr/raw.csv      — métricas finales por (config, seed)
  results/part/batch_lr/summary.csv  — media ± std por config
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

from _common import append_csv, batch_label, load_raw, split_scale, summarize_group, train
from shared.config_loader import load_config

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "batch_lr")
RAW_FIELDS = [
    "config_name", "batch_label", "batch_size", "learning_rate", "seed",
    "train_acc", "val_acc", "test_acc", "macro_f1", "epochs", "elapsed_s",
]
SUMMARY_FIELDS = [
    "config_name", "batch_label", "batch_size", "learning_rate",
    "mean_val_acc", "std_val_acc", "mean_test_acc", "std_test_acc",
    "mean_macro_f1", "std_macro_f1", "mean_epochs",
    "mean_elapsed_s", "std_elapsed_s",
]


def main():
    cfg = load_config(os.path.join(EJ2_DIR, "config.yaml"))
    search = cfg["search"]
    seeds = [int(s) for s in search["seeds"]]
    batch_sizes = [int(b) for b in search["batch_sizes"]]
    lrs = [float(lr) for lr in search["learning_rates"]]

    full_cfg_arch = cfg["model"]["architecture"]
    hidden = full_cfg_arch[1:-1]
    hidden_act = cfg["model"].get("hidden_activation", "tanh")
    max_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping"].get("patience", 20)
    adam_betas = cfg["training"].get("adam_betas", [0.9, 0.999])
    adam_eps = cfg["training"].get("adam_eps", 1e-8)

    X_all, y_all, X_test_raw, y_test, n_classes = load_raw(cfg, EJ2_DIR)
    full_arch = [X_all.shape[1]] + hidden + [n_classes]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, "raw.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")

    # Load already-computed results to skip them; clean up any duplicate header rows
    done_keys = set()
    if os.path.exists(raw_path):
        existing = pd.read_csv(raw_path)
        existing = existing[existing["batch_size"] != "batch_size"].reset_index(drop=True)
        existing.to_csv(raw_path, index=False)  # rewrite clean (single header)
        for _, row in existing.iterrows():
            done_keys.add((int(row["batch_size"]), float(row["learning_rate"]), int(row["seed"])))
        print(f"Retomando: {len(done_keys)} runs ya guardados, se saltean.")

    summaries = []
    for bs in batch_sizes:
        for lr in lrs:
            cname = f"{batch_label(bs)}_lr{lr:.0e}"
            print(f"\nbatch={batch_label(bs)}  lr={lr:.0e}")

            for seed in seeds:
                if (bs, lr, seed) in done_keys:
                    print(f"  seed={seed}: ya calculado, salteo.")
                    continue
                X_tr, y_tr, X_va, y_va, X_te = split_scale(X_all, y_all, X_test_raw, cfg, seed)
                opt_cfg = {
                    "optimizer": "gd",
                    "learning_rate": lr,
                }
                result = train(X_tr, y_tr, X_va, y_va, X_te, y_test, n_classes,
                               full_arch, hidden_act, opt_cfg, bs, seed, max_epochs, patience)
                append_csv(raw_path, [{
                    "config_name": cname,
                    "batch_label": batch_label(bs),
                    "batch_size": bs,
                    "learning_rate": lr,
                    "seed": seed,
                    "train_acc": result["train_acc"],
                    "val_acc": result["val_acc"],
                    "test_acc": result["test_acc"],
                    "macro_f1": result["macro_f1"],
                    "epochs": result["epochs"],
                    "elapsed_s": result["elapsed_s"],
                }], RAW_FIELDS)
                print(f"  seed={seed}: val_acc={result['val_acc']:.4f}  "
                      f"epochs={result['epochs']}  t={result['elapsed_s']:.1f}s")

            raw_df = pd.read_csv(raw_path)
            s = summarize_group(raw_df, cname, extra_fields=["batch_label", "batch_size", "learning_rate"])
            summaries = [x for x in summaries if x["config_name"] != cname]
            summaries.append(s)
            pd.DataFrame(summaries, columns=SUMMARY_FIELDS).to_csv(summary_path, index=False)
            print(f"  → val_acc = {s['mean_val_acc']:.4f} ± {s['std_val_acc']:.4f}")

    print(f"\nResultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
