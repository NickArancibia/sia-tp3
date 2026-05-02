"""Experimento 4: comparación de funciones de activación para capas ocultas.

Usa la mejor arquitectura de architecture/summary.csv y la mejor config de
batch_lr/summary.csv con Adam.
Produce:
  results/part/activation/raw.csv      — métricas finales por (config, seed)
  results/part/activation/curves.csv   — pérdida por (config, seed, epoca)
  results/part/activation/summary.csv  — media ± std por config
"""
import os
import sys

EJ2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ2_DIR)
PART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EJ2_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PART_DIR)

import ast

import pandas as pd

from _common import (
    append_csv, load_raw, split_scale, summarize_group, train,
)
from shared.config_loader import load_config

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "activation")
OPTIMIZER_SUMMARY = os.path.join(EJ2_DIR, "results", "part", "optimizer", "summary.csv")
ARCH_SUMMARY = os.path.join(EJ2_DIR, "results", "part", "architecture", "summary.csv")

RAW_FIELDS = [
    "config_name", "hidden_activation", "architecture", "seed",
    "train_acc", "val_acc", "test_acc", "macro_f1", "epochs", "elapsed_s",
]
CURVES_FIELDS = ["config_name", "seed", "epoch", "train_loss", "val_loss"]
SUMMARY_FIELDS = [
    "config_name", "hidden_activation", "architecture",
    "mean_val_acc", "std_val_acc", "mean_test_acc", "std_test_acc",
    "mean_macro_f1", "std_macro_f1", "mean_epochs", "mean_elapsed_s", "std_elapsed_s",
]


def _best_arch(arch_summary_df, n_features, n_classes):
    """Devuelve la mejor arquitectura completa del experimento de arquitecturas."""
    if arch_summary_df is None:
        return [n_features, 128, 64, n_classes]
    best_row = arch_summary_df.loc[arch_summary_df["mean_test_acc"].idxmax()]
    return ast.literal_eval(best_row["architecture"])


def main():
    cfg = load_config(os.path.join(EJ2_DIR, "config.yaml"))
    search = cfg["search"]
    seeds = [int(s) for s in search["seeds"]]
    activations = search["hidden_activations"]

    if not os.path.exists(OPTIMIZER_SUMMARY):
        print(f"ERROR: ejecutar optimizer/run.py primero (falta {OPTIMIZER_SUMMARY})")
        return

    opt_summary = pd.read_csv(OPTIMIZER_SUMMARY)
    adam_rows = opt_summary[opt_summary["optimizer"] == "adam"]
    lr = float(adam_rows.loc[adam_rows["mean_val_acc"].idxmax(), "learning_rate"])
    bs = 32

    arch_summary = pd.read_csv(ARCH_SUMMARY) if os.path.exists(ARCH_SUMMARY) else None

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
    full_arch = _best_arch(arch_summary, X_all.shape[1], n_classes)

    print(f"Arquitectura: {full_arch}")
    print(f"Optimizer: Adam, batch_size={bs}, lr={lr:.0e}")
    if arch_summary is None:
        print("(AVISO: architecture/summary.csv no encontrado, usando arquitectura por defecto)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, "raw.csv")
    curves_path = os.path.join(RESULTS_DIR, "curves.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    for p in (raw_path, curves_path, summary_path):
        if os.path.exists(p):
            os.remove(p)

    summaries = []
    for act in activations:
        cname = act
        print(f"\n{'='*50}")
        print(f"Activación oculta: {act}")
        print(f"{'='*50}")

        for seed in seeds:
            X_tr, y_tr, X_va, y_va, X_te = split_scale(X_all, y_all, X_test_raw, cfg, seed)
            result = train(X_tr, y_tr, X_va, y_va, X_te, y_test, n_classes,
                           full_arch, act, opt_cfg, bs, seed, max_epochs, patience)

            append_csv(raw_path, [{
                "config_name": cname,
                "hidden_activation": act,
                "architecture": str(full_arch),
                "seed": seed,
                "train_acc": result["train_acc"],
                "val_acc": result["val_acc"],
                "test_acc": result["test_acc"],
                "macro_f1": result["macro_f1"],
                "epochs": result["epochs"],
                "elapsed_s": result["elapsed_s"],
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
        s = summarize_group(raw_df, cname,
                            extra_fields=["hidden_activation", "architecture"])
        summaries = [x for x in summaries if x["config_name"] != cname]
        summaries.append(s)
        pd.DataFrame(summaries, columns=SUMMARY_FIELDS).to_csv(summary_path, index=False)
        print(f"  → val_acc = {s['mean_val_acc']:.4f} ± {s['std_val_acc']:.4f}")

    print(f"\nResultados guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
