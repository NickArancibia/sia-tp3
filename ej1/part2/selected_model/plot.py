import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import numpy as np
import pandas as pd

from plots import (plot_confusion_matrix, plot_internal_function, plot_learning_curves,
                   plot_pr, plot_threshold_sweep)
from shared.config_loader import load_config
from shared.metrics import auc, pr_curve


def batch_label(batch_size):
    batch_size = int(batch_size)
    if batch_size == -1:
        return "full"
    if batch_size == 1:
        return "online"
    return str(batch_size)


def format_lr(lr):
    if lr < 1e-3:
        mantissa, exp = f"{lr:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp = exp.replace("+0", "+").replace("-0", "-")
        return f"{mantissa}e{exp}"
    return f"{lr:.4f}".rstrip("0").rstrip(".")


def main():
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "selected_model")
    raw_df = pd.read_csv(os.path.join(out_dir, "selected_model_raw.csv"))
    summary = pd.read_csv(os.path.join(out_dir, "selected_model_summary.csv")).iloc[0]
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))

    selected_seed = int(summary["seed"])
    candidate_name = summary["candidate_name"]

    curve_rows = raw_df[(raw_df["record_type"] == "curve_point") & (raw_df["seed"].astype(int) == selected_seed)]
    train_runs = []
    val_runs = []
    for split_id in sorted(curve_rows["split_id"].dropna().unique()):
        split_rows = curve_rows[curve_rows["split_id"] == split_id].sort_values("epoch")
        train_runs.append(split_rows["train_mse"].astype(float).tolist())
        val_runs.append(split_rows["val_mse"].astype(float).tolist())

    plot_learning_curves(
        train_runs,
        val_runs,
        title=(
            f"Curvas del modelo seleccionado ({candidate_name}, lr={format_lr(float(summary['learning_rate']))}, "
            f"batch={batch_label(int(summary['batch_size']))}, seed={selected_seed})"
        ),
        path=os.path.join(out_dir, "selected_model_learning_curve.png"),
        zoom_tail=True,
    )

    threshold_rows = raw_df[raw_df["record_type"] == "threshold_point"].sort_values("threshold")
    plot_threshold_sweep(
        threshold_rows["threshold"].astype(float).to_numpy(),
        threshold_rows["val_precision"].astype(float).to_numpy(),
        threshold_rows["val_recall"].astype(float).to_numpy(),
        threshold_rows["val_f1"].astype(float).to_numpy(),
        float(summary["threshold"]),
        path=os.path.join(out_dir, "selected_model_threshold_sweep.png"),
        precisions_std=threshold_rows["generalization_gap_auc_pr"].astype(float).to_numpy(),
        recalls_std=threshold_rows["generalization_gap_f1"].astype(float).to_numpy(),
        f1s_std=threshold_rows["generalization_gap_mse"].astype(float).to_numpy(),
    )

    test_rows = raw_df[raw_df["record_type"] == "test_output"].sort_values("sample_idx")
    y_true = test_rows["y_true"].astype(int).to_numpy()
    scores = test_rows["score"].astype(float).to_numpy()
    precs, recs = pr_curve(y_true, scores)
    plot_pr(
        precs,
        recs,
        auc(recs, precs),
        path=os.path.join(out_dir, "selected_model_pr_curve.png"),
    )

    pred_optimal = test_rows["pred_optimal"].astype(int).to_numpy()
    cm = np.array([
        [int(np.sum((pred_optimal == 0) & (y_true == 0))), int(np.sum((pred_optimal == 1) & (y_true == 0)))],
        [int(np.sum((pred_optimal == 0) & (y_true == 1))), int(np.sum((pred_optimal == 1) & (y_true == 1)))],
    ])
    plot_confusion_matrix(
        cm,
        path=os.path.join(out_dir, "selected_model_confusion_matrix.png"),
    )

    plot_internal_function(
        test_rows["pre_activation"].astype(float).to_numpy(),
        test_rows["target_bigmodel"].astype(float).to_numpy(),
        cfg["model"]["activation"],
        beta=cfg["model"].get("beta", 1.0),
        path=os.path.join(out_dir, "selected_model_internal_function.png"),
    )

    conf_dir = os.path.join(out_dir, "confusion_by_threshold")
    os.makedirs(conf_dir, exist_ok=True)
    confusion_rows = raw_df[raw_df["record_type"] == "confusion_point"].sort_values("threshold")
    for _, row in confusion_rows.iterrows():
        cm_t = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]] )
        threshold_str = f"{float(row['threshold']):.3f}".replace(".", "_")
        plot_confusion_matrix(cm_t, path=os.path.join(conf_dir, f"threshold_{threshold_str}.png"))


if __name__ == "__main__":
    main()
