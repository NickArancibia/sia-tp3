import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PART2_DIR)

import numpy as np
import pandas as pd

from common import read_raw_csv
from plots import (
    plot_confusion_matrix,
    plot_cost_threshold_sweep,
    plot_internal_function,
    plot_strategy_overfitting_curves,
)
from run import fixed_threshold_grid
from shared.config_loader import load_config


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


def remove_stale_outputs(out_dir):
    for name in ("selected_model_pr_curve.png", "selected_model_confusion_matrix.png"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)

    conf_dir = os.path.join(out_dir, "confusion_by_threshold")
    os.makedirs(conf_dir, exist_ok=True)
    for name in os.listdir(conf_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(conf_dir, name))
    return conf_dir


def sample_rows(raw_df, seed, subset):
    rows = raw_df[
        (raw_df["record_type"] == "sample_output")
        & (raw_df["seed"].astype(int) == int(seed))
        & (raw_df["subset"] == subset)
    ].copy()
    if rows.empty:
        return rows
    rows["sample_idx_int"] = rows["sample_idx"].astype(int)
    return rows.sort_values(["split_kind", "split_id", "sample_idx_int"])


def curve_runs(raw_df, seed):
    rows = raw_df[
        (raw_df["record_type"] == "curve_point")
        & (raw_df["seed"].astype(int) == int(seed))
        & (raw_df["split_kind"] != "final_retrain")
    ].copy()
    if rows.empty:
        return [], []

    rows["epoch_int"] = rows["epoch"].astype(int)
    train_runs = []
    val_runs = []
    for _, split_rows in rows.groupby(["split_kind", "split_id"], sort=True):
        split_rows = split_rows.sort_values("epoch_int")
        train_runs.append(split_rows["train_mse"].astype(float).tolist())
        val_runs.append(split_rows["val_mse"].astype(float).tolist())
    return train_runs, val_runs


def aggregate_threshold_costs(val_rows):
    thresholds = fixed_threshold_grid()
    cost_runs = []

    for _, split_rows in val_rows.groupby(["split_kind", "split_id"], sort=True):
        y_true = split_rows["y_true"].astype(int).to_numpy()
        scores = split_rows["score"].astype(float).to_numpy()
        costs = []
        for threshold in thresholds:
            pred = (scores >= float(threshold)).astype(int)
            cm = binary_confusion_matrix(y_true, pred)
            fn = int(cm[1, 0])
            fp = int(cm[0, 1])
            costs.append(float(2 * fn + fp))
        cost_runs.append(costs)

    cost_arr = np.asarray(cost_runs, dtype=float)
    return {
        "thresholds": thresholds,
        "costs_mean": cost_arr.mean(axis=0),
        "costs_std": cost_arr.std(axis=0),
    }


def binary_confusion_matrix(y_true, pred):
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    return np.array([
        [int(np.sum((pred == 0) & (y_true == 0))), int(np.sum((pred == 1) & (y_true == 0)))],
        [int(np.sum((pred == 0) & (y_true == 1))), int(np.sum((pred == 1) & (y_true == 1)))],
    ])


def main():
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "selected_model")
    conf_dir = remove_stale_outputs(out_dir)
    raw_df = read_raw_csv(os.path.join(out_dir, "selected_model_raw.csv"))
    summary = pd.read_csv(os.path.join(out_dir, "selected_model_summary.csv")).iloc[0]
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))

    selected_seed = int(summary["seed"])
    optimizer_name = str(summary["optimizer"])

    train_runs, val_runs = curve_runs(raw_df, selected_seed)
    plot_strategy_overfitting_curves(
        {
            optimizer_name: {
                "train": train_runs,
                "val": val_runs,
            }
        },
        title=(
            f"Curvas del modelo seleccionado ({optimizer_name}, lr={format_lr(float(summary['learning_rate']))}, "
            f"batch={batch_label(int(summary['batch_size']))}, seed={selected_seed})"
        ),
        path=os.path.join(out_dir, "selected_model_learning_curve.png"),
        zoom_tail=False,
        show_std=False,
        sharey=True,
        y_limits=(0.0, 0.06),
    )

    val_rows = sample_rows(raw_df, selected_seed, "val")
    threshold_sweep = aggregate_threshold_costs(val_rows)
    plot_cost_threshold_sweep(
        threshold_sweep["thresholds"],
        threshold_sweep["costs_mean"],
        float(summary["threshold"]),
        path=os.path.join(out_dir, "selected_model_threshold_sweep.png"),
        costs_std=threshold_sweep["costs_std"],
        best_label="Umbral seleccionado",
    )

    test_rows = sample_rows(raw_df, selected_seed, "test")
    y_true = test_rows["y_true"].astype(int).to_numpy()
    scores = test_rows["score"].astype(float).to_numpy()

    plot_internal_function(
        test_rows["pre_activation"].astype(float).to_numpy(),
        test_rows["target_bigmodel"].astype(float).to_numpy(),
        cfg["model"]["activation"],
        beta=cfg["model"].get("beta", 1.0),
        path=os.path.join(out_dir, "selected_model_internal_function.png"),
    )

    for threshold in fixed_threshold_grid():
        pred_t = (scores >= float(threshold)).astype(int)
        cm_t = binary_confusion_matrix(y_true, pred_t)
        threshold_str = f"{float(threshold):.3f}".replace(".", "_")
        plot_confusion_matrix(cm_t, path=os.path.join(conf_dir, f"threshold_{threshold_str}.png"))


if __name__ == "__main__":
    main()
