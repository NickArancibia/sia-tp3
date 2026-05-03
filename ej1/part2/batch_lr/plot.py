import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd

from plots import plot_heatmap, plot_strategy_overfitting_curves


def format_lr(lr):
    if lr < 1e-3:
        mantissa, exp = f"{lr:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp = exp.replace("+0", "+").replace("-0", "-")
        return f"{mantissa}e{exp}"
    return f"{lr:.4f}".rstrip("0").rstrip(".")


def batch_label(batch_size):
    batch_size = int(batch_size)
    if batch_size == -1:
        return "full"
    if batch_size == 1:
        return "online"
    return str(batch_size)


def main():
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "batch_lr")
    raw_df = pd.read_csv(os.path.join(out_dir, "batch_lr_raw.csv"))
    summary_df = pd.read_csv(os.path.join(out_dir, "batch_lr_summary.csv"))

    batch_order = sorted(summary_df["batch_size"].astype(int).unique(), key=lambda value: (value == -1, value))
    if -1 in batch_order:
        batch_order = [value for value in batch_order if value != -1] + [-1]
    lr_order = sorted(summary_df["learning_rate"].astype(float).unique(), reverse=True)
    row_labels = [batch_label(batch) for batch in batch_order]
    col_labels = [format_lr(lr) for lr in lr_order]

    matrix = []
    annotations = []
    for batch_size in batch_order:
        row = []
        ann_row = []
        for learning_rate in lr_order:
            cell = summary_df[
                (summary_df["batch_size"].astype(int) == batch_size)
                & (summary_df["learning_rate"].astype(float).round(12) == round(learning_rate, 12))
            ].iloc[0]
            row.append(float(cell["mean_val_auc_pr"]))
            ann_row.append(f"{float(cell['mean_val_auc_pr']):.4f}\n±{float(cell['std_val_auc_pr']):.4f}")
        matrix.append(row)
        annotations.append(ann_row)

    plot_heatmap(
        matrix,
        row_labels,
        col_labels,
        title="Heatmap batch size + learning rate (AUC-PR)",
        cbar_label="AUC-PR en validación",
        path=os.path.join(out_dir, "batch_lr_heatmap_aucpr.png"),
        annotations=annotations,
    )

    best_per_batch = []
    for batch_size in batch_order:
        rows = summary_df[summary_df["batch_size"].astype(int) == batch_size].copy()
        rows = rows.sort_values(["mean_val_auc_pr", "std_val_auc_pr", "mean_val_recall"], ascending=[False, True, False])
        best_per_batch.append(rows.iloc[0])

    curve_dict = {}
    for row in best_per_batch:
        config_name = row["config_name"]
        learning_rate = float(row["learning_rate"])
        batch_size = int(row["batch_size"])
        label = f"{config_name}\nb={batch_label(batch_size)}, lr={format_lr(learning_rate)}"
        rows = raw_df[
            (raw_df["record_type"] == "curve_point")
            & (raw_df["config_name"] == config_name)
            & (raw_df["batch_size"].astype(int) == batch_size)
            & (raw_df["learning_rate"].astype(float).round(12) == round(learning_rate, 12))
            & (raw_df["split_kind"] != "final_retrain")
        ]
        train_runs = []
        val_runs = []
        for split_id in sorted(rows["split_id"].dropna().unique()):
            split_rows = rows[rows["split_id"] == split_id].sort_values("epoch")
            train_runs.append(split_rows["train_mse"].astype(float).tolist())
            val_runs.append(split_rows["val_mse"].astype(float).tolist())
        curve_dict[label] = {"train": train_runs, "val": val_runs}

    plot_strategy_overfitting_curves(
        curve_dict,
        path=os.path.join(out_dir, "batch_lr_overfitting_curves.png"),
        title="Overfitting por combinación batch size + learning rate",
        zoom_tail=False,
        show_std=True,
    )


if __name__ == "__main__":
    main()
