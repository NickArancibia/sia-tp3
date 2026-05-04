import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd

from plots import plot_grouped_metric_bars, plot_heatmap


def remove_stale_plots(out_dir, names):
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


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


def heatmap_batch_label(batch_size):
    batch_size = int(batch_size)
    if batch_size > 1:
        return f"mini{batch_size}"
    return batch_label(batch_size)


def main():
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "batch_lr")
    remove_stale_plots(out_dir, ["batch_lr_overfitting_curves.png", "batch_lr_heatmap_aucpr.png"])

    summary_df = pd.read_csv(os.path.join(out_dir, "batch_lr_summary.csv"))

    batch_order = sorted(summary_df["batch_size"].astype(int).unique(), key=lambda value: (value == -1, value))
    if -1 in batch_order:
        batch_order = [value for value in batch_order if value != -1] + [-1]
    lr_order = sorted(summary_df["learning_rate"].astype(float).unique(), reverse=True)
    row_labels = [batch_label(batch) for batch in batch_order]
    heatmap_row_labels = [heatmap_batch_label(batch) for batch in batch_order]
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
            row.append(float(cell["mean_val_f2"]))
            ann_row.append(f"{float(cell['mean_val_f2']):.4f}\n±{float(cell['std_val_f2']):.4f}")
        matrix.append(row)
        annotations.append(ann_row)

    plot_heatmap(
        matrix,
        heatmap_row_labels,
        col_labels,
        title="Heatmap batch size + learning rate (F2)",
        cbar_label="F2 en validación",
        path=os.path.join(out_dir, "batch_lr_heatmap_f2.png"),
        annotations=annotations,
        cmap="RdYlGn",
    )

    time_series = {}
    for learning_rate, col_label in zip(lr_order, col_labels):
        means = []
        stds = []
        for batch_size in batch_order:
            cell = summary_df[
                (summary_df["batch_size"].astype(int) == batch_size)
                & (summary_df["learning_rate"].astype(float).round(12) == round(learning_rate, 12))
            ].iloc[0]
            means.append(float(cell["mean_elapsed_s"]))
            stds.append(float(cell["std_elapsed_s"]))
        time_series[f"lr={col_label}"] = {"means": means, "stds": stds}

    plot_grouped_metric_bars(
        row_labels,
        time_series,
        ylabel="Tiempo total [s]",
        title="Tiempo total por batch size y learning rate (escala log)",
        path=os.path.join(out_dir, "batch_lr_time.png"),
        yscale="log",
        annotation_fontsize=5,
    )

if __name__ == "__main__":
    main()
