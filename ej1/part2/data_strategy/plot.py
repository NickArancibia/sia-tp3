import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd

from plots import plot_grouped_metric_bars, plot_metric_bars, plot_strategy_overfitting_curves


def main():
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "data_strategy")
    raw_path = os.path.join(out_dir, "strategy_raw.csv")
    summary_path = os.path.join(out_dir, "strategy_summary.csv")

    raw_df = pd.read_csv(raw_path)
    summary_df = pd.read_csv(summary_path)

    plot_metric_bars(
        summary_df["strategy_label"].tolist(),
        summary_df["mean_val_auc_pr"].to_numpy(),
        summary_df["std_val_auc_pr"].to_numpy(),
        ylabel="AUC-PR en validación",
        title="Comparación de estrategias de datos",
        path=os.path.join(out_dir, "data_strategy_aucpr.png"),
    )

    plot_grouped_metric_bars(
        summary_df["strategy_label"].tolist(),
        {
            "Precision": {
                "means": summary_df["mean_val_precision"].to_numpy(),
                "stds": summary_df["std_val_precision"].to_numpy(),
            },
            "Recall": {
                "means": summary_df["mean_val_recall"].to_numpy(),
                "stds": summary_df["std_val_recall"].to_numpy(),
            },
        },
        ylabel="Score en validación",
        title="Precision y Recall por estrategia de datos",
        path=os.path.join(out_dir, "data_strategy_precision_recall.png"),
    )

    strategy_curves = {}
    for _, summary in summary_df.iterrows():
        rows = raw_df[
            (raw_df["record_type"] == "curve_point")
            & (raw_df["strategy"] == summary["strategy"])
            & (raw_df["split_kind"] != "final_retrain")
        ]
        train_runs = []
        val_runs = []
        for split_id in sorted(rows["split_id"].dropna().unique()):
            split_rows = rows[rows["split_id"] == split_id].sort_values("epoch")
            train_runs.append(split_rows["train_mse"].astype(float).tolist())
            val_runs.append(split_rows["val_mse"].astype(float).tolist())
        strategy_curves[summary["strategy_label"]] = {"train": train_runs, "val": val_runs}

    plot_strategy_overfitting_curves(
        strategy_curves,
        path=os.path.join(out_dir, "data_strategy_overfitting_curves.png"),
        zoom_tail=True,
        show_std=False,
    )


if __name__ == "__main__":
    main()
