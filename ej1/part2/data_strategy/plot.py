import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd

from plots import plot_metric_bars, plot_strategy_overfitting_curves


def remove_stale_plots(out_dir, names):
    for name in names:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)


def main():
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "data_strategy")
    raw_path = os.path.join(out_dir, "strategy_raw.csv")
    summary_path = os.path.join(out_dir, "strategy_summary.csv")

    remove_stale_plots(
        out_dir,
        [
            "data_strategy_aucpr.png",
            "data_strategy_precision_recall.png",
        ],
    )

    raw_df = pd.read_csv(raw_path)
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[summary_df["strategy"].isin(["S1", "S3"])].copy()

    plot_metric_bars(
        summary_df["strategy_label"].tolist(),
        summary_df["mean_val_f2"].to_numpy(),
        summary_df["std_val_f2"].to_numpy(),
        ylabel="F2 en validación",
        title="Comparación de estrategias de datos por F2",
        path=os.path.join(out_dir, "data_strategy_f2.png"),
    )

    plot_metric_bars(
        summary_df["strategy_label"].tolist(),
        summary_df["mean_elapsed_s"].to_numpy(),
        summary_df["std_elapsed_s"].to_numpy(),
        ylabel="Tiempo total [s]",
        title="Tiempo total por estrategia de datos",
        path=os.path.join(out_dir, "data_strategy_time.png"),
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
        grouped = rows.groupby(["seed", "split_kind", "split_id"], sort=True)
        for _, split_rows in grouped:
            split_rows = split_rows.sort_values("epoch")
            train_runs.append(split_rows["train_mse"].astype(float).tolist())
            val_runs.append(split_rows["val_mse"].astype(float).tolist())
        strategy_curves[summary["strategy_label"]] = {"train": train_runs, "val": val_runs}

    plot_strategy_overfitting_curves(
        strategy_curves,
        path=os.path.join(out_dir, "data_strategy_overfitting_curves.png"),
        zoom_tail=False,
        show_std=False,
        sharey=True,
        y_limits=(0.0, 0.06),
    )


if __name__ == "__main__":
    main()
