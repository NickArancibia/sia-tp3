"""Bar chart + curvas de convergencia por función de activación oculta."""
import os
import sys

EJ2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ2_DIR)
sys.path.insert(0, EJ2_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd

from shared.utils import plot_multi_bar, plot_multi_learning_curves

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "activation")


def _load_curves_per_config(curves_df):
    result = {}
    for cname, grp in curves_df.groupby("config_name"):
        per_seed = []
        for _, sgrp in grp.groupby("seed"):
            per_seed.append(sgrp.sort_values("epoch")["val_loss"].tolist())
        result[cname] = per_seed
    return result


def main():
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    curves_path = os.path.join(RESULTS_DIR, "curves.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: ejecutar run.py primero (falta {summary_path})")
        return

    summary_df = pd.read_csv(summary_path)

    acc_data = {
        row["config_name"]: (row["mean_test_acc"], row["std_test_acc"])
        for _, row in summary_df.iterrows()
    }
    plot_multi_bar(
        acc_data,
        title="Test accuracy por activación oculta (media ± std, 5 semillas)",
        ylabel="Test accuracy",
        path=os.path.join(RESULTS_DIR, "test_acc_bar.png"),
    )

    if os.path.exists(curves_path):
        curves_df = pd.read_csv(curves_path)
        curves_per_config = _load_curves_per_config(curves_df)
        labels = summary_df["config_name"].tolist()
        ordered = {c: curves_per_config[c] for c in labels if c in curves_per_config}
        plot_multi_learning_curves(
            ordered,
            title="Convergencia por activación oculta",
            ylabel="Val MSE",
            path=os.path.join(RESULTS_DIR, "convergence_curves.png"),
        )

    best = summary_df.loc[summary_df["mean_test_acc"].idxmax()]
    print(f"\nMejor activación: {best['config_name']}")
    print(f"  test_acc = {best['mean_test_acc']:.4f} ± {best['std_test_acc']:.4f}")
    print(f"Gráficos guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
