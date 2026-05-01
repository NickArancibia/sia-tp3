import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd

from plots import plot_metric_bars


def format_lr(lr):
    if lr < 1e-3:
        mantissa, exp = f"{lr:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp = exp.replace("+0", "+").replace("-0", "-")
        return f"{mantissa}e{exp}"
    return f"{lr:.4f}".rstrip("0").rstrip(".")


def main():
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "optimizer")
    summary_df = pd.read_csv(os.path.join(out_dir, "optimizer_summary.csv"))

    labels = [
        f"{row['candidate_name']}\n{row['batch_label']}\nlr={format_lr(float(row['learning_rate']))}"
        for _, row in summary_df.iterrows()
    ]
    plot_metric_bars(
        labels,
        summary_df["mean_val_auc_pr"].to_numpy(),
        summary_df["std_val_auc_pr"].to_numpy(),
        ylabel="AUC-PR en validación",
        title="Comparación final de métodos de optimización",
        path=os.path.join(out_dir, "optimizer_comparison_aucpr.png"),
    )


if __name__ == "__main__":
    main()
