import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shared.config_loader import load_config
import pandas as pd


def load_data(cfg):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, cfg["data"]["dataset_path"])
    df = pd.read_csv(path)

    target_col = cfg["data"]["target_col"]
    label_col = cfg["data"]["label_col"]

    fraud_counts = df[label_col].value_counts().sort_index()
    missing_total = int(df.isnull().sum().sum())
    fraud_summary = ", ".join(
        f"{v}: {cnt} ({cnt/len(df)*100:.2f}%)"
        for v, cnt in fraud_counts.items()
    )

    df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s").dt.dayofweek
    df = df.drop(columns=["timestamp"])

    feature_cols = [c for c in df.columns if c not in (target_col, label_col)]
    print(
        f"Dataset: {df.shape[0]} filas, {len(feature_cols)} features | "
        f"faltantes={missing_total} | {label_col}: {fraud_summary}"
    )

    X = df[feature_cols].values.astype(float)
    t = df[target_col].values.astype(float)
    y = df[label_col].values.astype(int)

    return X, t, y, feature_cols


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, "config.yaml"))

    results_dir = os.path.join(base, cfg["experiment"]["results_dir"])
    part1_results_dir = os.path.join(results_dir, "part1")
    part2_results_dir = os.path.join(results_dir, "part2")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(part1_results_dir, exist_ok=True)
    os.makedirs(part2_results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)

    from main_part1 import run_part1
    from main_part2 import run_part2

    run_part1(X, t, cfg, part1_results_dir)
    run_part2(X, t, y, cfg, part2_results_dir)
