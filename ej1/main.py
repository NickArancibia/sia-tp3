import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shared.config_loader import load_config
import pandas as pd


def load_data(cfg):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, cfg["data"]["dataset_path"])
    df = pd.read_csv(path)

    print("=" * 60)
    print("EDA — Exploración del dataset")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumnas: {list(df.columns)}")
    print(f"\nValores faltantes:\n{df.isnull().sum().to_string()}")

    target_col = cfg["data"]["target_col"]
    label_col = cfg["data"]["label_col"]

    fraud_counts = df[label_col].value_counts().sort_index()
    print(f"\nDistribución de {label_col}:")
    for v, cnt in fraud_counts.items():
        print(f"  {v}: {cnt} ({cnt/len(df)*100:.2f}%)")

    print(f"\nEstadísticas de {target_col} (soft labels BigModel):")
    print(df[target_col].describe().round(4).to_string())

    df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s").dt.dayofweek
    df = df.drop(columns=["timestamp"])

    feature_cols = [c for c in df.columns if c not in (target_col, label_col)]

    print(f"\nFeatures usadas ({len(feature_cols)}): {feature_cols}")
    print("\nRangos de features:")
    stats = df[feature_cols].describe().loc[["min", "max", "mean", "std"]].round(3)
    print(stats.to_string())

    X = df[feature_cols].values.astype(float)
    t = df[target_col].values.astype(float)
    y = df[label_col].values.astype(int)

    return X, t, y, feature_cols


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, "config.yaml"))

    results_dir = os.path.join(base, cfg["experiment"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)

    from main_part1 import run_part1
    from main_part2 import run_part2

    run_part1(X, t, cfg, results_dir)
    run_part2(X, t, y, cfg, results_dir)