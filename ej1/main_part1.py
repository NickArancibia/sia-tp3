import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from shared.config_loader import load_config
from shared.optimizers import build_optimizer
from shared.perceptron import SimplePerceptron
from shared.preprocessing import build_scaler
from plots import plot_multi_learning_curves


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


def run_part1(X, t, cfg, results_dir):
    print("\n" + "=" * 60)
    print("PARTE 1 — Análisis de aprendizaje (todos los datos)")
    print("=" * 60)

    seeds = cfg["experiment"].get("seeds", [cfg["experiment"]["seed"]])
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"].get("batch_size", 32)
    shuffle = cfg["training"].get("shuffle", True)
    beta = cfg["model"].get("beta", 1.0)
    init_scale = cfg["model"].get("init_scale", 0.1)
    initializer = cfg["model"].get("initializer", "random_normal")

    scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
    X_scaled = scaler.fit_transform(X) if scaler else X.copy()

    model_specs = [
        ("identity", "Lineal (identidad)"),
        (cfg["model"]["activation"], f"No Lineal ({cfg['model']['activation']})"),
    ]

    all_runs = {label: [] for _, label in model_specs}

    for activation, label in model_specs:
        print(f"\n--- Perceptrón {label} ({len(seeds)} seeds) ---")
        for seed in seeds:
            rng_s = np.random.default_rng(seed)
            opt = build_optimizer(cfg["training"])
            p = SimplePerceptron(
                n_inputs=X_scaled.shape[1],
                activation=activation,
                beta=beta,
                initializer=initializer,
                init_scale=init_scale,
                seed=seed,
                weight_decay=cfg["training"].get("weight_decay", 0.0),
            )
            epoch_losses = []
            for epoch in range(1, epochs + 1):
                loss, _ = p.train_epoch(X_scaled, t, opt,
                                        batch_size=batch_size,
                                        shuffle=shuffle, rng=rng_s)
                epoch_losses.append(loss)
            all_runs[label].append(epoch_losses)
            print(f"  seed={seed} | MSE final: {epoch_losses[-1]:.6f}")

        finals = [runs[-1] for runs in all_runs[label]]
        print(f"  → media ± std: {np.mean(finals):.6f} ± {np.std(finals):.6f}")

    plot_multi_learning_curves(
        all_runs,
        title=f"Parte 1 — Lineal vs No Lineal ({len(seeds)} seeds, media ± std)",
        path=os.path.join(results_dir, "part1_learning_curves.png"),
    )
    print(f"\n[Gráfico guardado en results/part1_learning_curves.png]")

    print("\nResumen Parte 1:")
    for _, label in model_specs:
        finals = [runs[-1] for runs in all_runs[label]]
        print(f"  {label}: MSE = {np.mean(finals):.6f} ± {np.std(finals):.6f}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, "config.yaml"))

    results_dir = os.path.join(base, cfg["experiment"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)
    run_part1(X, t, cfg, results_dir)