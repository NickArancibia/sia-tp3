import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from shared.activations import activate
from shared.config_loader import load_config
from shared.losses import mse
from shared.optimizers import build_optimizer
from shared.perceptron import SimplePerceptron
from shared.preprocessing import build_scaler
from plots import (plot_internal_function,
                   plot_learning_curve_comparison,
                   plot_target_vs_prediction)


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


def _representative_run(runs):
    mean_final_mse = float(np.mean([run["final_mse"] for run in runs]))
    rep_idx = int(np.argmin([abs(run["final_mse"] - mean_final_mse) for run in runs]))
    return runs[rep_idx]


def _slugify_activation(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _autoscale_limits(values, margin_ratio=0.05, min_span=1e-6):
    arr = np.asarray(values, dtype=float)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = max(hi - lo, min_span)
    margin = span * margin_ratio
    return lo - margin, hi + margin


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
        ("relu", "No Lineal (ReLU)"),
    ]

    model_runs = {label: [] for _, label in model_specs}

    for activation, label in model_specs:
        print(f"Entrenando perceptrón {label} ({len(seeds)} seeds)...")
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
            predictions = p.predict(X_scaled)
            pre_activations = p.pre_activation(X_scaled)
            final_mse = float(mse(t, predictions))
            model_runs[label].append({
                "seed": seed,
                "activation": activation,
                "losses": epoch_losses,
                "predictions": predictions,
                "pre_activations": pre_activations,
                "final_mse": final_mse,
            })

        finals = [run["final_mse"] for run in model_runs[label]]
        print(f"  {label}: MSE final={np.mean(finals):.6f} ± {np.std(finals):.6f}")

    learning_runs = {
        label: [run["losses"] for run in runs]
        for label, runs in model_runs.items()
    }
    plot_learning_curve_comparison(
        learning_runs,
        title=f"Parte 1 — Lineal vs no lineal ({len(seeds)} seeds, media ± std)",
        path=os.path.join(results_dir, "learning_curve_linear_nonlinear_relu.png"),
        y_max=0.06,
    )

    rep_runs = []
    for activation, label in model_specs:
        rep = _representative_run(model_runs[label])
        rep_runs.append({
            "label": label,
            "activation": activation,
            "predictions": rep["predictions"],
            "pre_activations": rep["pre_activations"],
        })

    all_predictions = np.concatenate([np.asarray(run["predictions"]) for run in rep_runs])
    prediction_lo = float(min(np.min(t), np.min(all_predictions)))
    prediction_hi = float(max(np.max(t), np.max(all_predictions)))

    rep_by_activation = {run["activation"]: run for run in rep_runs}
    base_activation = cfg["model"]["activation"]
    identity_run = rep_by_activation["identity"]
    base_nonlinear_run = rep_by_activation[base_activation]

    linear_x_limits = _autoscale_limits(identity_run["pre_activations"])
    nonlinear_x_limits = _autoscale_limits(base_nonlinear_run["pre_activations"])
    legacy_y_values = np.concatenate([
        np.asarray(t, dtype=float),
        activate(np.asarray(identity_run["pre_activations"]), "identity", beta),
        activate(np.asarray(base_nonlinear_run["pre_activations"]), base_activation, beta),
    ])
    internal_y_limits = _autoscale_limits(legacy_y_values)

    for run in rep_runs:
        activation = run["activation"]
        slug = _slugify_activation(activation)
        x_limits = nonlinear_x_limits if activation == base_activation else linear_x_limits
        plot_internal_function(
            run["pre_activations"],
            t,
            activation=activation,
            beta=beta,
            x_limits=x_limits,
            y_limits=internal_y_limits,
            path=os.path.join(results_dir, f"internal_function_{slug}.png"),
        )
        plot_target_vs_prediction(
            t,
            run["predictions"],
            axis_limits=(prediction_lo, prediction_hi),
            path=os.path.join(results_dir, f"target_vs_prediction_{slug}.png"),
        )
    print("Gráficos guardados en results/part1/")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, "config.yaml"))

    results_dir = os.path.join(base, cfg["experiment"]["results_dir"], "part1")
    os.makedirs(results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)
    run_part1(X, t, cfg, results_dir)
