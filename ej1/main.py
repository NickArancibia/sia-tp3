import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from shared.config_loader import load_config
from shared.losses import mse
from shared.metrics import (auc, confusion_matrix, pr_curve, precision_recall_f1,
                             roc_curve, threshold_sweep)
from shared.optimizers import build_optimizer
from shared.perceptron import SimplePerceptron
from shared.preprocessing import build_scaler, stratified_split
from shared.regularization import EarlyStopping
import shared.utils as utils


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

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

    # Feature engineering on timestamp
    df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s").dt.dayofweek
    df = df.drop(columns=["timestamp"])

    feature_cols = [c for c in df.columns if c not in (target_col, label_col)]

    print(f"\nFeatures usadas ({len(feature_cols)}): {feature_cols}")
    print("\nRangos de features:")
    stats = df[feature_cols].describe().loc[["min", "max", "mean", "std"]].round(3)
    print(stats.to_string())

    X = df[feature_cols].values.astype(float)
    t = df[target_col].values.astype(float)   # soft labels — training target
    y = df[label_col].values.astype(int)       # hard labels — business evaluation

    return X, t, y, feature_cols


# ---------------------------------------------------------------------------
# Parte 1 — Análisis de aprendizaje (lineal vs no lineal, todos los datos)
# ---------------------------------------------------------------------------

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

    # Scaler fit on all data (no randomness here)
    scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
    X_scaled = scaler.fit_transform(X) if scaler else X.copy()

    model_specs = [
        ("identity", "Lineal (identidad)"),
        (cfg["model"]["activation"], f"No Lineal ({cfg['model']['activation']})"),
    ]

    # all_runs[label] = list of epoch_losses arrays, one per seed
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

    utils.plot_multi_learning_curves(
        all_runs,
        title=f"Parte 1 — Lineal vs No Lineal ({len(seeds)} seeds, media ± std)",
        path=os.path.join(results_dir, "part1_learning_curves.png"),
    )
    print(f"\n[Gráfico guardado en results/part1_learning_curves.png]")

    print("\nResumen Parte 1:")
    for _, label in model_specs:
        finals = [runs[-1] for runs in all_runs[label]]
        print(f"  {label}: MSE = {np.mean(finals):.6f} ± {np.std(finals):.6f}")


# ---------------------------------------------------------------------------
# Parte 2 — Generalización con split estratificado
# ---------------------------------------------------------------------------

def _make_perceptron(cfg, n_inputs, seed):
    return SimplePerceptron(
        n_inputs=n_inputs,
        activation=cfg["model"]["activation"],
        beta=cfg["model"].get("beta", 1.0),
        initializer=cfg["model"].get("initializer", "random_normal"),
        init_scale=cfg["model"].get("init_scale", 0.1),
        seed=seed,
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )


def run_part2(X, t, y, cfg, results_dir):
    print("\n" + "=" * 60)
    print("PARTE 2 — Generalización")
    print("=" * 60)

    seeds = cfg["experiment"].get("seeds", [cfg["experiment"]["seed"]])
    split_cfg = cfg["data"]["split"]
    val_frac = split_cfg.get("val_frac", 0.15)
    test_frac = split_cfg.get("test_frac", 0.15)
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"].get("batch_size", 32)
    shuffle = cfg["training"].get("shuffle", True)
    es_cfg = cfg["training"].get("early_stopping", {})

    print(f"Entrenando {len(seeds)} seeds "
          f"(optimizer={cfg['training']['optimizer']}, "
          f"lr={cfg['training']['learning_rate']}, batch={batch_size})...")

    all_train_losses = []   # list of epoch_losses per seed
    all_val_losses = []
    all_metrics = []        # list of dicts per seed
    # For the representative confusion matrix and threshold plot we keep the
    # seed whose F1 is closest to the mean.
    seed_test_data = []     # (t_test, y_test, test_scores, test_h, best_t) per seed

    for seed in seeds:
        rng_s = np.random.default_rng(seed)

        # Each seed gets its own split (varies both init and data order)
        train_idx, val_idx, test_idx = stratified_split(y, val_frac, test_frac, seed)
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        t_train, t_val, t_test = t[train_idx], t[val_idx], t[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)

        opt = build_optimizer(cfg["training"])
        p = _make_perceptron(cfg, X_train.shape[1], seed)

        early_stopping = None
        if es_cfg.get("enabled", False):
            early_stopping = EarlyStopping(patience=es_cfg.get("patience", 30))

        epoch_train_losses, val_losses = [], []
        stopped_at = epochs
        for epoch in range(1, epochs + 1):
            train_loss, _ = p.train_epoch(X_train, t_train, opt,
                                          batch_size=batch_size,
                                          shuffle=shuffle, rng=rng_s)
            val_preds = p.predict(X_val)
            val_loss = mse(t_val, val_preds)
            epoch_train_losses.append(train_loss)
            val_losses.append(val_loss)

            if early_stopping is not None and early_stopping(val_loss, p.w):
                p.w = early_stopping.best_weights
                stopped_at = epoch
                break

        all_train_losses.append(epoch_train_losses)
        all_val_losses.append(val_losses)

        # Threshold on val → metrics on test
        val_scores  = p.predict(X_val)
        test_scores = p.predict(X_test)
        test_h = p.pre_activation(X_test)
        _, _, _, _, best_t = threshold_sweep(y_val, val_scores)
        y_pred = (test_scores >= best_t).astype(int)

        fpr, tpr = roc_curve(y_test, test_scores)
        auc_roc = auc(fpr, tpr)
        precs, recs = pr_curve(y_test, test_scores)
        auc_pr = auc(recs, precs)
        precision, recall, f1 = precision_recall_f1(y_test, y_pred)

        all_metrics.append(dict(auc_roc=auc_roc, auc_pr=auc_pr,
                                precision=precision, recall=recall,
                                f1=f1, threshold=best_t))
        seed_test_data.append((t_test, y_test, test_scores, test_h, best_t))

        es_info = f" (early stop é.{stopped_at})" if stopped_at < epochs else ""
        print(f"  seed={seed}{es_info} | "
              f"train MSE={epoch_train_losses[-1]:.5f} val MSE={val_losses[-1]:.5f} | "
              f"F1={f1:.4f} AUC-ROC={auc_roc:.4f}")

    # --- Plots with bands ---
    utils.plot_learning_curves(
        all_train_losses, all_val_losses,
        title=f"Parte 2 — Curvas de aprendizaje ({len(seeds)} seeds, media ± std)",
        path=os.path.join(results_dir, "part2_learning_curves.png"),
    )
    print(f"\n[Gráfico guardado en results/part2_learning_curves.png]")

    # --- Representative run (F1 closest to mean) for ROC / PR / threshold / CM ---
    mean_f1 = np.mean([m["f1"] for m in all_metrics])
    rep_idx = int(np.argmin([abs(m["f1"] - mean_f1) for m in all_metrics]))
    t_test_rep, y_test_rep, test_scores_rep, test_h_rep, best_t_rep = seed_test_data[rep_idx]

    fpr, tpr = roc_curve(y_test_rep, test_scores_rep)
    auc_roc_rep = auc(fpr, tpr)
    utils.plot_roc(fpr, tpr, auc_roc_rep,
                   path=os.path.join(results_dir, "part2_roc.png"))

    precs, recs = pr_curve(y_test_rep, test_scores_rep)
    auc_pr_rep = auc(recs, precs)
    utils.plot_pr(precs, recs, auc_pr_rep,
                  path=os.path.join(results_dir, "part2_pr.png"))

    th, th_precs, th_recs, th_f1s, _ = threshold_sweep(y_test_rep, test_scores_rep)
    utils.plot_threshold_sweep(th, th_precs, th_recs, th_f1s, best_t_rep,
                               path=os.path.join(results_dir, "part2_threshold_sweep.png"))

    utils.plot_target_vs_prediction(
        t_test_rep,
        test_scores_rep,
        path=os.path.join(results_dir, "part2_target_vs_prediction.png"),
    )

    utils.plot_internal_function(
        test_h_rep,
        t_test_rep,
        activation=cfg["model"]["activation"],
        beta=cfg["model"].get("beta", 1.0),
        path=os.path.join(results_dir, "part2_internal_function.png"),
    )

    y_pred_rep = (test_scores_rep >= best_t_rep).astype(int)
    cm = confusion_matrix(y_test_rep, y_pred_rep)
    utils.plot_confusion_matrix(cm,
                                path=os.path.join(results_dir, "part2_confusion_matrix.png"))

    # --- Summary ---
    print("\n--- Métricas en TEST (media ± std sobre seeds) ---")
    for key in ("auc_roc", "auc_pr", "precision", "recall", "f1", "threshold"):
        vals = [m[key] for m in all_metrics]
        print(f"  {key:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    mean_t = np.mean([m["threshold"] for m in all_metrics])
    mean_rec = np.mean([m["recall"] for m in all_metrics])
    mean_prec = np.mean([m["precision"] for m in all_metrics])
    mean_f1 = np.mean([m["f1"] for m in all_metrics])

    print("\n" + "=" * 60)
    print("RECOMENDACIÓN AL CLIENTE")
    print("=" * 60)
    print(f"  Modelo:   Perceptrón {cfg['model']['activation']} "
          f"(β={cfg['model'].get('beta', 1.0)})")
    print(f"  Umbral recomendado (media sobre {len(seeds)} seeds): {mean_t:.4f}")
    print(f"    → Detecta el {mean_rec*100:.1f}% de los fraudes reales (Recall)")
    print(f"    → {mean_prec*100:.1f}% de las alertas son fraude real (Precision)")
    print(f"    → F1 = {mean_f1:.4f}")
    print()
    print("  Nota: reducir el umbral aumenta Recall (detecta más fraudes)")
    print("        pero baja Precision (más falsas alarmas).")
    print("        Subir el umbral reduce falsas alarmas pero pierde fraudes.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, "config.yaml"))

    results_dir = os.path.join(base, cfg["experiment"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)

    run_part1(X, t, cfg, results_dir)
    run_part2(X, t, y, cfg, results_dir)
