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

def run_part1(X, t, cfg, rng, results_dir):
    print("\n" + "=" * 60)
    print("PARTE 1 — Análisis de aprendizaje (todos los datos)")
    print("=" * 60)

    scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
    X_scaled = scaler.fit_transform(X) if scaler else X.copy()

    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"].get("batch_size", 32)
    shuffle = cfg["training"].get("shuffle", True)
    log_every = cfg["experiment"].get("log_every", 50)
    beta = cfg["model"].get("beta", 1.0)
    init_scale = cfg["model"].get("init_scale", 0.1)
    initializer = cfg["model"].get("initializer", "random_normal")
    seed = cfg["experiment"]["seed"]

    curves = {}
    step_curves = {}

    for activation, label in [("identity", "Lineal (identidad)"),
                               (cfg["model"]["activation"],
                                f"No Lineal ({cfg['model']['activation']})")]:
        print(f"\n--- Perceptrón {label} ---")
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
        step_losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss, batch_losses = p.train_epoch(X_scaled, t, opt,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle, rng=rng)
            epoch_losses.append(epoch_loss)
            step_losses.extend(batch_losses)
            if epoch % log_every == 0:
                print(f"  Época {epoch:5d} | MSE: {epoch_loss:.6f}")

        print(f"  MSE final: {epoch_losses[-1]:.6f}")
        curves[label] = epoch_losses
        step_curves[label] = step_losses

    utils.plot_multi_learning_curves(
        curves, step_curves,
        title="Parte 1 — Lineal vs No Lineal (todos los datos)",
        path=os.path.join(results_dir, "part1_learning_curves.png"),
    )
    print(f"\n[Gráfico guardado en results/part1_learning_curves.png]")

    linear_final = curves["Lineal (identidad)"][-1]
    nl_key = f"No Lineal ({cfg['model']['activation']})"
    nonlin_final = curves[nl_key][-1]
    print(f"\nResumen Parte 1:")
    print(f"  MSE final Lineal:     {linear_final:.6f}")
    print(f"  MSE final No Lineal:  {nonlin_final:.6f}")
    print(f"  Diferencia:           {linear_final - nonlin_final:.6f}")


# ---------------------------------------------------------------------------
# Parte 2 — Generalización con split estratificado
# ---------------------------------------------------------------------------

def _make_perceptron(cfg, n_inputs):
    return SimplePerceptron(
        n_inputs=n_inputs,
        activation=cfg["model"]["activation"],
        beta=cfg["model"].get("beta", 1.0),
        initializer=cfg["model"].get("initializer", "random_normal"),
        init_scale=cfg["model"].get("init_scale", 0.1),
        seed=cfg["experiment"]["seed"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )


def run_part2(X, t, y, cfg, rng, results_dir):
    print("\n" + "=" * 60)
    print("PARTE 2 — Generalización")
    print("=" * 60)

    split_cfg = cfg["data"]["split"]
    val_frac = split_cfg.get("val_frac", 0.15)
    test_frac = split_cfg.get("test_frac", 0.15)
    seed = split_cfg.get("seed", 42)

    # Stratified split using hard labels (fraud class is imbalanced)
    train_idx, val_idx, test_idx = stratified_split(y, val_frac, test_frac, seed)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    t_train, t_val, t_test = t[train_idx], t[val_idx], t[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    print(f"Split: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")
    print(f"Tasa de fraude — Train: {y_train.mean():.4f} | "
          f"Val: {y_val.mean():.4f} | Test: {y_test.mean():.4f}")

    # Fit scaler on TRAIN only
    scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Build model and optimizer
    opt = build_optimizer(cfg["training"])
    p = _make_perceptron(cfg, X_train.shape[1])

    es_cfg = cfg["training"].get("early_stopping", {})
    early_stopping = None
    if es_cfg.get("enabled", False):
        early_stopping = EarlyStopping(patience=es_cfg.get("patience", 30))

    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"].get("batch_size", 32)
    shuffle = cfg["training"].get("shuffle", True)
    log_every = cfg["experiment"].get("log_every", 50)

    epoch_train_losses, val_losses, step_train_losses = [], [], []
    print(f"\nEntrenando {epochs} épocas (optimizer={cfg['training']['optimizer']}, "
          f"lr={cfg['training']['learning_rate']}, batch={batch_size})...")

    for epoch in range(1, epochs + 1):
        train_loss, batch_losses = p.train_epoch(X_train, t_train, opt,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle, rng=rng)
        val_preds = p.predict(X_val)
        val_loss = mse(t_val, val_preds)

        epoch_train_losses.append(train_loss)
        val_losses.append(val_loss)
        step_train_losses.extend(batch_losses)

        if epoch % log_every == 0:
            print(f"  Época {epoch:5d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if early_stopping is not None and early_stopping(val_loss, p.w):
            print(f"  Early stopping en época {epoch} (mejor val MSE: {early_stopping._best:.6f})")
            p.w = early_stopping.best_weights
            break

    utils.plot_learning_curves(
        epoch_train_losses, val_losses, step_train_losses,
        title="Parte 2 — Curvas de aprendizaje",
        path=os.path.join(results_dir, "part2_learning_curves.png"),
    )
    print(f"\n[Gráfico guardado en results/part2_learning_curves.png]")

    # -----------------------------------------------------------------
    # Threshold selection on VAL set → evaluation on TEST set
    # -----------------------------------------------------------------
    val_scores = p.predict(X_val)
    test_scores = p.predict(X_test)

    _, _, _, _, best_t = threshold_sweep(y_val, val_scores)
    print(f"\nUmbral óptimo (F1 en val): {best_t:.4f}")

    # ROC — test
    fpr, tpr = roc_curve(y_test, test_scores)
    auc_roc = auc(fpr, tpr)
    utils.plot_roc(fpr, tpr, auc_roc,
                   path=os.path.join(results_dir, "part2_roc.png"))

    # PR — test
    precs, recs = pr_curve(y_test, test_scores)
    auc_pr = auc(recs, precs)
    utils.plot_pr(precs, recs, auc_pr,
                  path=os.path.join(results_dir, "part2_pr.png"))

    # Threshold sweep — test (for visualization)
    th, th_precs, th_recs, th_f1s, _ = threshold_sweep(y_test, test_scores)
    utils.plot_threshold_sweep(th, th_precs, th_recs, th_f1s, best_t,
                               path=os.path.join(results_dir, "part2_threshold_sweep.png"))

    # Final metrics at best threshold on TEST
    y_pred = (test_scores >= best_t).astype(int)
    precision, recall, f1 = precision_recall_f1(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    utils.plot_confusion_matrix(cm,
                                path=os.path.join(results_dir, "part2_confusion_matrix.png"))

    print("\n--- Métricas en TEST ---")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"  AUC-PR:    {auc_pr:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"\n  Matriz de confusión (umbral = {best_t:.4f}):")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    print("\n" + "=" * 60)
    print("RECOMENDACIÓN AL CLIENTE")
    print("=" * 60)
    print(f"  Modelo:   Perceptrón {cfg['model']['activation']} "
          f"(β={cfg['model'].get('beta', 1.0)})")
    print(f"  Features: {X_train.shape[1]} variables de transacción")
    print(f"  Umbral recomendado: {best_t:.4f}")
    print(f"    → Detecta el {recall*100:.1f}% de los fraudes reales (Recall)")
    print(f"    → {precision*100:.1f}% de las alertas son fraude real (Precision)")
    print(f"    → F1 = {f1:.4f}")
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

    rng = np.random.default_rng(cfg["experiment"]["seed"])
    results_dir = os.path.join(base, cfg["experiment"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)

    run_part1(X, t, cfg, rng, results_dir)
    run_part2(X, t, y, cfg, rng, results_dir)
