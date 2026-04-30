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
from shared.preprocessing import build_scaler, stratified_kfold_indices, stratified_split
from shared.regularization import EarlyStopping


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


def _run_fold(X, t, y, cfg, train_idx, val_idx, test_idx, seed):
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"].get("batch_size", 32)
    shuffle = cfg["training"].get("shuffle", True)
    es_cfg = cfg["training"].get("early_stopping", {})

    rng_s = np.random.default_rng(seed)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    t_train, t_val = t[train_idx], t[val_idx]
    y_val, y_test = y[val_idx], y[test_idx]

    scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    opt = build_optimizer(cfg["training"])
    p = _make_perceptron(cfg, X_train.shape[1], seed)

    early_stopping = None
    if es_cfg.get("enabled", False):
        early_stopping = EarlyStopping(patience=es_cfg.get("patience", 30))

    epoch_train_losses, val_losses = [], []
    stopped_at = epochs
    for epoch in range(1, epochs + 1):
        train_loss, _ = p.train_epoch(
            X_train,
            t_train,
            opt,
            batch_size=batch_size,
            shuffle=shuffle,
            rng=rng_s,
        )
        val_preds = p.predict(X_val)
        val_loss = mse(t_val, val_preds)
        epoch_train_losses.append(train_loss)
        val_losses.append(val_loss)

        if early_stopping is not None and early_stopping(val_loss, p.get_params()):
            stopped_at = epoch
            break

    if early_stopping is not None and early_stopping.best_params is not None:
        p.set_params(early_stopping.best_params)

    val_scores = p.predict(X_val)
    thresholds, th_precs, th_recs, th_f1s, best_t = threshold_sweep(y_val, val_scores)
    val_pred = (val_scores >= best_t).astype(int)
    val_precision, val_recall, val_f1 = precision_recall_f1(y_val, val_pred)

    test_scores = p.predict(X_test)
    test_preacts = p.pre_activation(X_test)

    return {
        "train_losses": epoch_train_losses,
        "val_losses": val_losses,
        "thresholds": thresholds,
        "threshold_precisions": th_precs,
        "threshold_recalls": th_recs,
        "threshold_f1s": th_f1s,
        "best_t": float(best_t),
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "test_idx": test_idx,
        "test_scores": test_scores,
        "test_preacts": test_preacts,
        "stopped_at": stopped_at,
        "y_test": y_test,
    }


def _interp_curve(grid, x, y):
    if len(x) == 1:
        return np.full_like(grid, y[0], dtype=float)
    return np.interp(grid, x, y)


def _aggregate_threshold_curves(fold_results, n_points=300):
    lo = min(float(fold["thresholds"][0]) for fold in fold_results)
    hi = max(float(fold["thresholds"][-1]) for fold in fold_results)
    grid = np.array([lo]) if np.isclose(lo, hi) else np.linspace(lo, hi, n_points)

    prec_curves, rec_curves, f1_curves = [], [], []
    for fold in fold_results:
        thresholds = fold["thresholds"]
        prec_curves.append(_interp_curve(grid, thresholds, fold["threshold_precisions"]))
        rec_curves.append(_interp_curve(grid, thresholds, fold["threshold_recalls"]))
        f1_curves.append(_interp_curve(grid, thresholds, fold["threshold_f1s"]))

    prec_arr = np.array(prec_curves)
    rec_arr = np.array(rec_curves)
    f1_arr = np.array(f1_curves)
    return {
        "thresholds": grid,
        "precisions_mean": prec_arr.mean(axis=0),
        "precisions_std": prec_arr.std(axis=0),
        "recalls_mean": rec_arr.mean(axis=0),
        "recalls_std": rec_arr.std(axis=0),
        "f1s_mean": f1_arr.mean(axis=0),
        "f1s_std": f1_arr.std(axis=0),
    }


def _summarize_seed(seed, fold_results, y, t):
    n_samples = len(y)
    oof_scores = np.zeros(n_samples, dtype=float)
    oof_preacts = np.zeros(n_samples, dtype=float)
    seen = np.zeros(n_samples, dtype=int)

    thresholds = []
    val_f1s = []
    train_curves = []
    val_curves = []

    for fold in fold_results:
        idx = fold["test_idx"]
        oof_scores[idx] = fold["test_scores"]
        oof_preacts[idx] = fold["test_preacts"]
        seen[idx] += 1

        thresholds.append(fold["best_t"])
        val_f1s.append(fold["val_f1"])
        train_curves.append(fold["train_losses"])
        val_curves.append(fold["val_losses"])

    if not np.all(seen == 1):
        raise ValueError("Cada muestra debe aparecer exactamente una vez en el fold externo")

    mean_threshold = float(np.mean(thresholds))
    oof_pred = (oof_scores >= mean_threshold).astype(int)
    fpr, tpr = roc_curve(y, oof_scores)
    auc_roc = auc(fpr, tpr)
    precs, recs = pr_curve(y, oof_scores)
    auc_pr = auc(recs, precs)
    precision, recall, f1 = precision_recall_f1(y, oof_pred)

    return {
        "seed": seed,
        "fold_results": fold_results,
        "train_curves": train_curves,
        "val_curves": val_curves,
        "val_mean_f1": float(np.mean(val_f1s)),
        "val_std_f1": float(np.std(val_f1s)),
        "threshold_mean": mean_threshold,
        "threshold_std": float(np.std(thresholds)),
        "oof_scores": oof_scores,
        "oof_preacts": oof_preacts,
        "oof_pred": oof_pred,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": confusion_matrix(y, oof_pred),
        "fpr": fpr,
        "tpr": tpr,
        "pr_precisions": precs,
        "pr_recalls": recs,
        "threshold_curves": _aggregate_threshold_curves(fold_results),
    }


def _metric_summary(values):
    return float(np.mean(values)), float(np.std(values))


def run_part2(X, t, y, cfg, results_dir):
    print("\n" + "=" * 60)
    print("PARTE 2 — Generalización (CV puro)")
    print("=" * 60)

    seeds = cfg["experiment"].get("seeds", [cfg["experiment"]["seed"]])
    split_cfg = cfg["data"]["split"]
    cv_folds = split_cfg.get("cv_folds", 5)
    inner_val_frac = split_cfg.get("inner_val_frac", split_cfg.get("val_frac", 0.15))
    split_seed = split_cfg.get("seed", 42)
    batch_size = cfg["training"].get("batch_size", 32)

    outer_folds = stratified_kfold_indices(y, cv_folds, split_seed)

    print(
        f"CV puro: {cv_folds} folds x {len(seeds)} seeds "
        f"(optimizer={cfg['training']['optimizer']}, "
        f"lr={cfg['training']['learning_rate']}, batch={batch_size}, "
        f"val interno={inner_val_frac:.2f})"
    )

    seed_results = []
    for idx, seed in enumerate(seeds, start=1):
        fold_results = []
        for fold_idx in range(cv_folds):
            test_idx = outer_folds[fold_idx]
            train_val_idx = np.concatenate([
                fold for i, fold in enumerate(outer_folds) if i != fold_idx
            ])

            train_rel, val_rel, _ = stratified_split(
                y[train_val_idx],
                inner_val_frac,
                test_frac=0.0,
                seed=split_seed + fold_idx,
            )
            train_idx = train_val_idx[train_rel]
            val_idx = train_val_idx[val_rel]

            fold_results.append(_run_fold(X, t, y, cfg, train_idx, val_idx, test_idx, seed))

        seed_result = _summarize_seed(seed, fold_results, y, t)
        seed_results.append(seed_result)
        print(
            f"  seed {idx}/{len(seeds)} lista | "
            f"VAL F1={seed_result['val_mean_f1']:.4f} ± {seed_result['val_std_f1']:.4f}"
        )

    best_seed_result = max(seed_results, key=lambda result: result["val_mean_f1"])
    best_seed = best_seed_result["seed"]

    # Los gráficos finales de generalización se generan desde
    # simulate_generalization.py para evitar duplicar salidas.

    print("\n--- Métricas OOF sobre seeds (media ± std) ---")
    metric_specs = [
        ("val_mean_f1", "val_f1"),
        ("auc_roc", "auc_roc"),
        ("auc_pr", "auc_pr"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("threshold_mean", "threshold"),
    ]
    for key, label in metric_specs:
        mean, std = _metric_summary([result[key] for result in seed_results])
        print(f"  {label:12s}: {mean:.4f} ± {std:.4f}")

    print("\n" + "=" * 60)
    print("MEJOR SEED POR F1 EN VAL")
    print("=" * 60)
    print(f"  seed:     {best_seed}")
    print(
        f"  VAL F1:   {best_seed_result['val_mean_f1']:.4f} "
        f"± {best_seed_result['val_std_f1']:.4f}"
    )
    print(f"  AUC-ROC:  {best_seed_result['auc_roc']:.4f}")
    print(f"  AUC-PR:   {best_seed_result['auc_pr']:.4f}")
    print(f"  Precision:{best_seed_result['precision']:.4f}")
    print(f"  Recall:   {best_seed_result['recall']:.4f}")
    print(f"  F1:       {best_seed_result['f1']:.4f}")

    print("\n" + "=" * 60)
    print("RECOMENDACIÓN AL CLIENTE")
    print("=" * 60)
    print(f"  Modelo:   Perceptrón {cfg['model']['activation']} (β={cfg['model'].get('beta', 1.0)})")
    print(
        f"  Umbral recomendado: {best_seed_result['threshold_mean']:.4f} "
        f"± {best_seed_result['threshold_std']:.4f}"
    )
    print(f"    → Detecta el {best_seed_result['recall']*100:.1f}% de los fraudes reales (Recall)")
    print(f"    → {best_seed_result['precision']*100:.1f}% de las alertas son fraude real (Precision)")
    print(f"    → F1 = {best_seed_result['f1']:.4f}")
    print()
    print("  Nota: el umbral se seleccionó en validación interna y se promedió sobre folds.")
    print("        Reducirlo aumenta Recall pero baja Precision; subirlo hace lo inverso.")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, "config.yaml"))

    results_dir = os.path.join(base, cfg["experiment"]["results_dir"], "part2")
    os.makedirs(results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)
    run_part2(X, t, y, cfg, results_dir)
