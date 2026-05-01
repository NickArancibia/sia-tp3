import copy
import csv
import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd
import numpy as np

from main_part2 import _make_perceptron, load_data
from shared.config_loader import load_config
from shared.losses import mse
from shared.metrics import auc, pr_curve, precision_recall_f1, threshold_sweep
from shared.optimizers import build_optimizer
from shared.preprocessing import build_scaler, stratified_kfold_indices, stratified_split
from shared.regularization import EarlyStopping


STRATEGY_LABELS = {
    "S1": "S1 Holdout",
    "S2": "S2 Repeated Holdout",
    "S3": "S3 5-Fold CV",
}

RAW_FIELDNAMES = [
    "record_type", "strategy", "scaler", "optimizer", "learning_rate", "batch_size", "batch_label",
    "seed", "split_kind", "split_id", "epoch", "train_size", "val_size", "threshold_selected",
    "train_mse", "val_mse", "train_auc_pr", "val_auc_pr", "train_precision", "train_recall",
    "train_f1", "val_precision", "val_recall", "val_f1", "generalization_gap_auc_pr",
    "generalization_gap_f1", "generalization_gap_mse", "stopped_at",
]

SUMMARY_FIELDNAMES = [
    "strategy", "strategy_label", "scaler", "optimizer", "learning_rate", "batch_size", "batch_label",
    "mean_val_auc_pr", "std_val_auc_pr", "mean_val_precision", "std_val_precision", "mean_val_recall",
    "std_val_recall", "mean_val_f1", "std_val_f1", "mean_train_mse", "std_train_mse", "mean_val_mse",
    "std_val_mse", "gap_mse", "mean_threshold", "std_threshold", "mean_stopped_at",
]


def copy_cfg(cfg, scaler_name=None, training_overrides=None):
    new_cfg = copy.deepcopy(cfg)
    if scaler_name is not None:
        new_cfg["data"]["preprocess"]["feature_scaler"] = scaler_name
    if training_overrides:
        new_cfg["training"].update(training_overrides)
    return new_cfg


def effective_batch_size(batch_size):
    return 0 if int(batch_size) == -1 else int(batch_size)


def batch_label(batch_size):
    batch_size = int(batch_size)
    if batch_size == -1:
        return "full"
    if batch_size == 1:
        return "online"
    return str(batch_size)


def append_rows_csv(path, rows, fieldnames):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def build_splits(strategy, train_val_idx, y, split_cfg):
    base_seed = split_cfg.get("seed", 42)
    val_frac = split_cfg.get("inner_val_frac", split_cfg.get("val_frac", 0.15))

    if strategy == "S1":
        train_rel, val_rel, _ = stratified_split(y[train_val_idx], val_frac, test_frac=0.0, seed=base_seed)
        return [{
            "split_kind": "holdout",
            "split_id": 0,
            "train_idx": train_val_idx[train_rel],
            "val_idx": train_val_idx[val_rel],
        }]

    if strategy == "S2":
        repeats = split_cfg.get("repeated_holdout_repeats", 5)
        splits = []
        for repeat in range(repeats):
            train_rel, val_rel, _ = stratified_split(
                y[train_val_idx], val_frac, test_frac=0.0, seed=base_seed + repeat
            )
            splits.append({
                "split_kind": "repeated_holdout",
                "split_id": repeat,
                "train_idx": train_val_idx[train_rel],
                "val_idx": train_val_idx[val_rel],
            })
        return splits

    if strategy == "S3":
        cv_folds = split_cfg.get("cv_folds", 5)
        folds = stratified_kfold_indices(y[train_val_idx], cv_folds, base_seed)
        splits = []
        for fold_id, val_rel in enumerate(folds):
            train_rel = np.concatenate([fold for idx, fold in enumerate(folds) if idx != fold_id])
            splits.append({
                "split_kind": "cv_fold",
                "split_id": fold_id,
                "train_idx": train_val_idx[train_rel],
                "val_idx": train_val_idx[val_rel],
            })
        return splits

    raise ValueError(f"Unknown strategy: {strategy}")


def fit_split_model(cfg, X_train, t_train, X_val, t_val, seed):
    epochs = cfg["training"]["epochs"]
    batch_size = effective_batch_size(cfg["training"].get("batch_size", 32))
    shuffle = cfg["training"].get("shuffle", True)
    es_cfg = cfg["training"].get("early_stopping", {})

    model = _make_perceptron(cfg, X_train.shape[1], seed)
    optimizer = build_optimizer(cfg["training"])
    rng = __import__("numpy").random.default_rng(seed)

    early_stopping = None
    if es_cfg.get("enabled", False):
        early_stopping = EarlyStopping(patience=es_cfg.get("patience", 30))

    train_losses = []
    val_losses = []
    stopped_at = epochs
    for epoch in range(1, epochs + 1):
        train_loss, _ = model.train_epoch(
            X_train, t_train, optimizer, batch_size=batch_size, shuffle=shuffle, rng=rng
        )
        val_scores = model.predict(X_val)
        val_loss = mse(t_val, val_scores)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if early_stopping is not None and early_stopping(val_loss, model.get_params()):
            stopped_at = epoch
            break

    if early_stopping is not None and early_stopping.best_params is not None:
        model.set_params(early_stopping.best_params)

    return model, train_losses, val_losses, stopped_at


def evaluate_split(cfg, X, t, y, split_spec, seed):
    train_idx = split_spec["train_idx"]
    val_idx = split_spec["val_idx"]
    X_train = X[train_idx]
    X_val = X[val_idx]
    t_train = t[train_idx]
    t_val = t[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    model, train_losses, val_losses, stopped_at = fit_split_model(cfg, X_train, t_train, X_val, t_val, seed)
    train_scores = model.predict(X_train)
    val_scores = model.predict(X_val)
    _, _, _, _, best_t = threshold_sweep(y_val, val_scores)
    train_pred = (train_scores >= best_t).astype(int)
    val_pred = (val_scores >= best_t).astype(int)
    train_precision, train_recall, train_f1 = precision_recall_f1(y_train, train_pred)
    val_precision, val_recall, val_f1 = precision_recall_f1(y_val, val_pred)
    train_precs, train_recs = pr_curve(y_train, train_scores)
    val_precs, val_recs = pr_curve(y_val, val_scores)

    return {
        "split_kind": split_spec["split_kind"],
        "split_id": split_spec["split_id"],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_auc_pr": auc(train_recs, train_precs),
        "val_auc_pr": auc(val_recs, val_precs),
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "threshold": float(best_t),
        "train_mse": float(mse(t_train, train_scores)),
        "val_mse": float(mse(t_val, val_scores)),
        "stopped_at": stopped_at,
    }


def raw_rows_for_run(cfg, strategy, seed, run):
    base = {
        "strategy": strategy,
        "scaler": cfg["data"]["preprocess"]["feature_scaler"],
        "optimizer": cfg["training"]["optimizer"],
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size": cfg["training"]["batch_size"],
        "batch_label": batch_label(cfg["training"]["batch_size"]),
        "seed": seed,
        "split_kind": run["split_kind"],
        "split_id": run["split_id"],
        "train_size": len(run["train_idx"]),
        "val_size": len(run["val_idx"]),
        "threshold_selected": run["threshold"],
        "train_mse": run["train_mse"],
        "val_mse": run["val_mse"],
        "train_auc_pr": run["train_auc_pr"],
        "val_auc_pr": run["val_auc_pr"],
        "train_precision": run["train_precision"],
        "train_recall": run["train_recall"],
        "train_f1": run["train_f1"],
        "val_precision": run["val_precision"],
        "val_recall": run["val_recall"],
        "val_f1": run["val_f1"],
        "generalization_gap_auc_pr": run["train_auc_pr"] - run["val_auc_pr"],
        "generalization_gap_f1": run["train_f1"] - run["val_f1"],
        "generalization_gap_mse": run["val_mse"] - run["train_mse"],
        "stopped_at": run["stopped_at"],
    }
    rows = [{"record_type": "run_metric", "epoch": "", **base}]
    for epoch_idx, (train_loss, val_loss) in enumerate(zip(run["train_losses"], run["val_losses"]), start=1):
        rows.append({
            "record_type": "curve_point",
            "strategy": strategy,
            "scaler": cfg["data"]["preprocess"]["feature_scaler"],
            "optimizer": cfg["training"]["optimizer"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["batch_size"],
            "batch_label": batch_label(cfg["training"]["batch_size"]),
            "seed": seed,
            "split_kind": run["split_kind"],
            "split_id": run["split_id"],
            "epoch": epoch_idx,
            "train_size": len(run["train_idx"]),
            "val_size": len(run["val_idx"]),
            "threshold_selected": "",
            "train_mse": train_loss,
            "val_mse": val_loss,
            "train_auc_pr": "",
            "val_auc_pr": "",
            "train_precision": "",
            "train_recall": "",
            "train_f1": "",
            "val_precision": "",
            "val_recall": "",
            "val_f1": "",
            "generalization_gap_auc_pr": "",
            "generalization_gap_f1": "",
            "generalization_gap_mse": "",
            "stopped_at": run["stopped_at"],
        })
    return rows


def summarize_strategy(raw_df, strategy):
    runs = raw_df[(raw_df["record_type"] == "run_metric") & (raw_df["strategy"] == strategy)].copy()
    return {
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "scaler": runs["scaler"].iloc[0],
        "optimizer": runs["optimizer"].iloc[0],
        "learning_rate": float(runs["learning_rate"].iloc[0]),
        "batch_size": int(runs["batch_size"].iloc[0]),
        "batch_label": runs["batch_label"].iloc[0],
        "mean_val_auc_pr": float(runs["val_auc_pr"].astype(float).mean()),
        "std_val_auc_pr": float(runs["val_auc_pr"].astype(float).std(ddof=0)),
        "mean_val_precision": float(runs["val_precision"].astype(float).mean()),
        "std_val_precision": float(runs["val_precision"].astype(float).std(ddof=0)),
        "mean_val_recall": float(runs["val_recall"].astype(float).mean()),
        "std_val_recall": float(runs["val_recall"].astype(float).std(ddof=0)),
        "mean_val_f1": float(runs["val_f1"].astype(float).mean()),
        "std_val_f1": float(runs["val_f1"].astype(float).std(ddof=0)),
        "mean_train_mse": float(runs["train_mse"].astype(float).mean()),
        "std_train_mse": float(runs["train_mse"].astype(float).std(ddof=0)),
        "mean_val_mse": float(runs["val_mse"].astype(float).mean()),
        "std_val_mse": float(runs["val_mse"].astype(float).std(ddof=0)),
        "gap_mse": float(runs["generalization_gap_mse"].astype(float).mean()),
        "mean_threshold": float(runs["threshold_selected"].astype(float).mean()),
        "std_threshold": float(runs["threshold_selected"].astype(float).std(ddof=0)),
        "mean_stopped_at": float(runs["stopped_at"].astype(float).mean()),
    }


def main():
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "data_strategy")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "strategy_raw.csv")
    summary_path = os.path.join(out_dir, "strategy_summary.csv")
    for path in (raw_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    X, t, y, _ = load_data(cfg)
    search_cfg = cfg["generalization_search"]
    baseline = search_cfg["strategy_baseline"]
    run_cfg = copy_cfg(
        cfg,
        scaler_name=search_cfg["selected_scaler"],
        training_overrides={
            "optimizer": baseline["optimizer"],
            "batch_size": baseline["batch_size"],
            "learning_rate": baseline["learning_rate"],
        },
    )

    split_seed = cfg["data"]["split"].get("seed", 42)
    test_frac = cfg["data"]["split"].get("test_frac", 0.15)
    train_val_idx, _, _ = stratified_split(y, val_frac=0.0, test_frac=test_frac, seed=split_seed)
    seed = cfg["experiment"].get("seed", 42)

    print("\n" + "=" * 60)
    print("PART 2 — DATA STRATEGY")
    print("=" * 60)

    for strategy in ("S1", "S2", "S3"):
        for split_spec in build_splits(strategy, train_val_idx, y, run_cfg["data"]["split"]):
            run = evaluate_split(run_cfg, X, t, y, split_spec, seed)
            append_rows_csv(raw_path, raw_rows_for_run(run_cfg, strategy, seed, run), RAW_FIELDNAMES)

        raw_df = pd.read_csv(raw_path)
        summary = summarize_strategy(raw_df, strategy)
        print(
            f"  {summary['strategy_label']}: AUC-PR val={summary['mean_val_auc_pr']:.4f} "
            f"± {summary['std_val_auc_pr']:.4f} | MSE val={summary['mean_val_mse']:.6f}"
        )

    raw_df = pd.read_csv(raw_path)
    summaries = [summarize_strategy(raw_df, strategy) for strategy in ("S1", "S2", "S3")]
    pd.DataFrame(summaries, columns=SUMMARY_FIELDNAMES).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
