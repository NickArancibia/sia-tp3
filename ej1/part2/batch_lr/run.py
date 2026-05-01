import copy
import csv
import os
import sys

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)

import numpy as np
import pandas as pd

from main_part2 import _make_perceptron, load_data
from shared.config_loader import load_config
from shared.losses import mse
from shared.metrics import auc, pr_curve, precision_recall_f1, threshold_sweep
from shared.optimizers import build_optimizer
from shared.preprocessing import build_scaler, stratified_kfold_indices, stratified_split
from shared.regularization import EarlyStopping


RAW_FIELDNAMES = [
    "record_type", "config_name", "strategy", "scaler", "optimizer", "learning_rate", "batch_size",
    "batch_label", "seed", "split_kind", "split_id", "epoch", "train_size", "val_size",
    "threshold_selected", "train_mse", "val_mse", "train_auc_pr", "val_auc_pr", "train_precision",
    "train_recall", "train_f1", "val_precision", "val_recall", "val_f1", "generalization_gap_auc_pr",
    "generalization_gap_f1", "generalization_gap_mse", "stopped_at",
]

SUMMARY_FIELDNAMES = [
    "config_name", "learning_rate", "batch_size", "batch_label", "mean_val_auc_pr", "std_val_auc_pr",
    "mean_val_precision", "std_val_precision", "mean_val_recall", "std_val_recall", "mean_val_f1",
    "std_val_f1", "mean_train_mse", "std_train_mse", "mean_val_mse", "std_val_mse", "gap_mse",
    "mean_threshold", "std_threshold", "mean_stopped_at",
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


def format_lr(lr):
    if lr < 1e-3:
        mantissa, exp = f"{lr:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp = exp.replace("+0", "+").replace("-0", "-")
        return f"{mantissa}e{exp}"
    return f"{lr:.4f}".rstrip("0").rstrip(".")


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


def build_s3_splits(train_val_idx, y, split_cfg):
    base_seed = split_cfg.get("seed", 42)
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


def fit_split_model(cfg, X_train, t_train, X_val, t_val, seed):
    epochs = cfg["training"]["epochs"]
    batch_size = effective_batch_size(cfg["training"].get("batch_size", 32))
    shuffle = cfg["training"].get("shuffle", True)
    es_cfg = cfg["training"].get("early_stopping", {})

    model = _make_perceptron(cfg, X_train.shape[1], seed)
    optimizer = build_optimizer(cfg["training"])
    rng = np.random.default_rng(seed)

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


def raw_rows_for_run(cfg, config_name, seed, run):
    base = {
        "config_name": config_name,
        "strategy": "S3",
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
            "config_name": config_name,
            "strategy": "S3",
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


def summarize_cell(raw_df, config_name, learning_rate, batch_size):
    runs = raw_df[
        (raw_df["record_type"] == "run_metric")
        & (raw_df["config_name"] == config_name)
        & (raw_df["batch_size"].astype(int) == int(batch_size))
        & (raw_df["learning_rate"].astype(float).round(12) == round(float(learning_rate), 12))
    ].copy()
    return {
        "config_name": config_name,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "batch_label": batch_label(batch_size),
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


def config_name_for_batch(batch_size):
    mapping = {1: "E0", 4: "E1", 16: "E2", 32: "E3", 128: "E4", 512: "E5", -1: "E6"}
    return mapping[int(batch_size)]


def main():
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "batch_lr")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "batch_lr_raw.csv")
    summary_path = os.path.join(out_dir, "batch_lr_summary.csv")
    for path in (raw_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    X, t, y, _ = load_data(cfg)
    search_cfg = cfg["generalization_search"]
    batch_sizes = [int(v) for v in search_cfg["heatmap_batch_sizes"]]
    learning_rates = [float(v) for v in search_cfg["heatmap_learning_rates"]]
    split_seed = cfg["data"]["split"].get("seed", 42)
    test_frac = cfg["data"]["split"].get("test_frac", 0.15)
    train_val_idx, _, _ = stratified_split(y, val_frac=0.0, test_frac=test_frac, seed=split_seed)
    split_specs = build_s3_splits(train_val_idx, y, cfg["data"]["split"])
    seed = cfg["experiment"].get("seed", 42)

    print("\n" + "=" * 60)
    print("PART 2 — BATCH/LR")
    print("=" * 60)

    summaries = []
    for batch_size in batch_sizes:
        config_name = config_name_for_batch(batch_size)
        for learning_rate in learning_rates:
            run_cfg = copy_cfg(
                cfg,
                scaler_name=search_cfg["selected_scaler"],
                training_overrides={
                    "optimizer": "adam",
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
            )
            for split_spec in split_specs:
                run = evaluate_split(run_cfg, X, t, y, split_spec, seed)
                append_rows_csv(raw_path, raw_rows_for_run(run_cfg, config_name, seed, run), RAW_FIELDNAMES)

            raw_df = pd.read_csv(raw_path)
            summary = summarize_cell(raw_df, config_name, learning_rate, batch_size)
            summaries = [
                s for s in summaries
                if not (s["config_name"] == config_name and s["batch_size"] == batch_size and round(s["learning_rate"], 12) == round(learning_rate, 12))
            ]
            summaries.append(summary)
            print(
                f"  {config_name} (batch={batch_label(batch_size)}, lr={format_lr(learning_rate)}): "
                f"AUC-PR val={summary['mean_val_auc_pr']:.4f} ± {summary['std_val_auc_pr']:.4f}"
            )

    summaries = sorted(summaries, key=lambda s: (batch_sizes.index(s["batch_size"]), learning_rates.index(s["learning_rate"])))
    pd.DataFrame(summaries, columns=SUMMARY_FIELDNAMES).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
