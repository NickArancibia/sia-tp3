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
    "record_type", "candidate_name", "strategy", "scaler", "optimizer", "learning_rate", "batch_size",
    "batch_label", "momentum", "seed", "split_kind", "split_id", "epoch", "train_size", "val_size",
    "threshold_selected", "train_mse", "val_mse", "train_auc_pr", "val_auc_pr", "train_precision",
    "train_recall", "train_f1", "val_precision", "val_recall", "val_f1", "generalization_gap_auc_pr",
    "generalization_gap_f1", "generalization_gap_mse", "stopped_at",
]

SUMMARY_FIELDNAMES = [
    "candidate_name", "optimizer", "learning_rate", "batch_size", "batch_label", "momentum",
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


def raw_rows_for_run(cfg, candidate_name, momentum, seed, run):
    base = {
        "candidate_name": candidate_name,
        "strategy": "S3",
        "scaler": cfg["data"]["preprocess"]["feature_scaler"],
        "optimizer": cfg["training"]["optimizer"],
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size": cfg["training"]["batch_size"],
        "batch_label": batch_label(cfg["training"]["batch_size"]),
        "momentum": momentum,
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
            "candidate_name": candidate_name,
            "strategy": "S3",
            "scaler": cfg["data"]["preprocess"]["feature_scaler"],
            "optimizer": cfg["training"]["optimizer"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["batch_size"],
            "batch_label": batch_label(cfg["training"]["batch_size"]),
            "momentum": momentum,
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


def summarize_candidate(raw_df, candidate_name):
    runs = raw_df[(raw_df["record_type"] == "run_metric") & (raw_df["candidate_name"] == candidate_name)].copy()
    return {
        "candidate_name": candidate_name,
        "optimizer": runs["optimizer"].iloc[0],
        "learning_rate": float(runs["learning_rate"].iloc[0]),
        "batch_size": int(runs["batch_size"].iloc[0]),
        "batch_label": runs["batch_label"].iloc[0],
        "momentum": float(runs["momentum"].iloc[0]),
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


def best_row(df, predicate=None):
    rows = df.copy()
    if predicate is not None:
        rows = rows[predicate(rows)]
    rows = rows.sort_values(["mean_val_auc_pr", "std_val_auc_pr", "mean_val_recall"], ascending=[False, True, False])
    return rows.iloc[0]


def main():
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "optimizer")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "optimizer_raw.csv")
    summary_path = os.path.join(out_dir, "optimizer_summary.csv")
    for path in (raw_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    batch_lr_summary = pd.read_csv(os.path.join(EJ1_DIR, "results", "part2", "batch_lr", "batch_lr_summary.csv"))
    cfg_search = cfg["generalization_search"]
    candidates = cfg_search["optimizer_screening_candidates"]
    selected_scaler = cfg_search["selected_scaler"]

    online_best = best_row(batch_lr_summary, lambda rows: rows["batch_size"].astype(int) == 1)
    full_best = best_row(batch_lr_summary, lambda rows: rows["batch_size"].astype(int) == -1)
    global_best = best_row(batch_lr_summary)

    X, t, y, _ = load_data(cfg)
    split_seed = cfg["data"]["split"].get("seed", 42)
    test_frac = cfg["data"]["split"].get("test_frac", 0.15)
    train_val_idx, _, _ = stratified_split(y, val_frac=0.0, test_frac=test_frac, seed=split_seed)
    split_specs = build_s3_splits(train_val_idx, y, cfg["data"]["split"])
    seed = cfg["experiment"].get("seed", 42)

    print("\n" + "=" * 60)
    print("PART 2 — OPTIMIZER COMPARISON")
    print("=" * 60)

    summaries = []
    for candidate in candidates:
        if candidate["source"] == "online_best_lr":
            batch_size = int(online_best["batch_size"])
            learning_rate = float(online_best["learning_rate"])
        elif candidate["source"] == "full_best_lr":
            batch_size = int(full_best["batch_size"])
            learning_rate = float(full_best["learning_rate"])
        else:
            batch_size = int(global_best["batch_size"])
            learning_rate = float(global_best["learning_rate"])

        run_cfg = copy_cfg(
            cfg,
            scaler_name=selected_scaler,
            training_overrides={
                "optimizer": candidate["optimizer"],
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "momentum": candidate.get("momentum", 0.9),
            },
        )

        for split_spec in split_specs:
            run = evaluate_split(run_cfg, X, t, y, split_spec, seed)
            append_rows_csv(
                raw_path,
                raw_rows_for_run(run_cfg, candidate["candidate_name"], candidate.get("momentum", 0.9), seed, run),
                RAW_FIELDNAMES,
            )

        raw_df = pd.read_csv(raw_path)
        summary = summarize_candidate(raw_df, candidate["candidate_name"])
        summaries = [s for s in summaries if s["candidate_name"] != candidate["candidate_name"]]
        summaries.append(summary)
        print(
            f"  {summary['candidate_name']}: AUC-PR val={summary['mean_val_auc_pr']:.4f} "
            f"± {summary['std_val_auc_pr']:.4f} | batch={summary['batch_label']} | lr={summary['learning_rate']}"
        )

    pd.DataFrame(summaries, columns=SUMMARY_FIELDNAMES).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
