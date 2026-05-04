import copy
import csv
import os
import pickle
import time

import numpy as np
import pandas as pd

from shared.losses import mse
from shared.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    pr_curve,
    precision_recall_f1,
    precision_recall_fbeta,
)
from shared.optimizers import build_optimizer
from shared.preprocessing import stratified_kfold_indices, stratified_split
from shared.regularization import EarlyStopping


RAW_FIELDNAMES = [
    "record_type", "experiment_type", "config_name", "candidate_name", "strategy",
    "scaler", "optimizer", "learning_rate", "batch_size", "batch_label", "momentum", "seed",
    "activation", "beta", "split_kind", "split_id", "subset", "epoch", "sample_idx",
    "threshold_selected", "threshold_rule", "train_size", "val_size", "test_size",
    "epochs_planned", "stopped_at", "elapsed_s", "elapsed_s_cumulative",
    "train_mse", "val_mse", "mse", "auc_pr", "accuracy", "precision", "recall", "f1", "f2",
    "cost_fn2_fp1", "cost_mean_fn2_fp1", "tn", "fp", "fn", "tp",
    "y_true", "target_bigmodel", "score", "pre_activation", "pred",
]


def read_raw_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


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


def append_rows_csv(path, rows, fieldnames=RAW_FIELDNAMES):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            out = {field: "" for field in fieldnames}
            out.update(row)
            writer.writerow(out)


def load_raw_df(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=RAW_FIELDNAMES)
    raw_df = read_raw_csv(path)
    for field in RAW_FIELDNAMES:
        if field not in raw_df.columns:
            raw_df[field] = ""
    return raw_df


def prepare_raw_file(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=RAW_FIELDNAMES)
    raw_df = read_raw_csv(path)
    missing = [field for field in RAW_FIELDNAMES if field not in raw_df.columns]
    if not missing:
        for field in RAW_FIELDNAMES:
            if field not in raw_df.columns:
                raw_df[field] = ""
        return raw_df

    backup_path = f"{path}.legacy"
    os.replace(path, backup_path)
    print(f"[INFO] Schema viejo detectado en {path}; se movio a {backup_path}")
    return pd.DataFrame(columns=RAW_FIELDNAMES)


def sanitize_key(value):
    text = str(value)
    safe = []
    for ch in text:
        safe.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(safe)


def build_train_val_test_indices(y, split_cfg):
    split_seed = split_cfg.get("seed", 42)
    test_frac = split_cfg.get("test_frac", 0.15)
    train_val_idx, _, test_idx = stratified_split(y, val_frac=0.0, test_frac=test_frac, seed=split_seed)
    return train_val_idx, test_idx


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
                y[train_val_idx], val_frac, test_frac=0.0, seed=base_seed + repeat,
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


def checkpoint_cfg(cfg):
    section = cfg.get("generalization_search", {}).get("checkpointing", {})
    return {
        "enabled": bool(section.get("enabled", True)),
        "interval_epochs": int(section.get("interval_epochs", 50)),
    }


class CheckpointManager:
    def __init__(self, base_dir, run_key, enabled=True, interval_epochs=50):
        self.enabled = enabled
        self.interval_epochs = interval_epochs
        self.run_dir = os.path.join(base_dir, sanitize_key(run_key))
        self.latest_path = os.path.join(self.run_dir, "latest.pkl")
        self.best_path = os.path.join(self.run_dir, "best.pkl")

    def load_latest(self):
        if not self.enabled or not os.path.exists(self.latest_path):
            return None
        with open(self.latest_path, "rb") as f:
            return pickle.load(f)

    def save_latest(self, state):
        if not self.enabled:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.latest_path, "wb") as f:
            pickle.dump(state, f)

    def save_best(self, state):
        if not self.enabled:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.best_path, "wb") as f:
            pickle.dump(state, f)


def _checkpoint_state(model, optimizer, rng, epoch, train_losses, val_losses, elapsed_s_total,
                      elapsed_s_cumulative, stopped_at, early_stopping=None):
    return {
        "model_params": copy.deepcopy(model.get_params()),
        "optimizer_state": optimizer.state_dict() if hasattr(optimizer, "state_dict") else None,
        "rng_state": copy.deepcopy(rng.bit_generator.state),
        "epoch": int(epoch),
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "elapsed_s_total": float(elapsed_s_total),
        "elapsed_s_cumulative": list(elapsed_s_cumulative),
        "stopped_at": int(stopped_at),
        "early_stopping_state": None if early_stopping is None else early_stopping.state_dict(),
    }


def _restore_checkpoint(model, optimizer, rng, early_stopping, state):
    model.set_params(copy.deepcopy(state["model_params"]))
    if state.get("optimizer_state") is not None and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(state["optimizer_state"])
    if state.get("rng_state") is not None:
        rng.bit_generator.state = state["rng_state"]
    if early_stopping is not None and state.get("early_stopping_state") is not None:
        early_stopping.load_state_dict(state["early_stopping_state"])


def fit_model_with_validation(cfg, model_builder, X_train, t_train, X_val, t_val, seed, checkpoint_dir, run_key):
    epochs = int(cfg["training"]["epochs"])
    batch_size = effective_batch_size(cfg["training"].get("batch_size", 32))
    shuffle = cfg["training"].get("shuffle", True)
    es_cfg = cfg["training"].get("early_stopping", {})
    checkpoint = CheckpointManager(checkpoint_dir, run_key, **checkpoint_cfg(cfg))

    model = model_builder(cfg, X_train.shape[1], seed)
    optimizer = build_optimizer(cfg["training"])
    rng = np.random.default_rng(seed)
    early_stopping = None
    if es_cfg.get("enabled", False):
        early_stopping = EarlyStopping(patience=es_cfg.get("patience", 50))

    train_losses = []
    val_losses = []
    elapsed_s_cumulative = []
    elapsed_s_total = 0.0
    stopped_at = epochs
    start_epoch = 1

    state = checkpoint.load_latest()
    if state is not None:
        _restore_checkpoint(model, optimizer, rng, early_stopping, state)
        train_losses = list(state.get("train_losses", []))
        val_losses = list(state.get("val_losses", []))
        elapsed_s_total = float(state.get("elapsed_s_total", 0.0))
        elapsed_s_cumulative = list(state.get("elapsed_s_cumulative", []))
        stopped_at = int(state.get("stopped_at", epochs))
        start_epoch = int(state.get("epoch", 0)) + 1

    if start_epoch > epochs:
        if early_stopping is not None and early_stopping.best_params is not None:
            model.set_params(early_stopping.best_params)
        return model, train_losses, val_losses, stopped_at, elapsed_s_total, elapsed_s_cumulative

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _ = model.train_epoch(
            X_train, t_train, optimizer, batch_size=batch_size, shuffle=shuffle, rng=rng,
        )
        val_scores = model.predict(X_val)
        val_loss = mse(t_val, val_scores)
        elapsed_s_total += time.perf_counter() - epoch_start
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        elapsed_s_cumulative.append(float(elapsed_s_total))

        should_stop = False
        if early_stopping is not None:
            should_stop = early_stopping(val_loss, model.get_params(), epoch=epoch)
            if should_stop:
                stopped_at = epoch

        if epoch % checkpoint.interval_epochs == 0 or epoch == epochs or should_stop:
            state = _checkpoint_state(
                model, optimizer, rng, epoch, train_losses, val_losses,
                elapsed_s_total, elapsed_s_cumulative, stopped_at, early_stopping,
            )
            checkpoint.save_latest(state)
            best_state = copy.deepcopy(state)
            if early_stopping is not None and early_stopping.best_params is not None:
                best_state["model_params"] = copy.deepcopy(early_stopping.best_params)
                best_state["epoch"] = int(early_stopping.best_epoch)
            checkpoint.save_best(best_state)

        if should_stop:
            break

    if early_stopping is not None and early_stopping.best_params is not None:
        model.set_params(early_stopping.best_params)

    return model, train_losses, val_losses, stopped_at, elapsed_s_total, elapsed_s_cumulative


def fit_model_fixed_epochs(cfg, model_builder, X_train, t_train, seed, epochs, checkpoint_dir, run_key):
    batch_size = effective_batch_size(cfg["training"].get("batch_size", 32))
    shuffle = cfg["training"].get("shuffle", True)
    checkpoint = CheckpointManager(checkpoint_dir, run_key, **checkpoint_cfg(cfg))

    model = model_builder(cfg, X_train.shape[1], seed)
    optimizer = build_optimizer(cfg["training"])
    rng = np.random.default_rng(seed)

    train_losses = []
    elapsed_s_cumulative = []
    elapsed_s_total = 0.0
    start_epoch = 1

    state = checkpoint.load_latest()
    if state is not None:
        _restore_checkpoint(model, optimizer, rng, None, state)
        train_losses = list(state.get("train_losses", []))
        elapsed_s_total = float(state.get("elapsed_s_total", 0.0))
        elapsed_s_cumulative = list(state.get("elapsed_s_cumulative", []))
        start_epoch = int(state.get("epoch", 0)) + 1

    if start_epoch > epochs:
        return model, train_losses, elapsed_s_total, elapsed_s_cumulative

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _ = model.train_epoch(
            X_train, t_train, optimizer, batch_size=batch_size, shuffle=shuffle, rng=rng,
        )
        elapsed_s_total += time.perf_counter() - epoch_start
        train_losses.append(float(train_loss))
        elapsed_s_cumulative.append(float(elapsed_s_total))

        if epoch % checkpoint.interval_epochs == 0 or epoch == epochs:
            state = _checkpoint_state(
                model, optimizer, rng, epoch, train_losses, [],
                elapsed_s_total, elapsed_s_cumulative, epochs, None,
            )
            checkpoint.save_latest(state)
            checkpoint.save_best(copy.deepcopy(state))

    return model, train_losses, elapsed_s_total, elapsed_s_cumulative


def threshold_grid(y_scores, n_points=300):
    y_scores = np.asarray(y_scores, dtype=float)
    unique = np.sort(np.unique(y_scores))
    lo = max(0.0, float(y_scores.min()))
    hi = min(1.0, float(y_scores.max()))
    if np.isclose(lo, hi):
        return np.array([lo], dtype=float)
    linspace = np.linspace(lo, hi, n_points)
    return np.sort(np.unique(np.concatenate([unique, linspace])))


def select_threshold_by_f2(y_true, y_scores, beta=2.0):
    thresholds = threshold_grid(y_scores)
    precisions, recalls, f2s = [], [], []
    for threshold in thresholds:
        pred = (y_scores >= threshold).astype(int)
        precision, recall, f2 = precision_recall_fbeta(y_true, pred, beta=beta)
        precisions.append(precision)
        recalls.append(recall)
        f2s.append(f2)
    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)
    f2s = np.asarray(f2s, dtype=float)
    order = np.lexsort((thresholds, -precisions, -recalls, -f2s))
    return float(thresholds[int(order[0])])


def evaluate_binary_scores(y_true, y_scores, threshold, targets=None, beta=2.0):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    pred = (y_scores >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, pred)
    precision, recall, f1 = precision_recall_f1(y_true, pred)
    _, _, f2 = precision_recall_fbeta(y_true, pred, beta=beta)
    precs, recs = pr_curve(y_true, y_scores)
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tn = int(cm[0, 0])
    cost = int(2 * fn + fp)
    result = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "auc_pr": float(auc(recs, precs)),
        "cost_fn2_fp1": cost,
        "cost_mean_fn2_fp1": float(cost / len(y_true)) if len(y_true) > 0 else 0.0,
        "pred": pred,
        "scores": y_scores,
        "mse": None if targets is None else float(mse(np.asarray(targets, dtype=float), y_scores)),
    }
    return result


def aggregate_seed_summary(raw_df, filters):
    mask = np.ones(len(raw_df), dtype=bool)
    for key, value in filters.items():
        mask &= raw_df[key].astype(str) == str(value)

    val_rows = raw_df[
        mask
        & (raw_df["record_type"] == "split_summary")
        & (raw_df["subset"] == "val")
    ].copy()
    train_rows = raw_df[
        mask
        & (raw_df["record_type"] == "split_summary")
        & (raw_df["subset"] == "train")
        & (raw_df["split_kind"] != "final_retrain")
    ].copy()
    test_rows = raw_df[
        mask
        & (raw_df["record_type"] == "split_summary")
        & (raw_df["split_kind"] == "final_retrain")
        & (raw_df["subset"] == "test")
    ].copy()
    if val_rows.empty or test_rows.empty:
        return None

    test_row = test_rows.iloc[-1]
    elapsed_total = float(val_rows["elapsed_s"].fillna(0.0).astype(float).sum()) + float(test_row["elapsed_s"])
    return {
        "mean_val_auc_pr": float(val_rows["auc_pr"].astype(float).mean()),
        "std_val_auc_pr": float(val_rows["auc_pr"].astype(float).std(ddof=0)),
        "mean_val_accuracy": float(val_rows["accuracy"].astype(float).mean()),
        "std_val_accuracy": float(val_rows["accuracy"].astype(float).std(ddof=0)),
        "mean_val_precision": float(val_rows["precision"].astype(float).mean()),
        "std_val_precision": float(val_rows["precision"].astype(float).std(ddof=0)),
        "mean_val_recall": float(val_rows["recall"].astype(float).mean()),
        "std_val_recall": float(val_rows["recall"].astype(float).std(ddof=0)),
        "mean_val_f1": float(val_rows["f1"].astype(float).mean()),
        "std_val_f1": float(val_rows["f1"].astype(float).std(ddof=0)),
        "mean_val_f2": float(val_rows["f2"].astype(float).mean()),
        "std_val_f2": float(val_rows["f2"].astype(float).std(ddof=0)),
        "mean_val_cost": float(val_rows["cost_mean_fn2_fp1"].astype(float).mean()),
        "std_val_cost": float(val_rows["cost_mean_fn2_fp1"].astype(float).std(ddof=0)),
        "mean_train_mse": float(train_rows["mse"].astype(float).mean()),
        "std_train_mse": float(train_rows["mse"].astype(float).std(ddof=0)),
        "mean_val_mse": float(val_rows["mse"].astype(float).mean()),
        "std_val_mse": float(val_rows["mse"].astype(float).std(ddof=0)),
        "gap_mse": float(val_rows["mse"].astype(float).mean() - train_rows["mse"].astype(float).mean()),
        "mean_threshold": float(val_rows["threshold_selected"].astype(float).mean()),
        "std_threshold": float(val_rows["threshold_selected"].astype(float).std(ddof=0)),
        "mean_stopped_at": float(val_rows["stopped_at"].astype(float).mean()),
        "test_auc_pr": float(test_row["auc_pr"]),
        "test_accuracy": float(test_row["accuracy"]),
        "test_precision": float(test_row["precision"]),
        "test_recall": float(test_row["recall"]),
        "test_f1": float(test_row["f1"]),
        "test_f2": float(test_row["f2"]),
        "test_cost": float(test_row["cost_mean_fn2_fp1"]),
        "test_mse": float(test_row["mse"]),
        "elapsed_s_total": elapsed_total,
    }


def base_row(cfg, experiment_type, seed, train_size, val_size, test_size,
             config_name="", candidate_name="", strategy="",
             split_kind="", split_id="", subset=""):
    return {
        "experiment_type": experiment_type,
        "config_name": config_name,
        "candidate_name": candidate_name,
        "strategy": strategy,
        "scaler": cfg["data"]["preprocess"]["feature_scaler"],
        "optimizer": cfg["training"]["optimizer"],
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size": cfg["training"]["batch_size"],
        "batch_label": batch_label(cfg["training"]["batch_size"]),
        "momentum": cfg["training"].get("momentum", 0.9),
        "seed": seed,
        "activation": cfg["model"]["activation"],
        "beta": cfg["model"].get("beta", 1.0),
        "split_kind": split_kind,
        "split_id": split_id,
        "subset": subset,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "epochs_planned": cfg["training"]["epochs"],
    }


def curve_rows(base, train_losses, val_losses=None, elapsed_s_cumulative=None, stopped_at=None):
    rows = []
    val_losses = [] if val_losses is None else list(val_losses)
    elapsed_s_cumulative = [] if elapsed_s_cumulative is None else list(elapsed_s_cumulative)
    for idx, train_loss in enumerate(train_losses, start=1):
        row = dict(base)
        row.update({
            "record_type": "curve_point",
            "epoch": idx,
            "train_mse": float(train_loss),
            "val_mse": "" if idx > len(val_losses) else float(val_losses[idx - 1]),
            "elapsed_s_cumulative": "" if idx > len(elapsed_s_cumulative) else float(elapsed_s_cumulative[idx - 1]),
            "stopped_at": "" if stopped_at is None else int(stopped_at),
        })
        rows.append(row)
    return rows


def split_summary_row(base, metrics, threshold_selected, threshold_rule, stopped_at, elapsed_s=""):
    row = dict(base)
    row.update({
        "record_type": "split_summary",
        "threshold_selected": float(threshold_selected),
        "threshold_rule": threshold_rule,
        "stopped_at": int(stopped_at),
        "elapsed_s": elapsed_s,
        "mse": metrics["mse"],
        "auc_pr": metrics["auc_pr"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "f2": metrics["f2"],
        "cost_fn2_fp1": metrics["cost_fn2_fp1"],
        "cost_mean_fn2_fp1": metrics["cost_mean_fn2_fp1"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    })
    return row


def sample_output_rows(base, y_true, targets, scores, pre_activations, pred, threshold_selected, threshold_rule):
    rows = []
    for idx, (label, target, score, pre_activation, pred_value) in enumerate(
        zip(y_true, targets, scores, pre_activations, pred)
    ):
        row = dict(base)
        row.update({
            "record_type": "sample_output",
            "sample_idx": idx,
            "threshold_selected": float(threshold_selected),
            "threshold_rule": threshold_rule,
            "y_true": int(label),
            "target_bigmodel": float(target),
            "score": float(score),
            "pre_activation": float(pre_activation),
            "pred": int(pred_value),
        })
        rows.append(row)
    return rows
