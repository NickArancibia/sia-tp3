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
from shared.metrics import auc, confusion_matrix, pr_curve, precision_recall_f1, threshold_sweep
from shared.optimizers import build_optimizer
from shared.preprocessing import build_scaler, stratified_kfold_indices, stratified_split
from shared.regularization import EarlyStopping


RAW_FIELDNAMES = [
    "record_type", "candidate_name", "strategy", "scaler", "optimizer", "learning_rate", "batch_size",
    "batch_label", "momentum", "seed", "split_kind", "split_id", "epoch", "threshold", "sample_idx",
    "train_size", "val_size", "threshold_selected", "train_mse", "val_mse", "train_auc_pr", "val_auc_pr",
    "train_precision", "train_recall", "train_f1", "val_precision", "val_recall", "val_f1",
    "generalization_gap_auc_pr", "generalization_gap_f1", "generalization_gap_mse", "stopped_at", "y_true",
    "target_bigmodel", "score", "pre_activation", "pred_optimal", "tn", "fp", "fn", "tp",
]

SUMMARY_FIELDNAMES = [
    "candidate_name", "strategy", "scaler", "optimizer", "learning_rate", "batch_size", "batch_label",
    "momentum", "seed", "threshold", "epochs", "train_auc_pr", "test_auc_pr", "test_precision",
    "test_recall", "test_f1", "train_mse", "test_mse",
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
    thresholds, th_precs, th_recs, th_f1s, best_t = threshold_sweep(y_val, val_scores)
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
        "thresholds": thresholds,
        "threshold_precisions": th_precs,
        "threshold_recalls": th_recs,
        "threshold_f1s": th_f1s,
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


def aggregate_threshold_curves(run_results, n_points=300):
    lo = min(float(run["thresholds"][0]) for run in run_results)
    hi = max(float(run["thresholds"][-1]) for run in run_results)
    grid = np.array([lo]) if np.isclose(lo, hi) else np.linspace(lo, hi, n_points)
    prec_curves = []
    rec_curves = []
    f1_curves = []
    for run in run_results:
        prec_curves.append(np.interp(grid, run["thresholds"], run["threshold_precisions"]))
        rec_curves.append(np.interp(grid, run["thresholds"], run["threshold_recalls"]))
        f1_curves.append(np.interp(grid, run["thresholds"], run["threshold_f1s"]))
    prec_arr = np.asarray(prec_curves)
    rec_arr = np.asarray(rec_curves)
    f1_arr = np.asarray(f1_curves)
    return {
        "thresholds": grid,
        "precisions_mean": prec_arr.mean(axis=0),
        "precisions_std": prec_arr.std(axis=0),
        "recalls_mean": rec_arr.mean(axis=0),
        "recalls_std": rec_arr.std(axis=0),
        "f1s_mean": f1_arr.mean(axis=0),
        "f1s_std": f1_arr.std(axis=0),
    }


def summarize_seed(run_results):
    mean_val_auc_pr = float(np.mean([run["val_auc_pr"] for run in run_results]))
    std_val_auc_pr = float(np.std([run["val_auc_pr"] for run in run_results]))
    mean_val_recall = float(np.mean([run["val_recall"] for run in run_results]))
    mean_threshold = float(np.mean([run["threshold"] for run in run_results]))
    std_threshold = float(np.std([run["threshold"] for run in run_results]))
    mean_stopped_at = float(np.mean([run["stopped_at"] for run in run_results]))
    return {
        "mean_val_auc_pr": mean_val_auc_pr,
        "std_val_auc_pr": std_val_auc_pr,
        "mean_val_recall": mean_val_recall,
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "mean_stopped_at": mean_stopped_at,
        "threshold_curves": aggregate_threshold_curves(run_results),
    }


def raw_rows_for_seed(cfg, candidate_name, seed, run):
    momentum = cfg["training"].get("momentum", 0.9)
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
    rows = [{"record_type": "seed_metric", "epoch": "", "threshold": "", "sample_idx": "", "y_true": "", "target_bigmodel": "", "score": "", "pre_activation": "", "pred_optimal": "", "tn": "", "fp": "", "fn": "", "tp": "", **base}]
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
            "threshold": "",
            "sample_idx": "",
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
            "y_true": "",
            "target_bigmodel": "",
            "score": "",
            "pre_activation": "",
            "pred_optimal": "",
            "tn": "",
            "fp": "",
            "fn": "",
            "tp": "",
        })
    return rows


def retrain_final_model(X, t, y, train_val_idx, test_idx, cfg, seed, threshold, epochs):
    X_train = X[train_val_idx]
    X_test = X[test_idx]
    t_train = t[train_val_idx]
    t_test = t[test_idx]
    y_train = y[train_val_idx]
    y_test = y[test_idx]
    scaler = build_scaler(cfg["data"]["preprocess"]["feature_scaler"])
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = _make_perceptron(cfg, X_train.shape[1], seed)
    optimizer = build_optimizer(cfg["training"])
    batch_size = effective_batch_size(cfg["training"].get("batch_size", 32))
    shuffle = cfg["training"].get("shuffle", True)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        model.train_epoch(X_train, t_train, optimizer, batch_size=batch_size, shuffle=shuffle, rng=rng)

    train_scores = model.predict(X_train)
    test_scores = model.predict(X_test)
    test_preacts = model.pre_activation(X_test)
    test_pred = (test_scores >= threshold).astype(int)
    train_precs, train_recs = pr_curve(y_train, train_scores)
    test_precs, test_recs = pr_curve(y_test, test_scores)
    precision, recall, f1 = precision_recall_f1(y_test, test_pred)

    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "test_preacts": test_preacts,
        "test_pred": test_pred,
        "y_test": y_test,
        "t_test": t_test,
        "threshold": float(threshold),
        "epochs": int(epochs),
        "train_auc_pr": auc(train_recs, train_precs),
        "test_auc_pr": auc(test_recs, test_precs),
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "train_mse": float(mse(t_train, train_scores)),
        "test_mse": float(mse(t_test, test_scores)),
    }


def confusion_rows(y_true, scores, step=0.025):
    thresholds = np.arange(0.0, 1.0001, step)
    rows = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred)
        precision, recall, f1 = precision_recall_f1(y_true, pred)
        rows.append({
            "threshold": round(float(threshold), 6),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def best_row(df):
    return df.sort_values(["mean_val_auc_pr", "std_val_auc_pr", "mean_val_recall"], ascending=[False, True, False]).iloc[0]


def main():
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "selected_model")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "selected_model_raw.csv")
    summary_path = os.path.join(out_dir, "selected_model_summary.csv")
    for path in (raw_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    optimizer_summary = pd.read_csv(os.path.join(EJ1_DIR, "results", "part2", "optimizer", "optimizer_summary.csv"))
    selected = best_row(optimizer_summary)

    run_cfg = copy_cfg(
        cfg,
        scaler_name=selected["scaler"],
        training_overrides={
            "optimizer": selected["optimizer"],
            "learning_rate": float(selected["learning_rate"]),
            "batch_size": int(selected["batch_size"]),
            "momentum": float(selected.get("momentum", 0.9)),
        },
    )

    X, t, y, _ = load_data(cfg)
    split_seed = cfg["data"]["split"].get("seed", 42)
    test_frac = cfg["data"]["split"].get("test_frac", 0.15)
    train_val_idx, _, test_idx = stratified_split(y, val_frac=0.0, test_frac=test_frac, seed=split_seed)
    split_specs = build_s3_splits(train_val_idx, y, cfg["data"]["split"])

    print("\n" + "=" * 60)
    print("PART 2 — SELECTED MODEL")
    print("=" * 60)

    seed_summaries = []
    for seed in cfg["experiment"].get("seeds", [cfg["experiment"].get("seed", 42)]):
        run_results = []
        for split_spec in split_specs:
            run = evaluate_split(run_cfg, X, t, y, split_spec, seed)
            run_results.append(run)
            append_rows_csv(raw_path, raw_rows_for_seed(run_cfg, selected["candidate_name"], seed, run), RAW_FIELDNAMES)
        seed_summary = summarize_seed(run_results)
        seed_summary["seed"] = seed
        seed_summaries.append(seed_summary)
        print(f"  seed={seed}: AUC-PR val={seed_summary['mean_val_auc_pr']:.4f} ± {seed_summary['std_val_auc_pr']:.4f}")

    seed_summaries_df = pd.DataFrame(seed_summaries)
    selected_seed = best_row(seed_summaries_df)
    final_epochs = max(1, int(round(selected_seed["mean_stopped_at"])))

    for threshold, precision_mean, precision_std, recall_mean, recall_std, f1_mean, f1_std in zip(
        selected_seed["threshold_curves"]["thresholds"],
        selected_seed["threshold_curves"]["precisions_mean"],
        selected_seed["threshold_curves"]["precisions_std"],
        selected_seed["threshold_curves"]["recalls_mean"],
        selected_seed["threshold_curves"]["recalls_std"],
        selected_seed["threshold_curves"]["f1s_mean"],
        selected_seed["threshold_curves"]["f1s_std"],
    ):
        append_rows_csv(raw_path, [{
            "record_type": "threshold_point",
            "candidate_name": selected["candidate_name"],
            "strategy": "S3",
            "scaler": run_cfg["data"]["preprocess"]["feature_scaler"],
            "optimizer": run_cfg["training"]["optimizer"],
            "learning_rate": run_cfg["training"]["learning_rate"],
            "batch_size": run_cfg["training"]["batch_size"],
            "batch_label": batch_label(run_cfg["training"]["batch_size"]),
            "momentum": run_cfg["training"].get("momentum", 0.9),
            "seed": int(selected_seed["seed"]),
            "split_kind": "",
            "split_id": "",
            "epoch": "",
            "threshold": threshold,
            "sample_idx": "",
            "train_size": len(train_val_idx),
            "val_size": len(test_idx),
            "threshold_selected": float(selected_seed["mean_threshold"]),
            "train_mse": "",
            "val_mse": "",
            "train_auc_pr": "",
            "val_auc_pr": "",
            "train_precision": "",
            "train_recall": "",
            "train_f1": "",
            "val_precision": precision_mean,
            "val_recall": recall_mean,
            "val_f1": f1_mean,
            "generalization_gap_auc_pr": precision_std,
            "generalization_gap_f1": recall_std,
            "generalization_gap_mse": f1_std,
            "stopped_at": final_epochs,
            "y_true": "",
            "target_bigmodel": "",
            "score": "",
            "pre_activation": "",
            "pred_optimal": "",
            "tn": "",
            "fp": "",
            "fn": "",
            "tp": "",
        }], RAW_FIELDNAMES)

    final_result = retrain_final_model(
        X, t, y, train_val_idx, test_idx, run_cfg, int(selected_seed["seed"]), float(selected_seed["mean_threshold"]), final_epochs
    )

    test_rows = []
    for idx, (y_true, target, score, pre_activation, pred) in enumerate(
        zip(final_result["y_test"], final_result["t_test"], final_result["test_scores"], final_result["test_preacts"], final_result["test_pred"])
    ):
        test_rows.append({
            "record_type": "test_output",
            "candidate_name": selected["candidate_name"],
            "strategy": "S3",
            "scaler": run_cfg["data"]["preprocess"]["feature_scaler"],
            "optimizer": run_cfg["training"]["optimizer"],
            "learning_rate": run_cfg["training"]["learning_rate"],
            "batch_size": run_cfg["training"]["batch_size"],
            "batch_label": batch_label(run_cfg["training"]["batch_size"]),
            "momentum": run_cfg["training"].get("momentum", 0.9),
            "seed": int(selected_seed["seed"]),
            "split_kind": "test",
            "split_id": 0,
            "epoch": "",
            "threshold": float(selected_seed["mean_threshold"]),
            "sample_idx": idx,
            "train_size": len(train_val_idx),
            "val_size": len(test_idx),
            "threshold_selected": float(selected_seed["mean_threshold"]),
            "train_mse": "",
            "val_mse": "",
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
            "stopped_at": final_epochs,
            "y_true": int(y_true),
            "target_bigmodel": float(target),
            "score": float(score),
            "pre_activation": float(pre_activation),
            "pred_optimal": int(pred),
            "tn": "",
            "fp": "",
            "fn": "",
            "tp": "",
        })
    append_rows_csv(raw_path, test_rows, RAW_FIELDNAMES)

    confusion_rows_list = []
    for row in confusion_rows(final_result["y_test"], final_result["test_scores"], step=0.025):
        confusion_rows_list.append({
            "record_type": "confusion_point",
            "candidate_name": selected["candidate_name"],
            "strategy": "S3",
            "scaler": run_cfg["data"]["preprocess"]["feature_scaler"],
            "optimizer": run_cfg["training"]["optimizer"],
            "learning_rate": run_cfg["training"]["learning_rate"],
            "batch_size": run_cfg["training"]["batch_size"],
            "batch_label": batch_label(run_cfg["training"]["batch_size"]),
            "momentum": run_cfg["training"].get("momentum", 0.9),
            "seed": int(selected_seed["seed"]),
            "split_kind": "test",
            "split_id": 0,
            "epoch": "",
            "threshold": row["threshold"],
            "sample_idx": "",
            "train_size": len(train_val_idx),
            "val_size": len(test_idx),
            "threshold_selected": float(selected_seed["mean_threshold"]),
            "train_mse": "",
            "val_mse": "",
            "train_auc_pr": "",
            "val_auc_pr": "",
            "train_precision": "",
            "train_recall": "",
            "train_f1": "",
            "val_precision": row["precision"],
            "val_recall": row["recall"],
            "val_f1": row["f1"],
            "generalization_gap_auc_pr": "",
            "generalization_gap_f1": "",
            "generalization_gap_mse": "",
            "stopped_at": final_epochs,
            "y_true": "",
            "target_bigmodel": "",
            "score": "",
            "pre_activation": "",
            "pred_optimal": "",
            "tn": row["tn"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tp": row["tp"],
        })
    append_rows_csv(raw_path, confusion_rows_list, RAW_FIELDNAMES)

    summary = pd.DataFrame([{
        "candidate_name": selected["candidate_name"],
        "strategy": "S3",
        "scaler": run_cfg["data"]["preprocess"]["feature_scaler"],
        "optimizer": run_cfg["training"]["optimizer"],
        "learning_rate": run_cfg["training"]["learning_rate"],
        "batch_size": run_cfg["training"]["batch_size"],
        "batch_label": batch_label(run_cfg["training"]["batch_size"]),
        "momentum": run_cfg["training"].get("momentum", 0.9),
        "seed": int(selected_seed["seed"]),
        "threshold": float(selected_seed["mean_threshold"]),
        "epochs": final_epochs,
        "train_auc_pr": final_result["train_auc_pr"],
        "test_auc_pr": final_result["test_auc_pr"],
        "test_precision": final_result["test_precision"],
        "test_recall": final_result["test_recall"],
        "test_f1": final_result["test_f1"],
        "train_mse": final_result["train_mse"],
        "test_mse": final_result["test_mse"],
    }], columns=SUMMARY_FIELDNAMES)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("PART 2 — SELECTED MODEL")
    print("=" * 60)
    print(f"  Candidate: {selected['candidate_name']}")
    print(f"  Optimizer: {run_cfg['training']['optimizer']}")
    print(f"  Batch:     {batch_label(run_cfg['training']['batch_size'])}")
    print(f"  LR:        {run_cfg['training']['learning_rate']}")
    print(f"  Seed:      {int(selected_seed['seed'])}")
    print(f"  Threshold: {float(selected_seed['mean_threshold']):.4f}")
    print(f"  AUC-PR:    {final_result['test_auc_pr']:.4f}")
    print(f"  F1:        {final_result['test_f1']:.4f}")


if __name__ == "__main__":
    main()
