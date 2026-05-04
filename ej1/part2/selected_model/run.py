import copy
import os
import sys
import time

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PART2_DIR)

import numpy as np
import pandas as pd

from common import (
    RAW_FIELDNAMES,
    append_rows_csv as append_standard_rows_csv,
    base_row,
    curve_rows,
    evaluate_binary_scores,
    sample_output_rows,
    split_summary_row,
)
from main_part2 import _make_perceptron, load_data
from shared.config_loader import load_config
from shared.losses import mse
from shared.metrics import confusion_matrix, precision_recall_f1, precision_recall_fbeta
from shared.optimizers import build_optimizer
from shared.preprocessing import build_scaler, stratified_kfold_indices, stratified_split
from shared.regularization import EarlyStopping


SUMMARY_FIELDNAMES = [
    "candidate_name", "strategy", "scaler", "optimizer", "learning_rate", "batch_size", "batch_label",
    "momentum", "seed", "threshold", "threshold_selection_rule", "epochs",
    "mean_val_f2", "std_val_f2", "mean_val_cost", "std_val_cost",
    "train_auc_pr", "test_auc_pr", "test_precision", "test_recall", "test_f1", "test_f2",
    "train_mse", "test_mse", "train_cost", "test_cost", "train_cost_mean", "test_cost_mean",
]

THRESHOLD_MIN = 0.75
THRESHOLD_MAX = 1.0
THRESHOLD_STEP = 0.025
THRESHOLD_GRID = np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP), 3)
THRESHOLD_SELECTION_RULE = "min(2*FN + FP)"
EXPERIMENT_TYPE = "selected_model"
SELECTED_STRATEGY = "S3"
SELECTED_OPTIMIZER = "momentum"
SELECTED_BATCH_SIZE = 16
SELECTED_LEARNING_RATE = 1e-4


def selected_candidate_name():
    return f"{SELECTED_OPTIMIZER}_b{SELECTED_BATCH_SIZE}_lr{SELECTED_LEARNING_RATE:.0e}"


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


def fixed_threshold_grid():
    return THRESHOLD_GRID.copy()


def evaluate_threshold_metrics(y_true, y_scores, threshold, beta=2.0):
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    pred = (y_scores >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, pred)
    precision, recall, f1 = precision_recall_f1(y_true, pred)
    _, _, f2 = precision_recall_fbeta(y_true, pred, beta=beta)
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    cost = int(2 * fn + fp)
    return {
        "pred": pred,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cost_fn2_fp1": cost,
        "cost_mean_fn2_fp1": float(cost / len(y_true)) if len(y_true) > 0 else 0.0,
    }


def select_threshold_idx(thresholds, costs, f2s, precisions, recalls):
    thresholds = np.asarray(thresholds, dtype=float)
    costs = np.asarray(costs, dtype=float)
    f2s = np.asarray(f2s, dtype=float)
    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)
    if len(thresholds) == 0:
        raise ValueError("La grilla de thresholds no puede estar vacia")
    order = np.lexsort((thresholds, -precisions, -recalls, -f2s, costs))
    return int(order[0])


def sweep_selected_thresholds(y_true, y_scores):
    thresholds = fixed_threshold_grid()
    precisions, recalls, f1s, f2s, costs = [], [], [], [], []
    for threshold in thresholds:
        metrics = evaluate_threshold_metrics(y_true, y_scores, threshold, beta=2.0)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])
        f2s.append(metrics["f2"])
        costs.append(metrics["cost_fn2_fp1"])

    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)
    f1s = np.asarray(f1s, dtype=float)
    f2s = np.asarray(f2s, dtype=float)
    costs = np.asarray(costs, dtype=float)
    best_idx = select_threshold_idx(thresholds, costs, f2s, precisions, recalls)
    return thresholds, precisions, recalls, f1s, f2s, costs, float(thresholds[best_idx])


def selected_threshold(thresholds, costs, f2s, precisions, recalls):
    best_idx = select_threshold_idx(thresholds, costs, f2s, precisions, recalls)
    return float(np.asarray(thresholds, dtype=float)[best_idx])


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
    elapsed_curve = []
    elapsed_s_total = 0.0
    stopped_at = epochs
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _ = model.train_epoch(
            X_train, t_train, optimizer, batch_size=batch_size, shuffle=shuffle, rng=rng
        )
        val_scores = model.predict(X_val)
        val_loss = mse(t_val, val_scores)
        elapsed_s_total += time.perf_counter() - epoch_start
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        elapsed_curve.append(float(elapsed_s_total))
        if early_stopping is not None and early_stopping(val_loss, model.get_params()):
            stopped_at = epoch
            break

    if early_stopping is not None and early_stopping.best_params is not None:
        model.set_params(early_stopping.best_params)

    return model, train_losses, val_losses, stopped_at, elapsed_s_total, elapsed_curve


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

    model, train_losses, val_losses, stopped_at, elapsed_s, elapsed_curve = fit_split_model(
        cfg, X_train, t_train, X_val, t_val, seed,
    )
    train_scores = model.predict(X_train)
    val_scores = model.predict(X_val)
    val_preacts = model.pre_activation(X_val)
    thresholds, th_precs, th_recs, th_f1s, th_f2s, th_costs, best_t = sweep_selected_thresholds(y_val, val_scores)
    train_metrics = evaluate_binary_scores(y_train, train_scores, best_t, targets=t_train, beta=2.0)
    val_metrics = evaluate_binary_scores(y_val, val_scores, best_t, targets=t_val, beta=2.0)

    return {
        "split_kind": split_spec["split_kind"],
        "split_id": split_spec["split_id"],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "elapsed_s": float(elapsed_s),
        "elapsed_curve": elapsed_curve,
        "thresholds": thresholds,
        "threshold_precisions": th_precs,
        "threshold_recalls": th_recs,
        "threshold_f1s": th_f1s,
        "threshold_f2s": th_f2s,
        "threshold_costs": th_costs,
        "train_auc_pr": train_metrics["auc_pr"],
        "val_auc_pr": val_metrics["auc_pr"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "train_f2": train_metrics["f2"],
        "train_cost": train_metrics["cost_fn2_fp1"],
        "train_cost_mean": train_metrics["cost_mean_fn2_fp1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_f2": val_metrics["f2"],
        "val_cost": val_metrics["cost_fn2_fp1"],
        "val_cost_mean": val_metrics["cost_mean_fn2_fp1"],
        "threshold": float(best_t),
        "train_mse": train_metrics["mse"],
        "val_mse": val_metrics["mse"],
        "stopped_at": stopped_at,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "y_val": y_val,
        "t_val": t_val,
        "val_scores": val_scores,
        "val_preacts": val_preacts,
    }


def aggregate_threshold_curves(run_results):
    grid = np.asarray(run_results[0]["thresholds"], dtype=float)
    prec_curves = [np.asarray(run_results[0]["threshold_precisions"], dtype=float)]
    rec_curves = [np.asarray(run_results[0]["threshold_recalls"], dtype=float)]
    f1_curves = [np.asarray(run_results[0]["threshold_f1s"], dtype=float)]
    f2_curves = [np.asarray(run_results[0]["threshold_f2s"], dtype=float)]
    cost_curves = [np.asarray(run_results[0]["threshold_costs"], dtype=float)]
    for run in run_results[1:]:
        if not np.allclose(run["thresholds"], grid):
            raise ValueError("Todos los barridos de threshold deben usar la misma grilla fija")
        prec_curves.append(np.asarray(run["threshold_precisions"], dtype=float))
        rec_curves.append(np.asarray(run["threshold_recalls"], dtype=float))
        f1_curves.append(np.asarray(run["threshold_f1s"], dtype=float))
        f2_curves.append(np.asarray(run["threshold_f2s"], dtype=float))
        cost_curves.append(np.asarray(run["threshold_costs"], dtype=float))
    prec_arr = np.asarray(prec_curves, dtype=float)
    rec_arr = np.asarray(rec_curves, dtype=float)
    f1_arr = np.asarray(f1_curves, dtype=float)
    f2_arr = np.asarray(f2_curves, dtype=float)
    cost_arr = np.asarray(cost_curves, dtype=float)
    return {
        "thresholds": grid,
        "precisions_mean": prec_arr.mean(axis=0),
        "precisions_std": prec_arr.std(axis=0),
        "recalls_mean": rec_arr.mean(axis=0),
        "recalls_std": rec_arr.std(axis=0),
        "f1s_mean": f1_arr.mean(axis=0),
        "f1s_std": f1_arr.std(axis=0),
        "f2s_mean": f2_arr.mean(axis=0),
        "f2s_std": f2_arr.std(axis=0),
        "costs_mean": cost_arr.mean(axis=0),
        "costs_std": cost_arr.std(axis=0),
    }


def summarize_seed(run_results):
    mean_val_auc_pr = float(np.mean([run["val_auc_pr"] for run in run_results]))
    std_val_auc_pr = float(np.std([run["val_auc_pr"] for run in run_results]))
    mean_val_recall = float(np.mean([run["val_recall"] for run in run_results]))
    mean_val_f2 = float(np.mean([run["val_f2"] for run in run_results]))
    std_val_f2 = float(np.std([run["val_f2"] for run in run_results]))
    mean_val_cost = float(np.mean([run["val_cost_mean"] for run in run_results]))
    std_val_cost = float(np.std([run["val_cost_mean"] for run in run_results]))
    mean_threshold = float(np.mean([run["threshold"] for run in run_results]))
    std_threshold = float(np.std([run["threshold"] for run in run_results]))
    mean_stopped_at = float(np.mean([run["stopped_at"] for run in run_results]))
    threshold_curves = aggregate_threshold_curves(run_results)
    return {
        "mean_val_auc_pr": mean_val_auc_pr,
        "std_val_auc_pr": std_val_auc_pr,
        "mean_val_recall": mean_val_recall,
        "mean_val_f2": mean_val_f2,
        "std_val_f2": std_val_f2,
        "mean_val_cost": mean_val_cost,
        "std_val_cost": std_val_cost,
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "mean_stopped_at": mean_stopped_at,
        "selected_threshold": selected_threshold(
            threshold_curves["thresholds"],
            threshold_curves["costs_mean"],
            threshold_curves["f2s_mean"],
            threshold_curves["precisions_mean"],
            threshold_curves["recalls_mean"],
        ),
        "threshold_curves": threshold_curves,
    }


def raw_rows_for_split(cfg, candidate_name, seed, test_size, run):
    curve_base = base_row(
        cfg,
        EXPERIMENT_TYPE,
        seed,
        len(run["train_idx"]),
        len(run["val_idx"]),
        test_size,
        candidate_name=candidate_name,
        strategy=SELECTED_STRATEGY,
        split_kind=run["split_kind"],
        split_id=run["split_id"],
    )
    train_base = base_row(
        cfg,
        EXPERIMENT_TYPE,
        seed,
        len(run["train_idx"]),
        len(run["val_idx"]),
        test_size,
        candidate_name=candidate_name,
        strategy=SELECTED_STRATEGY,
        split_kind=run["split_kind"],
        split_id=run["split_id"],
        subset="train",
    )
    val_base = base_row(
        cfg,
        EXPERIMENT_TYPE,
        seed,
        len(run["train_idx"]),
        len(run["val_idx"]),
        test_size,
        candidate_name=candidate_name,
        strategy=SELECTED_STRATEGY,
        split_kind=run["split_kind"],
        split_id=run["split_id"],
        subset="val",
    )

    rows = curve_rows(
        curve_base,
        run["train_losses"],
        run["val_losses"],
        run["elapsed_curve"],
        stopped_at=run["stopped_at"],
    )
    rows.append(split_summary_row(train_base, run["train_metrics"], run["threshold"], THRESHOLD_SELECTION_RULE, run["stopped_at"]))
    rows.append(
        split_summary_row(
            val_base,
            run["val_metrics"],
            run["threshold"],
            THRESHOLD_SELECTION_RULE,
            run["stopped_at"],
            elapsed_s=run["elapsed_s"],
        )
    )
    rows.extend(
        sample_output_rows(
            val_base,
            run["y_val"],
            run["t_val"],
            run["val_scores"],
            run["val_preacts"],
            run["val_metrics"]["pred"],
            run["threshold"],
            THRESHOLD_SELECTION_RULE,
        )
    )
    return rows


def final_raw_rows(cfg, candidate_name, seed, train_val_size, test_size, final_result):
    curve_base = base_row(
        cfg,
        EXPERIMENT_TYPE,
        seed,
        train_val_size,
        0,
        test_size,
        candidate_name=candidate_name,
        strategy=SELECTED_STRATEGY,
        split_kind="final_retrain",
        split_id=0,
    )
    train_base = base_row(
        cfg,
        EXPERIMENT_TYPE,
        seed,
        train_val_size,
        0,
        test_size,
        candidate_name=candidate_name,
        strategy=SELECTED_STRATEGY,
        split_kind="final_retrain",
        split_id=0,
        subset="train",
    )
    test_base = base_row(
        cfg,
        EXPERIMENT_TYPE,
        seed,
        train_val_size,
        0,
        test_size,
        candidate_name=candidate_name,
        strategy=SELECTED_STRATEGY,
        split_kind="final_retrain",
        split_id=0,
        subset="test",
    )

    rows = curve_rows(
        curve_base,
        final_result["train_losses"],
        val_losses=None,
        elapsed_s_cumulative=final_result["elapsed_curve"],
        stopped_at=final_result["epochs"],
    )
    rows.append(
        split_summary_row(
            train_base,
            final_result["train_metrics"],
            final_result["threshold"],
            THRESHOLD_SELECTION_RULE,
            final_result["epochs"],
        )
    )
    rows.append(
        split_summary_row(
            test_base,
            final_result["test_metrics"],
            final_result["threshold"],
            THRESHOLD_SELECTION_RULE,
            final_result["epochs"],
            elapsed_s=final_result["elapsed_s"],
        )
    )
    rows.extend(
        sample_output_rows(
            test_base,
            final_result["y_test"],
            final_result["t_test"],
            final_result["test_scores"],
            final_result["test_preacts"],
            final_result["test_metrics"]["pred"],
            final_result["threshold"],
            THRESHOLD_SELECTION_RULE,
        )
    )
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

    train_losses = []
    elapsed_curve = []
    elapsed_s_total = 0.0
    for _ in range(epochs):
        epoch_start = time.perf_counter()
        train_loss, _ = model.train_epoch(X_train, t_train, optimizer, batch_size=batch_size, shuffle=shuffle, rng=rng)
        elapsed_s_total += time.perf_counter() - epoch_start
        train_losses.append(float(train_loss))
        elapsed_curve.append(float(elapsed_s_total))

    train_scores = model.predict(X_train)
    test_scores = model.predict(X_test)
    test_preacts = model.pre_activation(X_test)
    train_metrics = evaluate_binary_scores(y_train, train_scores, threshold, targets=t_train, beta=2.0)
    test_metrics = evaluate_binary_scores(y_test, test_scores, threshold, targets=t_test, beta=2.0)

    return {
        "train_losses": train_losses,
        "elapsed_s": float(elapsed_s_total),
        "elapsed_curve": elapsed_curve,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "test_preacts": test_preacts,
        "y_test": y_test,
        "t_test": t_test,
        "threshold": float(threshold),
        "epochs": int(epochs),
        "train_auc_pr": train_metrics["auc_pr"],
        "test_auc_pr": test_metrics["auc_pr"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_f2": test_metrics["f2"],
        "train_mse": train_metrics["mse"],
        "test_mse": test_metrics["mse"],
        "train_cost": train_metrics["cost_fn2_fp1"],
        "test_cost": test_metrics["cost_fn2_fp1"],
        "train_cost_mean": train_metrics["cost_mean_fn2_fp1"],
        "test_cost_mean": test_metrics["cost_mean_fn2_fp1"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def selected_seed_row(df):
    return df.sort_values(["mean_val_cost", "std_val_cost", "mean_val_f2", "mean_val_recall"], ascending=[True, True, False, False]).iloc[0]


def main():
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))
    out_dir = os.path.join(EJ1_DIR, "results", "part2", "selected_model")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "selected_model_raw.csv")
    summary_path = os.path.join(out_dir, "selected_model_summary.csv")
    for path in (raw_path, summary_path):
        if os.path.exists(path):
            os.remove(path)

    selected = {
        "candidate_name": selected_candidate_name(),
        "strategy": SELECTED_STRATEGY,
        "optimizer": SELECTED_OPTIMIZER,
        "learning_rate": SELECTED_LEARNING_RATE,
        "batch_size": SELECTED_BATCH_SIZE,
        "momentum": float(cfg["training"].get("momentum", 0.9)),
    }

    run_cfg = copy_cfg(
        cfg,
        scaler_name=cfg.get("generalization_search", {}).get("selected_scaler", cfg["data"]["preprocess"]["feature_scaler"]),
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
            append_standard_rows_csv(
                raw_path,
                raw_rows_for_split(run_cfg, selected["candidate_name"], seed, len(test_idx), run),
                RAW_FIELDNAMES,
            )
        seed_summary = summarize_seed(run_results)
        seed_summary["seed"] = seed
        seed_summaries.append(seed_summary)
        print(
            f"  seed={seed}: val_f2={seed_summary['mean_val_f2']:.4f} ± {seed_summary['std_val_f2']:.4f} "
            f"val_cost={seed_summary['mean_val_cost']:.4f}"
        )

    seed_summaries_df = pd.DataFrame(seed_summaries)
    selected_seed = selected_seed_row(seed_summaries_df)
    final_epochs = max(1, int(round(selected_seed["mean_stopped_at"])))

    final_result = retrain_final_model(
        X, t, y, train_val_idx, test_idx, run_cfg, int(selected_seed["seed"]), float(selected_seed["selected_threshold"]), final_epochs
    )
    append_standard_rows_csv(
        raw_path,
        final_raw_rows(run_cfg, selected["candidate_name"], int(selected_seed["seed"]), len(train_val_idx), len(test_idx), final_result),
        RAW_FIELDNAMES,
    )

    summary = pd.DataFrame([{
        "candidate_name": selected["candidate_name"],
        "strategy": SELECTED_STRATEGY,
        "scaler": run_cfg["data"]["preprocess"]["feature_scaler"],
        "optimizer": run_cfg["training"]["optimizer"],
        "learning_rate": run_cfg["training"]["learning_rate"],
        "batch_size": run_cfg["training"]["batch_size"],
        "batch_label": batch_label(run_cfg["training"]["batch_size"]),
        "momentum": run_cfg["training"].get("momentum", 0.9),
        "seed": int(selected_seed["seed"]),
        "threshold": float(selected_seed["selected_threshold"]),
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "epochs": final_epochs,
        "mean_val_f2": float(selected_seed["mean_val_f2"]),
        "std_val_f2": float(selected_seed["std_val_f2"]),
        "mean_val_cost": float(selected_seed["mean_val_cost"]),
        "std_val_cost": float(selected_seed["std_val_cost"]),
        "train_auc_pr": final_result["train_auc_pr"],
        "test_auc_pr": final_result["test_auc_pr"],
        "test_precision": final_result["test_precision"],
        "test_recall": final_result["test_recall"],
        "test_f1": final_result["test_f1"],
        "test_f2": final_result["test_f2"],
        "train_mse": final_result["train_mse"],
        "test_mse": final_result["test_mse"],
        "train_cost": final_result["train_cost"],
        "test_cost": final_result["test_cost"],
        "train_cost_mean": final_result["train_cost_mean"],
        "test_cost_mean": final_result["test_cost_mean"],
    }], columns=SUMMARY_FIELDNAMES)
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("PART 2 — SELECTED MODEL")
    print("=" * 60)
    print(f"  Candidate: {selected['candidate_name']}")
    print(f"  Strategy:  {SELECTED_STRATEGY}")
    print(f"  Optimizer: {run_cfg['training']['optimizer']}")
    print(f"  Batch:     {batch_label(run_cfg['training']['batch_size'])}")
    print(f"  LR:        {run_cfg['training']['learning_rate']}")
    print(f"  Seed:      {int(selected_seed['seed'])}")
    print(f"  Threshold: {float(selected_seed['selected_threshold']):.4f}")
    print(f"  Rule:      {THRESHOLD_SELECTION_RULE}")
    print(f"  Val F2:    {float(selected_seed['mean_val_f2']):.4f}")
    print(f"  AUC-PR:    {final_result['test_auc_pr']:.4f}")
    print(f"  F1:        {final_result['test_f1']:.4f}")
    print(f"  F2:        {final_result['test_f2']:.4f}")
    print(f"  Cost:      {int(final_result['test_cost'])} ({final_result['test_cost_mean']:.4f} por muestra)")


if __name__ == "__main__":
    main()
