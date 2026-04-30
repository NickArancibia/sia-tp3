import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from main_part2 import _make_perceptron, load_data
from plots import (plot_confusion_matrix, plot_internal_function, plot_learning_curves,
                   plot_metric_bars, plot_pr, plot_strategy_overfitting_curves,
                   plot_threshold_sweep)
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


def _metric_summary(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def _copy_cfg(cfg, scaler_name=None, training_overrides=None):
    new_cfg = copy.deepcopy(cfg)
    if scaler_name is not None:
        new_cfg["data"]["preprocess"]["feature_scaler"] = scaler_name
    if training_overrides:
        new_cfg["training"].update(training_overrides)
    return new_cfg


def _search_space(cfg):
    search_cfg = cfg.get("generalization_search", {})
    batch_base = int(search_cfg.get("joint_batch_base", 32))
    lr_base = float(search_cfg.get("joint_lr_base", 1e-4))
    joint_batch_sizes = [int(batch) for batch in search_cfg.get("joint_batch_sizes", [8, 16, 32, 64, 128])]
    return {
        "optimizers": search_cfg.get("optimizers", ["gd", "adam"]),
        "optimizer_screening_learning_rate": float(search_cfg.get("optimizer_screening_learning_rate", lr_base)),
        "optimizer_screening_batch_size": int(search_cfg.get("optimizer_screening_batch_size", batch_base)),
        "joint_batch_base": batch_base,
        "joint_lr_base": lr_base,
        "joint_batch_sizes": joint_batch_sizes,
    }


def _joint_batch_lr_configs(space):
    configs = []
    for idx, batch_size in enumerate(space["joint_batch_sizes"], start=1):
        learning_rate = space["joint_lr_base"] * (batch_size / space["joint_batch_base"])
        configs.append({
            "name": f"E{idx}",
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "label": f"E{idx}\nb={batch_size}, lr={_format_lr(learning_rate)}",
        })
    return configs


def _build_protocol_splits(strategy, train_val_idx, y, split_cfg):
    base_seed = split_cfg.get("seed", 42)
    val_frac = split_cfg.get("inner_val_frac", split_cfg.get("val_frac", 0.15))

    if strategy == "S1":
        train_rel, val_rel, _ = stratified_split(
            y[train_val_idx], val_frac, test_frac=0.0, seed=base_seed
        )
        return [{
            "split_kind": "holdout",
            "split_id": 0,
            "train_idx": train_val_idx[train_rel],
            "val_idx": train_val_idx[val_rel],
        }]

    if strategy == "S2":
        repeats = split_cfg.get("repeated_holdout_repeats", 5)
        split_specs = []
        for repeat in range(repeats):
            train_rel, val_rel, _ = stratified_split(
                y[train_val_idx],
                val_frac,
                test_frac=0.0,
                seed=base_seed + repeat,
            )
            split_specs.append({
                "split_kind": "repeated_holdout",
                "split_id": repeat,
                "train_idx": train_val_idx[train_rel],
                "val_idx": train_val_idx[val_rel],
            })
        return split_specs

    if strategy == "S3":
        cv_folds = split_cfg.get("cv_folds", 5)
        folds = stratified_kfold_indices(y[train_val_idx], cv_folds, base_seed)
        split_specs = []
        for fold_id, val_rel in enumerate(folds):
            train_rel = np.concatenate([
                fold for idx, fold in enumerate(folds) if idx != fold_id
            ])
            split_specs.append({
                "split_kind": "cv_fold",
                "split_id": fold_id,
                "train_idx": train_val_idx[train_rel],
                "val_idx": train_val_idx[val_rel],
            })
        return split_specs

    raise ValueError(f"Unknown strategy: {strategy}")


def _fit_split_model(cfg, X_train, t_train, X_val, t_val, seed):
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"].get("batch_size", 32)
    shuffle = cfg["training"].get("shuffle", True)
    es_cfg = cfg["training"].get("early_stopping", {})

    model = _make_perceptron(cfg, X_train.shape[1], seed)
    opt = build_optimizer(cfg["training"])
    rng = np.random.default_rng(seed)

    early_stopping = None
    if es_cfg.get("enabled", False):
        early_stopping = EarlyStopping(patience=es_cfg.get("patience", 30))

    train_losses = []
    val_losses = []
    stopped_at = epochs
    for epoch in range(1, epochs + 1):
        train_loss, _ = model.train_epoch(
            X_train,
            t_train,
            opt,
            batch_size=batch_size,
            shuffle=shuffle,
            rng=rng,
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


def _evaluate_split(cfg, X, t, y, split_spec, seed):
    train_idx = split_spec["train_idx"]
    val_idx = split_spec["val_idx"]

    X_train = X[train_idx]
    X_val = X[val_idx]
    t_train = t[train_idx]
    t_val = t[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    scaler_name = cfg["data"]["preprocess"].get("feature_scaler", "z-score")
    scaler = build_scaler(scaler_name)
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    model, train_losses, val_losses, stopped_at = _fit_split_model(
        cfg, X_train, t_train, X_val, t_val, seed
    )

    train_scores = model.predict(X_train)
    val_scores = model.predict(X_val)

    thresholds, sweep_precs, sweep_recs, sweep_f1s, best_t = threshold_sweep(y_val, val_scores)
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
        "thresholds": thresholds,
        "threshold_precisions": sweep_precs,
        "threshold_recalls": sweep_recs,
        "threshold_f1s": sweep_f1s,
        "train_mse": float(mse(t_train, train_scores)),
        "val_mse": float(mse(t_val, val_scores)),
        "stopped_at": stopped_at,
    }


def _aggregate_threshold_curves(run_results, n_points=300):
    lo = min(float(run["thresholds"][0]) for run in run_results)
    hi = max(float(run["thresholds"][-1]) for run in run_results)
    grid = np.array([lo]) if np.isclose(lo, hi) else np.linspace(lo, hi, n_points)

    prec_curves = []
    rec_curves = []
    f1_curves = []
    for run in run_results:
        thresholds = run["thresholds"]
        prec_curves.append(np.interp(grid, thresholds, run["threshold_precisions"]))
        rec_curves.append(np.interp(grid, thresholds, run["threshold_recalls"]))
        f1_curves.append(np.interp(grid, thresholds, run["threshold_f1s"]))

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


def _summarize_protocol(cfg, strategy, seed, run_results):
    mean_val_auc_pr, std_val_auc_pr = _metric_summary([run["val_auc_pr"] for run in run_results])
    mean_val_precision, std_val_precision = _metric_summary([run["val_precision"] for run in run_results])
    mean_val_recall, std_val_recall = _metric_summary([run["val_recall"] for run in run_results])
    mean_val_f1, std_val_f1 = _metric_summary([run["val_f1"] for run in run_results])
    mean_train_auc_pr, std_train_auc_pr = _metric_summary([run["train_auc_pr"] for run in run_results])
    mean_train_f1, std_train_f1 = _metric_summary([run["train_f1"] for run in run_results])
    mean_train_mse, std_train_mse = _metric_summary([run["train_mse"] for run in run_results])
    mean_val_mse, std_val_mse = _metric_summary([run["val_mse"] for run in run_results])
    mean_threshold, std_threshold = _metric_summary([run["threshold"] for run in run_results])
    mean_stopped_at, _ = _metric_summary([run["stopped_at"] for run in run_results])

    return {
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "scaler": cfg["data"]["preprocess"].get("feature_scaler", "z-score"),
        "optimizer": cfg["training"]["optimizer"],
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "batch_size": int(cfg["training"].get("batch_size", 32)),
        "weight_decay": float(cfg["training"].get("weight_decay", 0.0)),
        "epochs": int(cfg["training"]["epochs"]),
        "seed": seed,
        "mean_val_auc_pr": mean_val_auc_pr,
        "std_val_auc_pr": std_val_auc_pr,
        "mean_val_precision": mean_val_precision,
        "std_val_precision": std_val_precision,
        "mean_val_recall": mean_val_recall,
        "std_val_recall": std_val_recall,
        "mean_val_f1": mean_val_f1,
        "std_val_f1": std_val_f1,
        "mean_train_auc_pr": mean_train_auc_pr,
        "std_train_auc_pr": std_train_auc_pr,
        "mean_train_f1": mean_train_f1,
        "std_train_f1": std_train_f1,
        "mean_train_mse": mean_train_mse,
        "std_train_mse": std_train_mse,
        "mean_val_mse": mean_val_mse,
        "std_val_mse": std_val_mse,
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "mean_stopped_at": mean_stopped_at,
        "gap_auc_pr": mean_train_auc_pr - mean_val_auc_pr,
        "gap_f1": mean_train_f1 - mean_val_f1,
        "gap_mse": mean_val_mse - mean_train_mse,
        "train_curves": [run["train_losses"] for run in run_results],
        "val_curves": [run["val_losses"] for run in run_results],
        "threshold_curves": _aggregate_threshold_curves(run_results),
        "run_results": run_results,
    }


def _protocol_to_rows(phase, cfg, strategy, seed, run_results):
    rows = []
    for run in run_results:
        rows.append({
            "phase": phase,
            "strategy": strategy,
            "joint_name": "",
            "split_kind": run["split_kind"],
            "split_id": run["split_id"],
            "seed": seed,
            "scaler": cfg["data"]["preprocess"].get("feature_scaler", "z-score"),
            "optimizer": cfg["training"]["optimizer"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"].get("batch_size", 32),
            "weight_decay": cfg["training"].get("weight_decay", 0.0),
            "epochs": cfg["training"]["epochs"],
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
        })
    return rows


def _evaluate_protocol(X, t, y, train_val_idx, cfg, strategy, seed, phase):
    split_specs = _build_protocol_splits(strategy, train_val_idx, y, cfg["data"]["split"])
    run_results = [
        _evaluate_split(cfg, X, t, y, split_spec, seed)
        for split_spec in split_specs
    ]
    summary = _summarize_protocol(cfg, strategy, seed, run_results)
    rows = _protocol_to_rows(phase, cfg, strategy, seed, run_results)
    return summary, rows


def _select_by_auc_pr(summaries):
    return max(
        summaries,
        key=lambda summary: (
            summary["mean_val_auc_pr"],
            -summary["std_val_auc_pr"],
            summary["mean_val_recall"],
        ),
    )


def _format_lr(lr):
    if lr < 1e-3:
        mantissa, exp = f"{lr:.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp = exp.replace("+0", "+").replace("-0", "-")
        return f"{mantissa}e{exp}"
    return f"{lr:.4f}".rstrip("0").rstrip(".")


def _retrain_final_model(X, t, y, train_val_idx, test_idx, cfg, seed, threshold, epochs):
    X_train = X[train_val_idx]
    X_test = X[test_idx]
    t_train = t[train_val_idx]
    t_test = t[test_idx]
    y_train = y[train_val_idx]
    y_test = y[test_idx]

    scaler = build_scaler(cfg["data"]["preprocess"].get("feature_scaler", "z-score"))
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = _make_perceptron(cfg, X_train.shape[1], seed)
    opt = build_optimizer(cfg["training"])
    batch_size = cfg["training"].get("batch_size", 32)
    shuffle = cfg["training"].get("shuffle", True)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        model.train_epoch(X_train, t_train, opt, batch_size=batch_size, shuffle=shuffle, rng=rng)

    train_scores = model.predict(X_train)
    test_scores = model.predict(X_test)
    test_preacts = model.pre_activation(X_test)
    test_pred = (test_scores >= threshold).astype(int)

    train_precs, train_recs = pr_curve(y_train, train_scores)
    test_precs, test_recs = pr_curve(y_test, test_scores)
    precision, recall, f1 = precision_recall_f1(y_test, test_pred)

    return {
        "threshold": float(threshold),
        "epochs": int(epochs),
        "train_mse": float(mse(t_train, train_scores)),
        "test_mse": float(mse(t_test, test_scores)),
        "train_auc_pr": auc(train_recs, train_precs),
        "test_auc_pr": auc(test_recs, test_precs),
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "pr_precisions": test_precs,
        "pr_recalls": test_recs,
        "test_confusion_matrix": np.array([
            [int(np.sum((test_pred == 0) & (y_test == 0))), int(np.sum((test_pred == 1) & (y_test == 0)))],
            [int(np.sum((test_pred == 0) & (y_test == 1))), int(np.sum((test_pred == 1) & (y_test == 1)))],
        ]),
        "test_preacts": test_preacts,
        "t_test": t_test,
        "y_test": y_test,
    }


def _print_hyperparam_summary(prefix, summaries, value_key):
    for summary in summaries:
        print(
            f"  {prefix}={summary[value_key]}: "
            f"AUC-PR val={summary['mean_val_auc_pr']:.4f} ± {summary['std_val_auc_pr']:.4f}"
        )


def run_generalization(X, t, y, cfg, results_dir):
    print("\n" + "=" * 60)
    print("SIMULACIONES — GENERALIZACIÓN")
    print("=" * 60)

    split_cfg = cfg["data"]["split"]
    test_frac = split_cfg.get("test_frac", 0.15)
    split_seed = split_cfg.get("seed", 42)
    baseline_seed = cfg["experiment"].get("seed", 42)
    training_seeds = cfg["experiment"].get("seeds", [baseline_seed])
    baseline_scaler = cfg["data"]["preprocess"].get("feature_scaler", "z-score")
    space = _search_space(cfg)

    train_val_idx, _, test_idx = stratified_split(y, val_frac=0.0, test_frac=test_frac, seed=split_seed)
    print(f"TEST fijo reservado: {len(test_idx)} muestras ({test_frac*100:.1f}%)")

    csv_rows = []

    baseline_cfg = _copy_cfg(cfg, scaler_name=baseline_scaler)
    strategy_summaries = []
    for strategy in ("S1", "S2", "S3"):
        summary, rows = _evaluate_protocol(
            X,
            t,
            y,
            train_val_idx,
            baseline_cfg,
            strategy,
            baseline_seed,
            phase="strategy",
        )
        strategy_summaries.append(summary)
        csv_rows.extend(rows)
        print(
            f"  {summary['strategy_label']}: "
            f"AUC-PR val={summary['mean_val_auc_pr']:.4f} ± {summary['std_val_auc_pr']:.4f} | "
            f"MSE val={summary['mean_val_mse']:.6f} | gap MSE={summary['gap_mse']:.6f}"
        )

    selected_strategy = next(
        summary for summary in strategy_summaries if summary["strategy"] == "S3"
    )

    plot_metric_bars(
        [summary["strategy_label"] for summary in strategy_summaries],
        [summary["mean_val_auc_pr"] for summary in strategy_summaries],
        [summary["std_val_auc_pr"] for summary in strategy_summaries],
        ylabel="AUC-PR en validación",
        title="Comparación de estrategias de datos",
        path=os.path.join(results_dir, "data_strategy_aucpr.png"),
    )
    plot_strategy_overfitting_curves(
        {
            summary["strategy_label"]: {
                "train": summary["train_curves"],
                "val": summary["val_curves"],
            }
            for summary in strategy_summaries
        },
        path=os.path.join(results_dir, "data_strategy_overfitting_curves.png"),
    )

    scaler_summaries = []
    for scaler_name in ("none", "z-score", "min-max"):
        scaler_cfg = _copy_cfg(cfg, scaler_name=scaler_name)
        summary, rows = _evaluate_protocol(
            X,
            t,
            y,
            train_val_idx,
            scaler_cfg,
            selected_strategy["strategy"],
            baseline_seed,
            phase="scaler",
        )
        scaler_summaries.append(summary)
        csv_rows.extend(rows)
        print(
            f"  scaler={scaler_name}: "
            f"AUC-PR val={summary['mean_val_auc_pr']:.4f} ± {summary['std_val_auc_pr']:.4f}"
        )

    selected_scaler = _select_by_auc_pr(scaler_summaries)
    plot_metric_bars(
        [summary["scaler"] for summary in scaler_summaries],
        [summary["mean_val_auc_pr"] for summary in scaler_summaries],
        [summary["std_val_auc_pr"] for summary in scaler_summaries],
        ylabel="AUC-PR en validación",
        title=f"Escalado sobre {selected_strategy['strategy_label']}",
        path=os.path.join(results_dir, "scaling_comparison.png"),
    )

    optimizer_summaries = []
    for optimizer_name in space["optimizers"]:
        opt_cfg = _copy_cfg(
            cfg,
            scaler_name=selected_scaler["scaler"],
            training_overrides={
                "learning_rate": space["optimizer_screening_learning_rate"],
                "optimizer": optimizer_name,
                "batch_size": space["optimizer_screening_batch_size"],
            },
        )
        summary, rows = _evaluate_protocol(
            X,
            t,
            y,
            train_val_idx,
            opt_cfg,
            selected_strategy["strategy"],
            baseline_seed,
            phase="optimizer_screening",
        )
        optimizer_summaries.append(summary)
        csv_rows.extend(rows)

    _print_hyperparam_summary("optimizer", optimizer_summaries, "optimizer")
    selected_optimizer = _select_by_auc_pr(optimizer_summaries)
    plot_metric_bars(
        [summary["optimizer"] for summary in optimizer_summaries],
        [summary["mean_val_auc_pr"] for summary in optimizer_summaries],
        [summary["std_val_auc_pr"] for summary in optimizer_summaries],
        ylabel="AUC-PR en validación",
        title=(
            "Screening de optimizador "
            f"(batch={space['optimizer_screening_batch_size']}, "
            f"lr={_format_lr(space['optimizer_screening_learning_rate'])})"
        ),
        path=os.path.join(results_dir, "optimizer_screening_aucpr.png"),
    )

    joint_summaries = []
    joint_configs = _joint_batch_lr_configs(space)
    for joint_cfg in joint_configs:
        batch_cfg = _copy_cfg(
            cfg,
            scaler_name=selected_scaler["scaler"],
            training_overrides={
                "optimizer": selected_optimizer["optimizer"],
                "learning_rate": joint_cfg["learning_rate"],
                "batch_size": joint_cfg["batch_size"],
            },
        )
        summary, rows = _evaluate_protocol(
            X,
            t,
            y,
            train_val_idx,
            batch_cfg,
            selected_strategy["strategy"],
            baseline_seed,
            phase="batch_lr_joint",
        )
        summary["joint_name"] = joint_cfg["name"]
        summary["joint_label"] = joint_cfg["label"]
        for row in rows:
            row["joint_name"] = joint_cfg["name"]
        joint_summaries.append(summary)
        csv_rows.extend(rows)

    for summary in joint_summaries:
        print(
            f"  {summary['joint_name']} (batch={summary['batch_size']}, "
            f"lr={_format_lr(summary['learning_rate'])}): "
            f"AUC-PR val={summary['mean_val_auc_pr']:.4f} ± {summary['std_val_auc_pr']:.4f}"
        )

    selected_joint = _select_by_auc_pr(joint_summaries)
    plot_metric_bars(
        [summary["joint_label"] for summary in joint_summaries],
        [summary["mean_val_auc_pr"] for summary in joint_summaries],
        [summary["std_val_auc_pr"] for summary in joint_summaries],
        ylabel="AUC-PR en validación",
        title="Búsqueda conjunta de batch size y learning rate",
        path=os.path.join(results_dir, "batch_lr_joint_comparison_aucpr.png"),
    )

    selected_cfg = _copy_cfg(
        cfg,
        scaler_name=selected_scaler["scaler"],
        training_overrides={
            "optimizer": selected_optimizer["optimizer"],
            "learning_rate": selected_joint["learning_rate"],
            "batch_size": selected_joint["batch_size"],
        },
    )

    seed_summaries = []
    for training_seed in training_seeds:
        summary, rows = _evaluate_protocol(
            X,
            t,
            y,
            train_val_idx,
            selected_cfg,
            selected_strategy["strategy"],
            training_seed,
            phase="seed_selection",
        )
        seed_summaries.append(summary)
        csv_rows.extend(rows)

    selected_seed = _select_by_auc_pr(seed_summaries)

    plot_learning_curves(
        selected_seed["train_curves"],
        selected_seed["val_curves"],
        title=(
            f"Curvas del modelo seleccionado ({selected_strategy['strategy_label']}, "
            f"scaler={selected_scaler['scaler']}, lr={_format_lr(selected_joint['learning_rate'])}, "
            f"opt={selected_optimizer['optimizer']}, batch={selected_joint['batch_size']}, "
            f"seed={selected_seed['seed']})"
        ),
        path=os.path.join(results_dir, "selected_model_learning_curve.png"),
    )

    threshold_curves = selected_seed["threshold_curves"]
    plot_threshold_sweep(
        threshold_curves["thresholds"],
        threshold_curves["precisions_mean"],
        threshold_curves["recalls_mean"],
        threshold_curves["f1s_mean"],
        selected_seed["mean_threshold"],
        path=os.path.join(results_dir, "selected_model_threshold_sweep.png"),
        precisions_std=threshold_curves["precisions_std"],
        recalls_std=threshold_curves["recalls_std"],
        f1s_std=threshold_curves["f1s_std"],
    )

    final_epochs = max(1, int(round(selected_seed["mean_stopped_at"])))
    final_result = _retrain_final_model(
        X,
        t,
        y,
        train_val_idx,
        test_idx,
        selected_cfg,
        selected_seed["seed"],
        selected_seed["mean_threshold"],
        final_epochs,
    )

    plot_pr(
        final_result["pr_precisions"],
        final_result["pr_recalls"],
        final_result["test_auc_pr"],
        path=os.path.join(results_dir, "selected_model_pr_curve.png"),
    )
    plot_confusion_matrix(
        final_result["test_confusion_matrix"],
        path=os.path.join(results_dir, "selected_model_confusion_matrix.png"),
    )
    plot_internal_function(
        final_result["test_preacts"],
        final_result["t_test"],
        cfg["model"]["activation"],
        beta=cfg["model"].get("beta", 1.0),
        path=os.path.join(results_dir, "selected_model_internal_function.png"),
    )

    csv_rows.append({
        "phase": "final_test",
        "strategy": selected_strategy["strategy"],
        "joint_name": selected_joint["joint_name"],
        "split_kind": "fixed_test",
        "split_id": 0,
        "seed": selected_seed["seed"],
        "scaler": selected_scaler["scaler"],
        "optimizer": selected_optimizer["optimizer"],
        "learning_rate": selected_joint["learning_rate"],
        "batch_size": selected_joint["batch_size"],
        "weight_decay": selected_cfg["training"].get("weight_decay", 0.0),
        "epochs": final_result["epochs"],
        "train_size": len(train_val_idx),
        "val_size": len(test_idx),
        "threshold_selected": final_result["threshold"],
        "train_mse": final_result["train_mse"],
        "val_mse": final_result["test_mse"],
        "train_auc_pr": final_result["train_auc_pr"],
        "val_auc_pr": final_result["test_auc_pr"],
        "train_precision": np.nan,
        "train_recall": np.nan,
        "train_f1": np.nan,
        "val_precision": final_result["test_precision"],
        "val_recall": final_result["test_recall"],
        "val_f1": final_result["test_f1"],
        "generalization_gap_auc_pr": final_result["train_auc_pr"] - final_result["test_auc_pr"],
        "generalization_gap_f1": np.nan,
        "generalization_gap_mse": final_result["test_mse"] - final_result["train_mse"],
        "stopped_at": final_result["epochs"],
    })

    csv_path = os.path.join(results_dir, "generalization_experiments.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("SELECCIÓN FINAL")
    print("=" * 60)
    print(f"  Estrategia seleccionada: {selected_strategy['strategy_label']} (forzada por criterio metodológico)")
    print(f"  Scaler seleccionado:    {selected_scaler['scaler']}")
    print(
        f"  Screening optimizador:  batch={space['optimizer_screening_batch_size']}, "
        f"lr={_format_lr(space['optimizer_screening_learning_rate'])}"
    )
    print(f"  Optimizer seleccionado: {selected_optimizer['optimizer']}")
    print(f"  Config conjunta:        {selected_joint['joint_name']}")
    print(f"  Learning rate:          {selected_joint['learning_rate']}")
    print(f"  Batch size:             {selected_joint['batch_size']}")
    print(f"  Seed seleccionada:      {selected_seed['seed']}")
    print(f"  Threshold final:        {selected_seed['mean_threshold']:.4f} ± {selected_seed['std_threshold']:.4f}")
    print(f"  AUC-PR test:            {final_result['test_auc_pr']:.4f}")
    print(f"  Precision test:         {final_result['test_precision']:.4f}")
    print(f"  Recall test:            {final_result['test_recall']:.4f}")
    print(f"  F1 test:                {final_result['test_f1']:.4f}")
    print("  CSV:                    results/part2/generalization_experiments.csv")
    print("  Gráficos:               results/part2/data_strategy_*.png, scaling_comparison.png,")
    print("                           optimizer_screening_aucpr.png, batch_lr_joint_comparison_aucpr.png,")
    print("                           selected_model_*.png")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base, "config.yaml"))

    results_dir = os.path.join(base, cfg["experiment"]["results_dir"], "part2")
    os.makedirs(results_dir, exist_ok=True)

    X, t, y, feature_cols = load_data(cfg)
    run_generalization(X, t, y, cfg, results_dir)
