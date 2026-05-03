import os
import sys

import numpy as np
import pandas as pd

EJ1_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ1_DIR)
PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EJ1_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PART2_DIR)

from common import (
    RAW_FIELDNAMES,
    aggregate_seed_summary,
    append_rows_csv,
    base_row,
    batch_label,
    build_splits,
    build_train_val_test_indices,
    copy_cfg,
    curve_rows,
    evaluate_binary_scores,
    fit_model_fixed_epochs,
    fit_model_with_validation,
    load_raw_df,
    prepare_raw_file,
    sample_output_rows,
    select_threshold_by_f2,
    split_summary_row,
)
from main_part2 import _make_perceptron, load_data
from shared.config_loader import load_config
from shared.preprocessing import build_scaler

EXPERIMENT_TYPE = "optimizer"
OUT_DIR = os.path.join(EJ1_DIR, "results", "part2", "optimizer")
RAW_PATH = os.path.join(OUT_DIR, "optimizer_raw.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "optimizer_summary.csv")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")

SUMMARY_FIELDNAMES = [
    "candidate_name", "optimizer", "learning_rate", "batch_size", "batch_label", "momentum",
    "mean_val_auc_pr", "std_val_auc_pr", "mean_val_accuracy", "std_val_accuracy",
    "mean_val_precision", "std_val_precision", "mean_val_recall", "std_val_recall",
    "mean_val_f1", "std_val_f1", "mean_val_f2", "std_val_f2", "mean_val_cost", "std_val_cost",
    "mean_train_mse", "std_train_mse", "mean_val_mse", "std_val_mse", "gap_mse",
    "mean_threshold", "std_threshold", "mean_stopped_at",
    "mean_test_auc_pr", "std_test_auc_pr", "mean_test_accuracy", "std_test_accuracy",
    "mean_test_precision", "std_test_precision", "mean_test_recall", "std_test_recall",
    "mean_test_f1", "std_test_f1", "mean_test_f2", "std_test_f2",
    "mean_test_cost", "std_test_cost", "mean_test_mse", "std_test_mse",
    "mean_elapsed_s", "std_elapsed_s",
]


def candidate_name(optimizer_name, learning_rate):
    return f"{optimizer_name}_lr{learning_rate:.0e}"


def split_done(raw_df, run_name, seed, split_spec):
    mask = (
        (raw_df["record_type"] == "split_summary")
        & (raw_df["experiment_type"] == EXPERIMENT_TYPE)
        & (raw_df["candidate_name"] == run_name)
        & (raw_df["seed"].astype(str) == str(seed))
        & (raw_df["split_kind"] == split_spec["split_kind"])
        & (raw_df["split_id"].astype(str) == str(split_spec["split_id"]))
        & (raw_df["subset"] == "val")
    )
    return bool(mask.any())


def final_retrain_done(raw_df, run_name, seed):
    mask = (
        (raw_df["record_type"] == "split_summary")
        & (raw_df["experiment_type"] == EXPERIMENT_TYPE)
        & (raw_df["candidate_name"] == run_name)
        & (raw_df["seed"].astype(str) == str(seed))
        & (raw_df["split_kind"] == "final_retrain")
        & (raw_df["subset"] == "test")
    )
    return bool(mask.any())


def summarize_candidate(raw_df, run_name, optimizer_name, learning_rate, batch_size, momentum):
    seed_rows = raw_df[
        (raw_df["record_type"] == "split_summary")
        & (raw_df["experiment_type"] == EXPERIMENT_TYPE)
        & (raw_df["candidate_name"] == run_name)
        & (raw_df["split_kind"] == "final_retrain")
        & (raw_df["subset"] == "test")
    ]
    seed_summaries = []
    for seed in sorted(seed_rows["seed"].astype(int).unique()):
        seed_summary = aggregate_seed_summary(
            raw_df,
            {"experiment_type": EXPERIMENT_TYPE, "candidate_name": run_name, "seed": seed},
        )
        if seed_summary is not None:
            seed_summaries.append(seed_summary)

    if not seed_summaries:
        return None

    summary = {
        "candidate_name": run_name,
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "batch_label": batch_label(batch_size),
        "momentum": float(momentum),
    }
    for metric in [
        "val_auc_pr", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_f2", "val_cost",
        "train_mse", "val_mse", "test_auc_pr", "test_accuracy", "test_precision", "test_recall",
        "test_f1", "test_f2", "test_cost", "test_mse", "elapsed_s",
    ]:
        if metric == "elapsed_s":
            values = np.asarray([row["elapsed_s_total"] for row in seed_summaries], dtype=float)
            prefix = "elapsed_s"
        else:
            key = f"mean_{metric}" if metric.startswith(("val_", "train_")) else metric
            values = np.asarray([row[key] for row in seed_summaries], dtype=float)
            prefix = metric
        summary[f"mean_{prefix}"] = float(values.mean())
        summary[f"std_{prefix}"] = float(values.std(ddof=0))

    summary["gap_mse"] = float(np.mean([row["gap_mse"] for row in seed_summaries]))
    summary["mean_threshold"] = float(np.mean([row["mean_threshold"] for row in seed_summaries]))
    summary["std_threshold"] = float(np.std([row["mean_threshold"] for row in seed_summaries], ddof=0))
    summary["mean_stopped_at"] = float(np.mean([row["mean_stopped_at"] for row in seed_summaries]))
    return summary


def main():
    cfg = load_config(os.path.join(EJ1_DIR, "config.yaml"))
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    raw_df = prepare_raw_file(RAW_PATH)
    X, t, y, _ = load_data(cfg)
    split_cfg = cfg["data"]["split"]
    train_val_idx, test_idx = build_train_val_test_indices(y, split_cfg)
    split_specs = build_splits("S3", train_val_idx, y, split_cfg)
    search_cfg = cfg["generalization_search"]
    learning_rates = [float(value) for value in search_cfg["optimizer_learning_rates"]]
    seeds = [int(seed) for seed in cfg["experiment"].get("seeds", [cfg["experiment"].get("seed", 42)])]
    batch_size = int(search_cfg["optimizer_batch_size"])
    optimizer_specs = [
        {"name": "gd", "optimizer": "gd", "momentum": cfg["training"].get("momentum", 0.9)},
        {"name": "adam", "optimizer": "adam", "momentum": cfg["training"].get("momentum", 0.9)},
        {"name": "momentum", "optimizer": "momentum", "momentum": cfg["training"].get("momentum", 0.9)},
    ]

    print("\n" + "=" * 60)
    print("PART 2 — OPTIMIZER")
    print("=" * 60)

    summaries = []
    for spec in optimizer_specs:
        for learning_rate in learning_rates:
            run_name = candidate_name(spec["name"], learning_rate)
            run_cfg = copy_cfg(
                cfg,
                scaler_name=search_cfg["selected_scaler"],
                training_overrides={
                    "optimizer": spec["optimizer"],
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "momentum": spec["momentum"],
                },
            )

            for seed in seeds:
                if final_retrain_done(raw_df, run_name, seed):
                    print(f"  {run_name} seed={seed}: ya completo, salteo.")
                    continue

                for split_spec in split_specs:
                    if split_done(raw_df, run_name, seed, split_spec):
                        continue

                    train_idx = split_spec["train_idx"]
                    val_idx = split_spec["val_idx"]
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    t_train = t[train_idx]
                    t_val = t[val_idx]
                    y_train = y[train_idx]
                    y_val = y[val_idx]

                    scaler = build_scaler(run_cfg["data"]["preprocess"]["feature_scaler"])
                    if scaler is not None:
                        X_train = scaler.fit_transform(X_train)
                        X_val = scaler.transform(X_val)

                    split_key = f"{run_name}_seed{seed}_{split_spec['split_kind']}_{split_spec['split_id']}"
                    model, train_losses, val_losses, stopped_at, elapsed_s, elapsed_curve = fit_model_with_validation(
                        run_cfg, _make_perceptron, X_train, t_train, X_val, t_val, seed,
                        CHECKPOINT_DIR, split_key,
                    )
                    train_scores = model.predict(X_train)
                    val_scores = model.predict(X_val)
                    threshold = select_threshold_by_f2(y_val, val_scores, beta=2.0)
                    train_metrics = evaluate_binary_scores(y_train, train_scores, threshold, targets=t_train, beta=2.0)
                    val_metrics = evaluate_binary_scores(y_val, val_scores, threshold, targets=t_val, beta=2.0)

                    curve_base = base_row(
                        run_cfg, EXPERIMENT_TYPE, seed, len(train_idx), len(val_idx), len(test_idx),
                        config_name=run_name, candidate_name=run_name, split_kind=split_spec["split_kind"], split_id=split_spec["split_id"],
                    )
                    train_base = base_row(
                        run_cfg, EXPERIMENT_TYPE, seed, len(train_idx), len(val_idx), len(test_idx),
                        config_name=run_name, candidate_name=run_name, split_kind=split_spec["split_kind"], split_id=split_spec["split_id"], subset="train",
                    )
                    val_base = base_row(
                        run_cfg, EXPERIMENT_TYPE, seed, len(train_idx), len(val_idx), len(test_idx),
                        config_name=run_name, candidate_name=run_name, split_kind=split_spec["split_kind"], split_id=split_spec["split_id"], subset="val",
                    )

                    rows = curve_rows(curve_base, train_losses, val_losses, elapsed_curve, stopped_at=stopped_at)
                    rows.append(split_summary_row(train_base, train_metrics, threshold, "f2_on_val", stopped_at))
                    rows.append(split_summary_row(val_base, val_metrics, threshold, "f2_on_val", stopped_at, elapsed_s=elapsed_s))
                    if search_cfg["raw_outputs"].get("include_val_samples", True):
                        rows.extend(
                            sample_output_rows(
                                val_base,
                                y_val,
                                t_val,
                                val_scores,
                                model.pre_activation(X_val),
                                val_metrics["pred"],
                                threshold,
                                "f2_on_val",
                            )
                        )
                    append_rows_csv(RAW_PATH, rows, RAW_FIELDNAMES)
                    raw_df = load_raw_df(RAW_PATH)

                if final_retrain_done(raw_df, run_name, seed):
                    continue

                val_rows = raw_df[
                    (raw_df["record_type"] == "split_summary")
                    & (raw_df["experiment_type"] == EXPERIMENT_TYPE)
                    & (raw_df["candidate_name"] == run_name)
                    & (raw_df["seed"].astype(str) == str(seed))
                    & (raw_df["subset"] == "val")
                ].copy()
                if val_rows.empty:
                    continue

                threshold = float(val_rows["threshold_selected"].astype(float).mean())
                final_epochs = max(1, int(round(val_rows["stopped_at"].astype(float).mean())))

                X_train = X[train_val_idx]
                X_test = X[test_idx]
                t_train = t[train_val_idx]
                t_test = t[test_idx]
                y_train = y[train_val_idx]
                y_test = y[test_idx]
                scaler = build_scaler(run_cfg["data"]["preprocess"]["feature_scaler"])
                if scaler is not None:
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                final_key = f"{run_name}_seed{seed}_final_retrain"
                model, train_losses, elapsed_s, elapsed_curve = fit_model_fixed_epochs(
                    run_cfg, _make_perceptron, X_train, t_train, seed, final_epochs,
                    CHECKPOINT_DIR, final_key,
                )
                train_scores = model.predict(X_train)
                test_scores = model.predict(X_test)
                train_metrics = evaluate_binary_scores(y_train, train_scores, threshold, targets=t_train, beta=2.0)
                test_metrics = evaluate_binary_scores(y_test, test_scores, threshold, targets=t_test, beta=2.0)

                curve_base = base_row(
                    run_cfg, EXPERIMENT_TYPE, seed, len(train_val_idx), 0, len(test_idx),
                    config_name=run_name, candidate_name=run_name, split_kind="final_retrain", split_id=0,
                )
                train_base = base_row(
                    run_cfg, EXPERIMENT_TYPE, seed, len(train_val_idx), 0, len(test_idx),
                    config_name=run_name, candidate_name=run_name, split_kind="final_retrain", split_id=0, subset="train",
                )
                test_base = base_row(
                    run_cfg, EXPERIMENT_TYPE, seed, len(train_val_idx), 0, len(test_idx),
                    config_name=run_name, candidate_name=run_name, split_kind="final_retrain", split_id=0, subset="test",
                )

                rows = curve_rows(curve_base, train_losses, val_losses=None, elapsed_s_cumulative=elapsed_curve, stopped_at=final_epochs)
                rows.append(split_summary_row(train_base, train_metrics, threshold, "mean_val_f2_threshold", final_epochs))
                rows.append(split_summary_row(test_base, test_metrics, threshold, "mean_val_f2_threshold", final_epochs, elapsed_s=elapsed_s))
                if search_cfg["raw_outputs"].get("include_test_samples", True):
                    rows.extend(
                        sample_output_rows(
                            test_base,
                            y_test,
                            t_test,
                            test_scores,
                            model.pre_activation(X_test),
                            test_metrics["pred"],
                            threshold,
                            "mean_val_f2_threshold",
                        )
                    )
                append_rows_csv(RAW_PATH, rows, RAW_FIELDNAMES)
                raw_df = load_raw_df(RAW_PATH)

                seed_summary = aggregate_seed_summary(
                    raw_df,
                    {"experiment_type": EXPERIMENT_TYPE, "candidate_name": run_name, "seed": seed},
                )
                if seed_summary is not None:
                    print(
                        f"  {run_name} seed={seed}: val_f2={seed_summary['mean_val_f2']:.4f} "
                        f"test_f2={seed_summary['test_f2']:.4f} elapsed={seed_summary['elapsed_s_total']:.1f}s"
                    )

            raw_df = load_raw_df(RAW_PATH)
            summary = summarize_candidate(raw_df, run_name, spec["name"], learning_rate, batch_size, spec["momentum"])
            summaries = [row for row in summaries if row["candidate_name"] != run_name]
            if summary is not None:
                summaries.append(summary)
                print(
                    f"  {run_name}: val_f2={summary['mean_val_f2']:.4f} ± {summary['std_val_f2']:.4f} "
                    f"test_f2={summary['mean_test_f2']:.4f} ± {summary['std_test_f2']:.4f}"
                )
                pd.DataFrame(summaries, columns=SUMMARY_FIELDNAMES).to_csv(SUMMARY_PATH, index=False)


if __name__ == "__main__":
    main()
