"""EJ3 — entry point standalone.

Equivalente a ej2/main.py pero con more_digits.csv como train.
Usa el mismo digits_test.csv que EJ2 (mundo real, 10 clases balanceadas).

Para experimentos comparativos, usar ej3/part2/{data_comparison,selected_model}/.
"""
import argparse
import os
import sys

EJ3_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EJ3_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, EJ3_DIR)

from common import (_aug_batch, build_mlp, evaluate_on_test, prepare_data, train_model)
from shared.config_loader import load_config
from shared.optimizers import build_optimizer
from shared.utils import (plot_accuracy_curves, plot_confusion_matrix,
                          plot_learning_curves)


def run_one(cfg, verbose=True):
    train_csv = os.path.join(EJ3_DIR, cfg["data"]["train_path"]) \
        if not os.path.isabs(cfg["data"]["train_path"]) else cfg["data"]["train_path"]
    test_csv = os.path.join(EJ3_DIR, cfg["data"]["test_path"]) \
        if not os.path.isabs(cfg["data"]["test_path"]) else cfg["data"]["test_path"]

    data = prepare_data(train_csv, test_csv,
                        val_frac=cfg["data"]["split"].get("val_frac", 0.2),
                        scaler=cfg["data"]["preprocess"]["feature_scaler"],
                        stratify=cfg["data"]["split"].get("stratify", True),
                        seed=cfg["data"]["split"].get("seed", 42))
    if verbose:
        print(f"Data: train={data['X_train'].shape} val={data['X_val'].shape} "
              f"test={data['X_test'].shape} n_classes={data['n_classes']}")

    seed = cfg["experiment"].get("seed", 42)
    model = build_mlp(
        cfg["model"]["architecture"],
        hidden_act=cfg["model"].get("hidden_activation", "tanh"),
        out_act=cfg["model"].get("output_activation", "logistic"),
        beta=cfg["model"].get("beta", 1.0),
        init_scale=cfg["model"].get("init_scale", 0.1),
        seed=seed,
        weight_decay=cfg["training"].get("weight_decay", 0.0),
        loss_name=cfg["training"].get("loss", "mse"),
    )
    opt = build_optimizer(cfg["training"])
    es_cfg = cfg["training"].get("early_stopping", {})
    patience = es_cfg.get("patience", 25) if es_cfg.get("enabled", True) else 0

    if verbose:
        print(f"Architecture: {cfg['model']['architecture']}")
        print(f"Optimizer: {cfg['training']['optimizer']} lr={cfg['training']['learning_rate']}")
        print(f"Batch size: {cfg['training']['batch_size']}, Max epochs: {cfg['training']['epochs']}")
        n_params = sum(W.size + b.size for W, b in zip(model.weights, model.biases))
        print(f"Parameters: {n_params:,}\n")

    online_aug = cfg["training"].get("online_augmentation", False)
    hist = train_model(model, data, opt,
                       max_epochs=cfg["training"]["epochs"],
                       batch_size=cfg["training"]["batch_size"],
                       early_stopping_patience=patience,
                       verbose=verbose, seed=seed,
                       augment_fn=_aug_batch if online_aug else None)
    ev = evaluate_on_test(model, data)

    if verbose:
        print(f"\n=== Resultado final ===")
        print(f"Train acc: {ev['train_acc']:.4f}")
        print(f"Val acc:   {ev['val_acc']:.4f}")
        print(f"Test acc:  {ev['test_acc']:.4f}")
        per_class = ev["test_per_class"]
        for c in range(data["n_classes"]):
            print(f"  Dígito {c}: P={per_class['precision'][c]:.4f} "
                  f"R={per_class['recall'][c]:.4f} F1={per_class['f1'][c]:.4f}")
        print(f"Macro F1:    {per_class['macro_f1']:.4f}")
        print(f"Weighted F1: {per_class['weighted_f1']:.4f}")
        print(f"Stopped at epoch {hist['stopped_at']} ({hist['elapsed']:.1f}s)")

    if cfg["experiment"].get("save_plots", True):
        results_dir = os.path.join(EJ3_DIR, "results", cfg["experiment"]["name"])
        os.makedirs(results_dir, exist_ok=True)
        plot_learning_curves(hist["train_losses"], hist["val_losses"],
                             title=f"{cfg['experiment']['name']} — Loss",
                             path=os.path.join(results_dir, "loss_curve.png"))
        plot_accuracy_curves(hist["train_accs"], hist["val_accs"],
                             title=f"{cfg['experiment']['name']} — Accuracy",
                             path=os.path.join(results_dir, "accuracy_curve.png"))
        plot_confusion_matrix(ev["test_cm"],
                              labels=[str(i) for i in range(data["n_classes"])],
                              title=f"{cfg['experiment']['name']} — Confusion Matrix (Test)",
                              path=os.path.join(results_dir, "confusion_matrix.png"))
        if cfg["experiment"].get("save_model", True):
            model.save(os.path.join(results_dir, "model.npz"))
            if data["scaler"] is not None:
                data["scaler"].save(os.path.join(results_dir, "scaler.npz"))
        print(f"\nResultados guardados en {results_dir}")

    return model, hist, ev


def main():
    parser = argparse.ArgumentParser(description="Ej3 - More Digits Classification")
    parser.parse_args()
    cfg = load_config(os.path.join(EJ3_DIR, "config.yaml"))
    run_one(cfg, verbose=True)


if __name__ == "__main__":
    main()
