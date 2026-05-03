"""EJ3 part2 — Config N: mejor de I con menor wd para más capacidad."""
import os
import pickle
import sys

EJ3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ3_DIR)
sys.path.insert(0, EJ3_DIR)
sys.path.insert(0, REPO_DIR)

from common import (RESULTS_PART2, TEST_CSV, TRAIN_CSV, build_mlp,
                    evaluate_on_test, prepare_data, train_model)
from shared.optimizers import Adam

CONFIG = {
    "arch": [784, 384, 192, 10],
    "lr": 5e-4,
    "batch_size": 64,
    "init_scale": 0.03,
    "weight_decay": 5e-6,
    "max_epochs": 220,
    "patience": 30,
}
SEEDS = [0, 1, 2]
OUT_DIR = os.path.join(RESULTS_PART2, "config_n_i_longer_wd5e6")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_data(TRAIN_CSV, TEST_CSV, val_frac=0.2, scaler="z-score", seed=42)
    print(f"Config N: {CONFIG}")
    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        model = build_mlp(CONFIG["arch"], seed=seed, init_scale=CONFIG["init_scale"], weight_decay=CONFIG["weight_decay"])
        opt = Adam(lr=CONFIG["lr"])
        hist = train_model(model, data, opt, max_epochs=CONFIG["max_epochs"], batch_size=CONFIG["batch_size"], early_stopping_patience=CONFIG["patience"], verbose=False, seed=seed)
        ev = evaluate_on_test(model, data)
        n_params = sum(W.size + b.size for W, b in zip(model.weights, model.biases))
        row = {"seed": seed, "n_params": n_params, "config": CONFIG, **hist, **ev}
        results.append(row)
        print(f"  test_acc={ev['test_acc']:.4f}")
    with open(os.path.join(OUT_DIR, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"\nConfig N avg test_acc: {sum(r['test_acc'] for r in results) / len(results):.4f}")

if __name__ == "__main__":
    main()
