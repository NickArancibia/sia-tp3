"""Data augmentation for underrepresented digit classes.

Generates augmented samples to balance the dataset, targeting classes
that fall below a minimum count threshold.

Class integrity is preserved by:
  - Conservative parameter bounds per transform type
  - Excluding flips (a mirrored 5 is not a 5)
  - Saving a visual inspection grid before writing the CSV

Usage:
  python3 augment.py                  # generates merged_augmented.csv
  python3 augment.py --inspect-only   # only saves the inspection grid, no CSV
"""

import argparse
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate, shift, zoom

# ---------------------------------------------------------------------------
# Augmentation parameters — adjust these to control transform intensity.
#
# All bounds are chosen so the digit remains unambiguous:
#   - rotation:    ±10° keeps digits readable (beyond ~20° a 6 looks like a 9)
#   - translation: ±3px on a 28×28 grid (~10% shift) keeps digit inside frame
#   - zoom:        0.85–1.15 — mild scale change, digit still fills the frame
#   - noise_std:   0.04 — adds texture without obscuring strokes
# ---------------------------------------------------------------------------
AUG_PARAMS = {
    "rotation_max_deg": 5.0,
    "translation_max_px": 2.0,
    "zoom_min": 0.95,
    "zoom_max": 1.05,
    "noise_std": 0.03,
}

IMG_SHAPE = (28, 28)


def _rotate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-AUG_PARAMS["rotation_max_deg"], AUG_PARAMS["rotation_max_deg"])
    return rotate(img, angle, reshape=False, mode="nearest")


def _translate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    max_px = AUG_PARAMS["translation_max_px"]
    dy = rng.uniform(-max_px, max_px)
    dx = rng.uniform(-max_px, max_px)
    return shift(img, [dy, dx], mode="nearest")


def _zoom_img(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(AUG_PARAMS["zoom_min"], AUG_PARAMS["zoom_max"])
    zoomed = zoom(img, factor, mode="nearest")
    # crop or pad back to IMG_SHAPE
    out = np.zeros(IMG_SHAPE, dtype=np.float32)
    h, w = zoomed.shape
    # center the zoomed image
    r0 = max(0, (h - IMG_SHAPE[0]) // 2)
    c0 = max(0, (w - IMG_SHAPE[1]) // 2)
    r1 = min(h, r0 + IMG_SHAPE[0])
    c1 = min(w, c0 + IMG_SHAPE[1])
    dr0 = max(0, (IMG_SHAPE[0] - h) // 2)
    dc0 = max(0, (IMG_SHAPE[1] - w) // 2)
    out[dr0:dr0 + (r1 - r0), dc0:dc0 + (c1 - c0)] = zoomed[r0:r1, c0:c1]
    return out


def _add_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, AUG_PARAMS["noise_std"], img.shape).astype(np.float32)
    return img + noise


TRANSFORMS = [_rotate, _translate, _zoom_img, _add_noise]


def augment_one(flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply rotation, translation, and gaussian noise, then clip to [0, 1]."""
    img = flat.reshape(IMG_SHAPE).astype(np.float32)
    img = _rotate(img, rng)
    img = _translate(img, rng)
    img = _add_noise(img, rng)
    return np.clip(img, 0.0, 1.0).flatten()


def save_inspection_grid(originals: np.ndarray, augmented: np.ndarray,
                         label: int, n: int, path: str) -> None:
    """Save a side-by-side grid: left=originals, right=augmented samples."""
    fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3))
    fig.suptitle(f"Digit {label} — top: originals, bottom: augmented", fontsize=10)
    for i in range(n):
        axes[0, i].imshow(originals[i % len(originals)].reshape(IMG_SHAPE),
                          cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(augmented[i].reshape(IMG_SHAPE), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Inspection grid saved → {path}")


def main(inspect_only: bool = False, seed: int = 42) -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    src_path = os.path.join(data_dir, "merged_digits.csv")
    dst_path = os.path.join(data_dir, "merged_augmented.csv")
    inspect_dir = os.path.join(os.path.dirname(__file__), "results", "augmentation_check")
    os.makedirs(inspect_dir, exist_ok=True)

    print(f"Loading {src_path} ...")
    df = pd.read_csv(src_path)
    df["image"] = df["image"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32))

    counts = df["label"].value_counts().sort_index()
    print("\nClass distribution before augmentation:")
    for lbl, cnt in counts.items():
        print(f"  Digit {lbl}: {cnt}")

    target = int(counts.quantile(0.75))  # target = 75th percentile of current counts
    print(f"\nTarget count per class: {target}")

    rng = np.random.default_rng(seed)
    new_rows = []

    for label in sorted(df["label"].unique()):
        current = counts[label]
        needed = target - current
        if needed <= 0:
            print(f"  Digit {label}: {current} samples — OK, skipping")
            continue

        print(f"  Digit {label}: {current} → generating {needed} augmented samples ...")
        class_images = np.stack(df[df["label"] == label]["image"].values)

        augmented = []
        for _ in range(needed):
            src = class_images[rng.integers(len(class_images))]
            augmented.append(augment_one(src, rng))
        augmented = np.array(augmented)

        # Save inspection grid (10 samples)
        n_show = min(10, needed)
        grid_path = os.path.join(inspect_dir, f"digit_{label}_augmented.png")
        save_inspection_grid(class_images, augmented, label, n_show, grid_path)

        for flat in augmented:
            new_rows.append({"label": label, "image": flat.tolist()})

    if not new_rows:
        print("\nNo augmentation needed — all classes at or above target.")
        return

    aug_df = pd.DataFrame(new_rows)
    aug_df["image"] = aug_df["image"].apply(str)
    orig_df = df.copy()
    orig_df["image"] = orig_df["image"].apply(lambda x: x.tolist().__str__())

    combined = pd.concat([orig_df, aug_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    new_counts = combined["label"].value_counts().sort_index()
    print("\nClass distribution after augmentation:")
    for lbl, cnt in new_counts.items():
        print(f"  Digit {lbl}: {cnt}")

    if not inspect_only:
        combined.to_csv(dst_path, index=False)
        print(f"\nAugmented dataset saved → {dst_path}")
        print(f"Total samples: {len(combined)}")
    else:
        print("\n--inspect-only: CSV not written.")

    print(f"\nInspection grids saved in {inspect_dir}/")
    print("Review them before retraining to verify class integrity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true",
                        help="Generate inspection grids but do not write the CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(inspect_only=args.inspect_only, seed=args.seed)
