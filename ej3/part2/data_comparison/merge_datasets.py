"""Merge digits datasets and remove exact duplicate rows.

Default behavior:
- reads ej2/data/digits.csv
- reads ej3/data/more_digits.csv
- concatenates both datasets
- removes exact duplicates using all shared columns
- writes a merged CSV under ej3/data/

This is useful if you want to train a single model with the union of both
sources while avoiding double-counting samples that appear in both files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _default_paths() -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    ej3_dir = script_path.parents[2]
    repo_dir = script_path.parents[3]
    digits_path = repo_dir / "ej2" / "data" / "digits.csv"
    more_digits_path = ej3_dir / "data" / "more_digits.csv"
    output_path = ej3_dir / "data" / "merged_digits.csv"
    return digits_path, more_digits_path, output_path


def _row_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].astype(str).agg("|".join, axis=1)


def merge_datasets(digits_path: Path, more_digits_path: Path, output_path: Path) -> dict:
    digits_df = pd.read_csv(digits_path)
    more_df = pd.read_csv(more_digits_path)

    shared_columns = [column for column in digits_df.columns if column in more_df.columns]
    if not shared_columns:
        raise ValueError("No shared columns found between the two datasets.")

    digits_keys = _row_key(digits_df, shared_columns)
    more_keys = _row_key(more_df, shared_columns)

    merged_df = pd.concat(
        [digits_df.assign(_row_key=digits_keys), more_df.assign(_row_key=more_keys)],
        ignore_index=True,
    )
    merged_df = merged_df.drop_duplicates(subset="_row_key", keep="first").drop(columns=["_row_key"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    duplicate_rows = len(digits_df) + len(more_df) - len(merged_df)
    label_counts = merged_df["label"].value_counts().sort_index().to_dict() if "label" in merged_df.columns else {}

    return {
        "digits_rows": len(digits_df),
        "more_digits_rows": len(more_df),
        "merged_rows": len(merged_df),
        "duplicate_rows_removed": duplicate_rows,
        "output_path": str(output_path),
        "label_counts": label_counts,
    }


def main() -> None:
    default_digits_path, default_more_digits_path, default_output_path = _default_paths()

    parser = argparse.ArgumentParser(
        description="Merge digits.csv and more_digits.csv, removing exact duplicate rows."
    )
    parser.add_argument("--digits", type=Path, default=default_digits_path, help="Path to digits.csv")
    parser.add_argument(
        "--more-digits",
        type=Path,
        default=default_more_digits_path,
        help="Path to more_digits.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path,
        help="Output path for the merged CSV",
    )
    args = parser.parse_args()

    stats = merge_datasets(args.digits, args.more_digits, args.output)

    print("Merged datasets successfully")
    print(f"  digits rows:          {stats['digits_rows']}")
    print(f"  more_digits rows:     {stats['more_digits_rows']}")
    print(f"  merged rows:          {stats['merged_rows']}")
    print(f"  duplicates removed:    {stats['duplicate_rows_removed']}")
    print(f"  output:                {stats['output_path']}")
    if stats["label_counts"]:
        print("  merged label counts:")
        for label, count in stats["label_counts"].items():
            print(f"    {label}: {count}")


if __name__ == "__main__":
    main()