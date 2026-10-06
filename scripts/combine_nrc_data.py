"""Combine daily NRC CSV files into one file."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=root / "data/processed/nrc/daily",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "data/processed/nrc",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*.csv"))
    df = pd.concat(
        [pd.read_csv(path) for path in files],
        ignore_index=True,
    )

    # Remove the last unnecessary column
    df = df.iloc[:, :-1].copy()
    df = df.sort_values(["unit_name", "date"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "nrc_reactor_daily.csv"
    df.to_csv(output, index=False)

    print(f"Combined {len(files)} files into {len(df):,} rows.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()