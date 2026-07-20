#!/usr/bin/env python3
"""Map dataset conditions to ICD-10-CM codes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.icd_mapping import ConditionICDMapper


def read_condition_counts(meta_path: Path, column: str) -> pd.Series:
    df = pd.read_csv(meta_path, usecols=[column], dtype={column: str})
    series = df[column].dropna().astype(str).str.strip()
    series = series[series != ""]
    return series.value_counts()


def format_summary(df: pd.DataFrame) -> str:
    total = int(df["count"].sum())
    mapped = df[df["icd_code"] != ""]
    mapped_total = int(mapped["count"].sum())
    return (
        f"{len(df)} unique conditions ({total:,} samples). "
        f"{len(mapped)} mapped ({mapped_total:,} samples)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Map conditions to ICD-10-CM codes.")
    parser.add_argument("--meta", required=True, type=Path, help="Path to metadata CSV")
    parser.add_argument(
        "--condition_col",
        default="condition",
        help="Column name containing condition labels.",
    )
    parser.add_argument(
        "--icd_table",
        type=Path,
        default=Path("assets/icd10/icd10cm_tabular_2026.csv"),
        help="Flattened ICD-10-CM CSV produced by parse_icd10_tabular.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/condition_icd_mapping.csv"),
        help="Destination CSV for mapping results.",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("results/condition_icd_overrides.csv"),
        help="Optional CSV with manual overrides (condition_raw, icd_code).",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=90.0,
        help="Minimum RapidFuzz score to accept a fuzzy match.",
    )
    parser.add_argument(
        "--suggestions",
        type=int,
        default=5,
        help="Number of fuzzy suggestions to record per condition.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of unique conditions to process (for testing).",
    )
    parser.add_argument(
        "--allowed-chapters",
        default="12,2,1,4,17",
        help="Comma-separated chapter numbers to search (default: common derm-related chapters 12,2,1,4,17). Use 'all' to disable filtering.",
    )
    args = parser.parse_args()

    if not args.meta.exists():
        raise FileNotFoundError(args.meta)
    if not args.icd_table.exists():
        raise FileNotFoundError(args.icd_table)

    counts = read_condition_counts(args.meta, args.condition_col)
    if args.limit is not None and args.limit > 0:
        counts = counts.head(args.limit)
    if counts.empty:
        raise RuntimeError("No conditions found in metadata.")

    if args.allowed_chapters.lower() == "all":
        allowed = None
    else:
        allowed = [ch.strip() for ch in args.allowed_chapters.split(",") if ch.strip()]

    mapper = ConditionICDMapper.from_csv(
        args.icd_table,
        min_score=args.min_score,
        max_suggestions=args.suggestions,
        allowed_chapters=allowed,
    )
    overrides = mapper.load_overrides(args.overrides)

    iterable = counts.items()
    progress = (
        tqdm(iterable, total=len(counts), desc="Mapping conditions")
        if tqdm is not None
        else iterable
    )

    rows = []
    for condition, count in progress:
        match = mapper.match_condition(condition, overrides=overrides)
        match["count"] = int(count)
        rows.append(match)

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"Wrote mapping to {args.output}")
    print(format_summary(result_df))


if __name__ == "__main__":
    main()
