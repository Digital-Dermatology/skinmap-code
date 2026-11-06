"""Load a condition to ICD mapping CSV and add its ICD columns to a metadata frame."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd


def load_icd_mapping(path: str) -> Optional[pd.DataFrame]:
    """Load a condition->ICD mapping CSV and derive the canonical ICD columns."""
    if not path:
        return None
    if not os.path.exists(path):
        logging.warning("ICD mapping file not found: %s", path)
        return None
    mapping = pd.read_csv(path).fillna("")
    if "condition_raw" not in mapping.columns:
        logging.warning("ICD mapping missing 'condition_raw' column: %s", path)
        return None
    keep_cols = [
        "condition_raw",
        "icd_code",
        "icd_description",
        "chapter",
        "chapter_title",
        "chapter_range",
        "section_id",
        "section_desc",
    ]
    missing = [c for c in keep_cols if c not in mapping.columns]
    if missing:
        logging.warning("ICD mapping missing columns: %s", ", ".join(missing))
        return None
    mapping = mapping[keep_cols].drop_duplicates("condition_raw")
    mapping = mapping.rename(
        columns={
            "chapter": "icd_chapter",
            "chapter_title": "icd_chapter_title",
            "chapter_range": "icd_chapter_range",
            "section_id": "icd_block",
            "section_desc": "icd_block_desc",
        }
    )
    mapping["icd_code"] = mapping["icd_code"].astype(str)
    mapping["icd_category"] = mapping["icd_code"].str.split(".").str[0]
    mapping["icd_category"] = mapping["icd_category"].where(
        mapping["icd_category"].str.strip().astype(bool), ""
    )
    category_desc = (
        mapping.loc[mapping["icd_category"].str.strip().astype(bool)]
        .groupby("icd_category")["icd_description"]
        .first()
    )
    mapping["icd_category_desc"] = mapping["icd_category"].map(category_desc).fillna("")
    mapping["icd_category_label"] = (
        mapping["icd_category"].astype(str).str.strip()
        + " - "
        + mapping["icd_category_desc"].astype(str).str.strip()
    ).str.strip(" -")
    return mapping


def annotate_icd(
    meta: pd.DataFrame, mapping: pd.DataFrame, condition_col: str
) -> pd.DataFrame:
    """Merge ICD columns into ``meta`` on ``condition_col`` -> ``condition_raw``.

    Idempotent (drops any pre-existing ICD columns first) and robust to a mapping
    that carries the RAW ``section_id``/``section_desc`` columns rather than the
    renamed ``icd_block``/``icd_block_desc`` (it backfills them), and derives
    ``icd_category`` from ``icd_code`` when the mapping lacks it.
    """
    icd_cols = [
        "icd_code",
        "icd_description",
        "icd_chapter",
        "icd_chapter_title",
        "icd_chapter_range",
        "icd_block",
        "icd_block_desc",
        "icd_category",
        "icd_category_desc",
        "icd_category_label",
    ]
    # Drop existing ICD columns to avoid conflicts with the merge
    existing_icd_cols = [
        col
        for col in meta.columns
        if col in icd_cols
        or col.endswith("_icdmap")
        or col in ["icd_chapter_label", "icd_block_label"]
    ]
    if existing_icd_cols:
        meta = meta.drop(columns=existing_icd_cols)

    merged = meta.merge(
        mapping,
        how="left",
        left_on=condition_col,
        right_on="condition_raw",
        suffixes=("", "_icdmap"),
    )
    if "condition_raw" in merged.columns:
        merged = merged.drop(columns=["condition_raw"])
    for col in icd_cols:
        if col not in merged.columns:
            merged[col] = ""
        merged[col] = merged[col].fillna("").astype(str)
    # Backfill block fields if the raw columns are present (e.g., section_id/section_desc from mapping CSV)
    if "icd_block" in merged.columns and "section_id" in merged.columns:
        needs_block = merged["icd_block"].eq("") & merged["section_id"].astype(
            str
        ).str.strip().ne("")
        merged.loc[needs_block, "icd_block"] = merged.loc[needs_block, "section_id"]
    if "icd_block_desc" in merged.columns and "section_desc" in merged.columns:
        needs_block_desc = merged["icd_block_desc"].eq("") & merged[
            "section_desc"
        ].astype(str).str.strip().ne("")
        merged.loc[needs_block_desc, "icd_block_desc"] = merged.loc[
            needs_block_desc, "section_desc"
        ]
    # If the mapping lacks icd_category, derive it from icd_code as in load_icd_mapping
    if "icd_category" in merged.columns:
        derived_category = merged["icd_category"]
        needs_category = derived_category.eq("") & merged["icd_code"].astype(
            str
        ).str.strip().ne("")
        if needs_category.any():
            merged.loc[needs_category, "icd_category"] = (
                merged.loc[needs_category, "icd_code"].astype(str).str.split(".").str[0]
            )
        cat_desc = (
            merged.loc[merged["icd_category"].str.strip().astype(bool)]
            .groupby("icd_category")["icd_description"]
            .first()
        )
        merged["icd_category_desc"] = merged["icd_category_desc"].replace("", np.nan)
        merged["icd_category_desc"] = merged["icd_category_desc"].fillna(
            merged["icd_category"].map(cat_desc)
        )
    merged["icd_chapter"] = merged["icd_chapter"].replace("", "Unmapped")
    merged["icd_chapter_title"] = merged["icd_chapter_title"].replace("", "Unmapped")
    merged["icd_block"] = merged["icd_block"].replace("", "Unmapped")
    merged["icd_block_desc"] = merged["icd_block_desc"].replace("", "Unmapped")
    merged["icd_category"] = merged["icd_category"].replace("", "Unmapped")
    merged["icd_category_desc"] = merged["icd_category_desc"].replace("", "Unmapped")
    merged["icd_category_label"] = merged["icd_category_label"].replace("", "Unmapped")
    merged["icd_chapter_label"] = (
        merged["icd_chapter"].astype(str).str.strip()
        + " - "
        + merged["icd_chapter_title"]
    ).str.strip(" -")
    merged["icd_block_label"] = (
        merged["icd_block"].astype(str).str.strip() + " - " + merged["icd_block_desc"]
    ).str.strip(" -")
    merged["icd_category_label"] = (
        merged["icd_category"].astype(str).str.strip()
        + " - "
        + merged["icd_category_desc"].astype(str).str.strip()
    ).str.strip(" -")
    return merged


__all__ = ["annotate_icd", "load_icd_mapping"]
