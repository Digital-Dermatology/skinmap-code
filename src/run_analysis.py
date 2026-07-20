import argparse
import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from src.analysis_plus import dataset_domain_shift, l2_normalize, yearly_novelty

AGE_BINS = [0, 18, 30, 50, 70, 120]
AGE_LABELS = ["0-17", "18-29", "30-49", "50-69", "70+"]
PREVALENCE_FEATURES: Dict[str, str] = {
    "fitzpatrick_pred": "Fitzpatrick Skin Type",
    "gender_pred": "Gender",
    "origin_pred": "Origin",
    "origin_continent": "Origin (Continent)",
    "body_region_pred": "Body Region",
    "laterality_pred": "Laterality",
    "age_pred_bin": "Age",
    "modality": "Modality",
}

# Comprehensive country-to-continent mapping
COUNTRY_TO_CONTINENT: Dict[str, str] = {
    # Africa
    "algeria": "Africa",
    "angola": "Africa",
    "benin": "Africa",
    "botswana": "Africa",
    "burkina faso": "Africa",
    "burundi": "Africa",
    "cameroon": "Africa",
    "cape verde": "Africa",
    "central african republic": "Africa",
    "chad": "Africa",
    "comoros": "Africa",
    "congo": "Africa",
    "democratic republic of the congo": "Africa",
    "djibouti": "Africa",
    "egypt": "Africa",
    "equatorial guinea": "Africa",
    "eritrea": "Africa",
    "ethiopia": "Africa",
    "gabon": "Africa",
    "gambia": "Africa",
    "ghana": "Africa",
    "guinea": "Africa",
    "guinea-bissau": "Africa",
    "ivory coast": "Africa",
    "kenya": "Africa",
    "lesotho": "Africa",
    "liberia": "Africa",
    "libya": "Africa",
    "madagascar": "Africa",
    "malawi": "Africa",
    "mali": "Africa",
    "mauritania": "Africa",
    "mauritius": "Africa",
    "morocco": "Africa",
    "mozambique": "Africa",
    "namibia": "Africa",
    "niger": "Africa",
    "nigeria": "Africa",
    "rwanda": "Africa",
    "sao tome and principe": "Africa",
    "senegal": "Africa",
    "seychelles": "Africa",
    "sierra leone": "Africa",
    "somalia": "Africa",
    "south africa": "Africa",
    "south sudan": "Africa",
    "sudan": "Africa",
    "swaziland": "Africa",
    "tanzania": "Africa",
    "togo": "Africa",
    "tunisia": "Africa",
    "uganda": "Africa",
    "zambia": "Africa",
    "zimbabwe": "Africa",
    # Asia
    "afghanistan": "Asia",
    "armenia": "Asia",
    "azerbaijan": "Asia",
    "bahrain": "Asia",
    "bangladesh": "Asia",
    "bhutan": "Asia",
    "brunei": "Asia",
    "cambodia": "Asia",
    "china": "Asia",
    "georgia": "Asia",
    "india": "Asia",
    "indonesia": "Asia",
    "iran": "Asia",
    "iraq": "Asia",
    "israel": "Asia",
    "japan": "Asia",
    "jordan": "Asia",
    "kazakhstan": "Asia",
    "kuwait": "Asia",
    "kyrgyzstan": "Asia",
    "laos": "Asia",
    "lebanon": "Asia",
    "malaysia": "Asia",
    "maldives": "Asia",
    "mongolia": "Asia",
    "myanmar": "Asia",
    "nepal": "Asia",
    "north korea": "Asia",
    "oman": "Asia",
    "pakistan": "Asia",
    "palestine": "Asia",
    "philippines": "Asia",
    "qatar": "Asia",
    "saudi arabia": "Asia",
    "singapore": "Asia",
    "south korea": "Asia",
    "sri lanka": "Asia",
    "syria": "Asia",
    "tajikistan": "Asia",
    "thailand": "Asia",
    "timor-leste": "Asia",
    "turkey": "Asia",
    "turkmenistan": "Asia",
    "united arab emirates": "Asia",
    "uzbekistan": "Asia",
    "vietnam": "Asia",
    "yemen": "Asia",
    # Europe
    "albania": "Europe",
    "andorra": "Europe",
    "austria": "Europe",
    "belarus": "Europe",
    "belgium": "Europe",
    "bosnia and herzegovina": "Europe",
    "bulgaria": "Europe",
    "croatia": "Europe",
    "cyprus": "Europe",
    "czech republic": "Europe",
    "denmark": "Europe",
    "estonia": "Europe",
    "finland": "Europe",
    "france": "Europe",
    "germany": "Europe",
    "greece": "Europe",
    "hungary": "Europe",
    "iceland": "Europe",
    "ireland": "Europe",
    "italy": "Europe",
    "kosovo": "Europe",
    "latvia": "Europe",
    "liechtenstein": "Europe",
    "lithuania": "Europe",
    "luxembourg": "Europe",
    "macedonia": "Europe",
    "malta": "Europe",
    "moldova": "Europe",
    "monaco": "Europe",
    "montenegro": "Europe",
    "netherlands": "Europe",
    "norway": "Europe",
    "poland": "Europe",
    "portugal": "Europe",
    "romania": "Europe",
    "russia": "Europe",
    "san marino": "Europe",
    "serbia": "Europe",
    "slovakia": "Europe",
    "slovenia": "Europe",
    "spain": "Europe",
    "sweden": "Europe",
    "switzerland": "Europe",
    "ukraine": "Europe",
    "united kingdom": "Europe",
    "vatican city": "Europe",
    # North America
    "antigua and barbuda": "North America",
    "bahamas": "North America",
    "barbados": "North America",
    "belize": "North America",
    "canada": "North America",
    "costa rica": "North America",
    "cuba": "North America",
    "dominica": "North America",
    "dominican republic": "North America",
    "el salvador": "North America",
    "grenada": "North America",
    "guatemala": "North America",
    "haiti": "North America",
    "honduras": "North America",
    "jamaica": "North America",
    "mexico": "North America",
    "nicaragua": "North America",
    "panama": "North America",
    "saint kitts and nevis": "North America",
    "saint lucia": "North America",
    "saint vincent and the grenadines": "North America",
    "trinidad and tobago": "North America",
    "united states": "North America",
    "usa": "North America",
    # South America
    "argentina": "South America",
    "bolivia": "South America",
    "brazil": "South America",
    "chile": "South America",
    "colombia": "South America",
    "ecuador": "South America",
    "guyana": "South America",
    "paraguay": "South America",
    "peru": "South America",
    "suriname": "South America",
    "uruguay": "South America",
    "venezuela": "South America",
    # Oceania
    "australia": "Oceania",
    "fiji": "Oceania",
    "kiribati": "Oceania",
    "marshall islands": "Oceania",
    "micronesia": "Oceania",
    "nauru": "Oceania",
    "new zealand": "Oceania",
    "palau": "Oceania",
    "papua new guinea": "Oceania",
    "samoa": "Oceania",
    "solomon islands": "Oceania",
    "tonga": "Oceania",
    "tuvalu": "Oceania",
    "vanuatu": "Oceania",
}

# Maximum categories to display before consolidating into "Other"
MAX_CATEGORIES_DISPLAY = 15


def load_icd_mapping(path: str) -> Optional[pd.DataFrame]:
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


def coerce_year_series(meta: pd.DataFrame, year_col: str) -> Optional[pd.Series]:
    if year_col not in meta.columns:
        logging.warning("Year column '%s' not found in metadata.", year_col)
        return None
    year_series = pd.to_numeric(meta[year_col], errors="coerce")
    if year_series.notna().sum() == 0:
        logging.warning("No numeric values found in %s column.", year_col)
        return None
    return year_series


def annotate_icd(
    meta: pd.DataFrame, mapping: pd.DataFrame, condition_col: str
) -> pd.DataFrame:
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


def run_icd_analysis(
    meta: pd.DataFrame,
    year_int: pd.Series,
    Xn: np.ndarray,
    args,
    csv_path,
    should_skip=None,
    subset_name: Optional[str] = None,
    subset_mask: Optional[pd.Series] = None,
) -> None:
    """
    Run ICD analysis on a subset of data.

    Args:
        meta: Full metadata DataFrame
        year_int: Year series (same length as meta)
        Xn: Normalized embeddings (same length as meta)
        args: Command line arguments
        csv_path: Function to generate output paths
        subset_name: Optional name for this subset (e.g., "TBP", "dermoscopy")
        subset_mask: Optional boolean mask to filter data (same length as meta)
    """
    # Default should_skip function if none provided
    if should_skip is None:
        should_skip = lambda name, label=None: False

    if "icd_code" not in meta.columns:
        logging.warning("ICD columns missing; skipping ICD analysis.")
        return
    year_valid = year_int.notna()
    icd_valid = meta["icd_code"].astype(str).str.strip() != ""
    base_mask = year_valid & icd_valid

    # Apply subset mask if provided
    if subset_mask is not None:
        base_mask = base_mask & subset_mask
    if base_mask.sum() == 0:
        subset_desc = f" for {subset_name}" if subset_name else ""
        logging.warning(
            f"No samples with both year and ICD mapping{subset_desc}; skipping ICD analysis."
        )
        return
    dataset_col = args.dataset_col if args.dataset_col in meta.columns else None
    dataset_series = None
    if dataset_col:
        dataset_series = (
            meta[dataset_col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    icd_df = meta.loc[
        base_mask,
        [
            args.condition_col,
            "icd_code",
            "icd_description",
            "icd_chapter",
            "icd_chapter_title",
            "icd_chapter_label",
            "icd_block",
            "icd_block_desc",
            "icd_block_label",
            "icd_category",
            "icd_category_desc",
            "icd_category_label",
        ],
    ].copy()
    icd_df["year"] = year_int[base_mask].astype(int)
    icd_df = icd_df.rename(columns={args.condition_col: "condition_label"})
    if dataset_series is not None:
        icd_df["dataset_label"] = dataset_series[base_mask].to_numpy()

    subset_desc = f" for {subset_name}" if subset_name else ""
    logging.info(
        f"ICD analysis{subset_desc} covering {len(icd_df)} samples across {icd_df['year'].nunique()} years."
    )

    code_activity = (
        icd_df.groupby("icd_code")
        .agg(
            icd_description=("icd_description", "first"),
            icd_chapter=("icd_chapter", "first"),
            icd_chapter_title=("icd_chapter_title", "first"),
            icd_chapter_label=("icd_chapter_label", "first"),
            icd_block=("icd_block", "first"),
            icd_block_label=("icd_block_label", "first"),
            icd_block_desc=("icd_block_desc", "first"),
            icd_category=("icd_category", "first"),
            icd_category_label=("icd_category_label", "first"),
            total_samples=("icd_code", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )
    total_samples = max(int(code_activity["total_samples"].sum()), 1)
    code_activity["prevalence_share"] = code_activity["total_samples"] / total_samples

    max_year = int(code_activity["last_year"].max())
    orphan_cutoff = max_year - int(args.icd_orphan_gap)
    code_activity["is_orphan"] = code_activity["last_year"] <= orphan_cutoff

    code_activity.to_csv(csv_path("icd_code_activity.csv"), index=False)
    code_activity[code_activity["is_orphan"]].sort_values("last_year").to_csv(
        csv_path("icd_orphan_codes.csv"), index=False
    )

    first_year_map = code_activity.set_index("icd_code")["first_year"]
    icd_df["is_new_code"] = icd_df["icd_code"].map(first_year_map).eq(icd_df["year"])

    # Yearly ICD Summary
    if not should_skip("yearly_icd_summary.csv"):
        yearly_summary = (
            icd_df.groupby("year")
            .agg(
                total_samples=("icd_code", "size"),
                unique_conditions=("condition_label", "nunique"),
                unique_icd_codes=("icd_code", "nunique"),
                new_icd_samples=("is_new_code", "sum"),
            )
            .reset_index()
            .sort_values("year")
        )
        new_code_counts = (
            code_activity.groupby("first_year")["icd_code"]
            .nunique()
            .rename("new_icd_codes")
        )
        yearly_summary = (
            yearly_summary.merge(
                new_code_counts, left_on="year", right_index=True, how="left"
            )
            .fillna({"new_icd_codes": 0})
            .sort_values("year")
        )
        yearly_summary["new_icd_sample_share"] = (
            yearly_summary["new_icd_samples"].astype(float)
            / yearly_summary["total_samples"].replace(0, np.nan)
        ).fillna(0.0)
        yearly_summary.to_csv(csv_path("yearly_icd_summary.csv"), index=False)

    level_specs = [
        (
            "icd_chapter_label",
            ["icd_chapter", "icd_chapter_title"],
            "yearly_icd_chapter_counts.csv",
        ),
        (
            "icd_block_label",
            ["icd_block", "icd_block_desc"],
            "yearly_icd_block_counts.csv",
        ),
        (
            "icd_category_label",
            ["icd_category", "icd_category_desc"],
            "yearly_icd_category_counts.csv",
        ),
    ]

    def save_level_counts(label_col, meta_cols, filename):
        if should_skip(filename):
            return
        df = (
            icd_df.groupby(["year", label_col])
            .size()
            .reset_index(name="count")
            .sort_values(["year", "count"], ascending=[True, False])
        )
        meta_info = icd_df.groupby(label_col)[meta_cols].first().reset_index()
        df = df.merge(meta_info, on=label_col, how="left")
        df.to_csv(csv_path(filename), index=False)

    for label_col, meta_cols, fname in level_specs:
        save_level_counts(label_col, meta_cols, fname)

    def novelty_by_group(
        label_col: str, extra_cols: list[str], filename: str
    ) -> pd.DataFrame:
        if should_skip(filename):
            return pd.DataFrame()
        rows = []
        unique_labels = (
            meta.loc[base_mask, label_col]
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        for label in unique_labels:
            label_str = str(label)
            if label_str.lower() == "unmapped":
                continue
            mask = base_mask & (meta[label_col] == label)
            if mask.sum() < args.icd_min_group_samples:
                continue
            years = year_int[mask].dropna()
            if years.nunique() < 2:
                continue
            subset_desc = f" ({subset_name})" if subset_name else ""
            logging.info(
                f"ICD novelty for {label_str}{subset_desc} ({mask.sum()} samples)"
            )
            mask_array = mask.to_numpy(dtype=bool)
            result = yearly_novelty(
                Xn[mask_array],
                years.astype(int).to_numpy(),
                k=args.k,
                n_bootstrap=args.novelty_bootstrap_reps,
                alpha=args.novelty_alpha,
                random_state=args.novelty_seed,
                bootstrap_sample_size=args.novelty_bootstrap_sample_size,
                max_bootstrap_queries=(
                    None
                    if args.novelty_max_baseline_queries == 0
                    else args.novelty_max_baseline_queries
                ),
            )
            for col in dict.fromkeys(extra_cols + [label_col]):
                val = meta.loc[mask, col].iloc[0]
                result[col] = val
            rows.append(result)
        if rows:
            df = pd.concat(rows, ignore_index=True)
            df.to_csv(csv_path(filename), index=False)
            return df
        return pd.DataFrame()

    novelty_by_group(
        "icd_chapter_label",
        ["icd_chapter", "icd_chapter_title"],
        "yearly_novelty_icd_chapter.csv",
    )
    novelty_block = novelty_by_group(
        "icd_block_label",
        ["icd_block", "icd_block_desc"],
        "yearly_novelty_icd_block.csv",
    )
    novelty_by_group(
        "icd_category_label",
        ["icd_category", "icd_category_desc"],
        "yearly_novelty_icd_category.csv",
    )
    if not novelty_block.empty:
        block_stats = []
        block_totals = icd_df.groupby("icd_block_label").size()
        dataset_count_map = (
            icd_df.groupby("icd_block_label")["dataset_label"].nunique()
            if dataset_series is not None
            else None
        )
        for block_label, block_df in novelty_block.groupby("icd_block_label"):
            years = block_df["year"].to_numpy()
            novelty_values = block_df["norm_novelty"].to_numpy()
            if len(years) >= 2:
                slope = float(np.polyfit(years, novelty_values, 1)[0])
            else:
                slope = 0.0
            mean_novelty = float(np.nanmean(novelty_values))
            latest_row = block_df.sort_values("year").iloc[-1]
            block_stats.append(
                {
                    "icd_block_label": block_label,
                    "icd_block": block_df["icd_block"].iloc[0],
                    "icd_block_desc": block_df["icd_block_desc"].iloc[0],
                    "sample_count": int(block_totals.get(block_label, 0)),
                    "prevalence_share": float(
                        block_totals.get(block_label, 0) / total_samples
                    ),
                    "dataset_count": (
                        int(dataset_count_map.get(block_label, 0))
                        if dataset_count_map is not None
                        else np.nan
                    ),
                    "mean_norm_novelty": mean_novelty,
                    "novelty_trend": slope,
                    "latest_year": int(latest_row["year"]),
                    "latest_norm_novelty": float(latest_row["norm_novelty"]),
                }
            )
        pd.DataFrame(block_stats).to_csv(
            csv_path("icd_block_novelty_stats.csv"), index=False
        )

    if dataset_series is not None:
        block_dataset_total = (
            icd_df.groupby(["icd_block_label", "dataset_label"])
            .size()
            .reset_index(name="count")
        )
        block_dataset_total.to_csv(
            csv_path("icd_block_dataset_total_counts.csv"), index=False
        )
        block_dataset_year = (
            icd_df.groupby(["year", "icd_block_label", "dataset_label"])
            .size()
            .reset_index(name="count")
            .sort_values(["icd_block_label", "year"])
        )
        block_dataset_year.to_csv(
            csv_path("icd_block_dataset_year_counts.csv"), index=False
        )
        block_coverage = (
            icd_df.groupby("icd_block_label")
            .agg(
                sample_count=("icd_block_label", "size"),
                dataset_count=(
                    "dataset_label",
                    lambda s: s.replace("Unknown", np.nan).nunique(),
                ),
            )
            .reset_index()
        )
        block_coverage.to_csv(csv_path("icd_block_coverage.csv"), index=False)

        if args.icd_max_blocks_for_shift > 0 and not should_skip(
            "icd_block_dataset_shift.csv"
        ):
            block_sizes = block_coverage.sort_values(
                "sample_count", ascending=False
            ).head(args.icd_max_blocks_for_shift)
            shift_rows = []
            for _, row in block_sizes.iterrows():
                block_label = row["icd_block_label"]
                mask = base_mask & (meta["icd_block_label"] == block_label)
                block_count = int(mask.sum())
                if block_count < args.icd_min_shift_samples:
                    continue
                datasets_subset = dataset_series[mask].to_numpy()
                unique_datasets = np.unique(datasets_subset)
                if len(unique_datasets) < 2:
                    continue
                subset_desc = f" ({subset_name})" if subset_name else ""
                logging.info(
                    f"Computing cross-dataset shift for block {block_label}{subset_desc}"
                )
                block_X = Xn[mask.to_numpy(dtype=bool)]
                shift_df = dataset_domain_shift(block_X, datasets_subset)
                shift_df["icd_block_label"] = block_label
                shift_df["sample_count"] = block_count
                shift_df["dataset_pairs"] = len(shift_df)
                shift_rows.append(shift_df)
            if shift_rows:
                pd.concat(shift_rows, ignore_index=True).to_csv(
                    csv_path("icd_block_dataset_shift.csv"), index=False
                )


def run_icd_analysis_by_modality(
    meta: pd.DataFrame,
    year_int: pd.Series,
    Xn: np.ndarray,
    args,
    outdir_path: Path,
    modality_col: str = "modality",
    should_skip=None,
) -> None:
    """
    Run ICD analysis grouped by modality, generating separate outputs for each.

    Creates subdirectories like:
        analysis/icd_by_modality/TBP/
        analysis/icd_by_modality/dermoscopy/
        analysis/icd_by_modality/clinical/
    """
    if modality_col not in meta.columns:
        logging.warning(
            f"Modality column '{modality_col}' not found; skipping modality-grouped ICD analysis."
        )
        return

    # Filter out NaN values (both actual NaN and string "nan")
    modalities_series = meta[modality_col].dropna()
    modalities_series = modalities_series[
        modalities_series.astype(str).str.lower() != "nan"
    ]
    modalities = modalities_series.unique()

    if len(modalities) == 0:
        logging.warning("No modalities found; skipping modality-grouped ICD analysis.")
        return

    modality_list = ", ".join(sorted(str(m) for m in modalities))
    logging.info(
        f"Running ICD analysis grouped by {len(modalities)} modalities: {modality_list}"
    )

    for modality in sorted(modalities):
        modality_str = str(modality)
        # Skip if modality is string "nan" (additional safety check)
        if modality_str.lower() == "nan":
            continue
        modality_mask = meta[modality_col] == modality
        n_samples = modality_mask.sum()

        logging.info(f"Processing modality '{modality_str}' ({n_samples} samples)...")

        # Create subdirectory for this modality
        modality_dir = outdir_path / "icd_by_modality" / modality_str
        modality_dir.mkdir(parents=True, exist_ok=True)

        def modality_csv_path(name: str) -> str:
            return str(modality_dir / name)

        def modality_should_skip(name, label=None):
            if should_skip is None:
                return False
            target = modality_csv_path(name)
            display = label or name
            skip = not args.force_recompute and os.path.exists(target)
            if skip:
                logging.info("Skipping %s (already exists)", display)
            return skip

        # Run ICD analysis for this modality subset
        run_icd_analysis(
            meta=meta,
            year_int=year_int,
            Xn=Xn,
            args=args,
            csv_path=modality_csv_path,
            should_skip=modality_should_skip,
            subset_name=modality_str,
            subset_mask=modality_mask,
        )

    logging.info("Completed modality-grouped ICD analysis.")


def run_icd_analysis_by_fst(
    meta: pd.DataFrame,
    year_int: pd.Series,
    Xn: np.ndarray,
    args,
    outdir_path: Path,
    fst_col: str = "fitzpatrick_grouped_pred",
    should_skip=None,
) -> None:
    """
    Run ICD analysis grouped by Fitzpatrick Skin Type, generating separate outputs for each.

    Creates subdirectories like:
        analysis/icd_by_fst_grouped/1-2/
        analysis/icd_by_fst_grouped/3-4/
        analysis/icd_by_fst_grouped/5-6/
    """
    if fst_col not in meta.columns:
        logging.warning(
            f"FST column '{fst_col}' not found; skipping FST-grouped ICD analysis."
        )
        return

    # Filter out NaN values (both actual NaN and string "nan")
    fst_series = meta[fst_col].dropna()
    fst_series = fst_series[fst_series.astype(str).str.lower() != "nan"]
    fst_types = fst_series.unique()

    if len(fst_types) == 0:
        logging.warning("No FST types found; skipping FST-grouped ICD analysis.")
        return

    fst_list = ", ".join(sorted(str(f) for f in fst_types))
    logging.info(
        f"Running ICD analysis grouped by {len(fst_types)} FST groups: {fst_list}"
    )

    for fst_type in sorted(fst_types):
        fst_str = str(fst_type)
        # Skip if FST is string "nan" (additional safety check)
        if fst_str.lower() == "nan":
            continue
        fst_mask = meta[fst_col] == fst_type
        n_samples = fst_mask.sum()

        logging.info(f"Processing FST group '{fst_str}' ({n_samples} samples)...")

        # Create subdirectory for this FST group
        fst_dir = outdir_path / "icd_by_fst_grouped" / fst_str
        fst_dir.mkdir(parents=True, exist_ok=True)

        def fst_csv_path(name: str) -> str:
            return str(fst_dir / name)

        def fst_should_skip(name, label=None):
            if should_skip is None:
                return False
            target = fst_csv_path(name)
            display = label or name
            skip = not args.force_recompute and os.path.exists(target)
            if skip:
                logging.info("Skipping %s (already exists)", display)
            return skip

        # Run ICD analysis for this FST subset
        run_icd_analysis(
            meta=meta,
            year_int=year_int,
            Xn=Xn,
            args=args,
            csv_path=fst_csv_path,
            should_skip=fst_should_skip,
            subset_name=fst_str,
            subset_mask=fst_mask,
        )

    logging.info("Completed FST-grouped ICD analysis.")


def load_embeddings(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        key = "image_embeddings" if "image_embeddings" in data.files else data.files[0]
        return data[key]
    else:
        raise ValueError("Unsupported embedding format; use .npy or .npz")


def assign_age_bins(meta: pd.DataFrame, source_col: str = "age_pred") -> None:
    if source_col not in meta.columns:
        return
    bins = pd.cut(
        meta[source_col],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
        right=False,
    )
    labels = bins.astype(str)
    labels = labels.where(labels != "nan", "Unknown")
    meta["age_pred_bin"] = pd.Categorical(
        labels, categories=AGE_LABELS + ["Unknown"], ordered=True
    )


def map_origin_to_continent(
    meta: pd.DataFrame,
    source_col: str = "origin_pred",
) -> None:
    """
    Map origin (country) values to continents.
    Creates a new column 'origin_continent' in the metadata.
    """
    if source_col not in meta.columns:
        logging.warning(
            "Source column '%s' not found for continent mapping.", source_col
        )
        return

    def get_continent(origin_val):
        if str(origin_val) == "None" or str(origin_val) == "nan":
            return "Unknown"
        # Normalize the value
        origin_str = str(origin_val).strip().lower()
        if not origin_str or origin_str == "unknown":
            return "Unknown"
        # Look up in mapping
        continent = COUNTRY_TO_CONTINENT.get(origin_str, "Other")
        return continent

    meta["origin_continent"] = meta[source_col].apply(get_continent)
    logging.info("Mapped %s to continents.", source_col)


def consolidate_categories(
    series: pd.Series,
    max_categories: int = MAX_CATEGORIES_DISPLAY,
    unknown_label: str = "Unknown",
) -> pd.Series:
    """
    Consolidate a categorical series by keeping the top N categories
    and grouping the rest as 'Other'. Preserves 'Unknown' category.

    Args:
        series: The categorical series to consolidate
        max_categories: Maximum number of categories to keep (excluding Unknown and Other)
        unknown_label: Label used for unknown/missing values

    Returns:
        Series with consolidated categories
    """
    if series.empty:
        return series

    # Count frequencies, excluding Unknown
    value_counts = series[series != unknown_label].value_counts()

    if len(value_counts) <= max_categories:
        return series

    # Keep top N categories
    top_categories = value_counts.head(max_categories).index.tolist()

    # Create consolidated series
    def consolidate_value(val):
        if pd.isna(val) or val == unknown_label:
            return unknown_label
        elif val in top_categories:
            return val
        else:
            return "Other"

    result = series.apply(consolidate_value)
    logging.info(
        "Consolidated %s from %d to %d categories (+ Unknown/Other)",
        series.name or "series",
        len(value_counts),
        max_categories,
    )
    return result


def load_metadata(parquet_path: Path) -> Optional[pd.DataFrame]:
    if not parquet_path.exists():
        logging.warning("Predicted metadata parquet not found: %s", parquet_path)
        return None
    pred_df = pd.read_parquet(parquet_path)
    if "img_path" not in pred_df.columns:
        logging.warning("Predicted metadata missing img_path column.")
        return None
    return pred_df


def sanitize_series(series: pd.Series, unknown_label: str = "Unknown") -> pd.Series:
    if series.empty:
        return series
    values = series.astype(str).str.strip()
    values = values.replace("", unknown_label)
    values = values.replace("nan", unknown_label)
    values = values.replace(
        "None", unknown_label
    )  # Handle Python None converted to string
    values = values.fillna(unknown_label)
    return values


def explode_multilabel_field(
    meta: pd.DataFrame, field: str, unknown_label: str = "Unknown"
) -> pd.DataFrame:
    """
    Explode a multilabel field (list) into separate rows for prevalence calculation.
    Preserves the original index for alignment with other fields.

    Args:
        meta: DataFrame containing the field
        field: Name of the multilabel field to explode
        unknown_label: Label to use for unknown/missing values

    Returns:
        DataFrame with the multilabel field exploded into separate rows (index preserved)
    """
    if field not in meta.columns:
        return meta

    # Create a copy to avoid modifying the original
    df = meta[[field]].copy()

    # Convert string representations of lists to actual lists
    def parse_list_field(val):
        # Handle arrays/lists that are already parsed
        if isinstance(val, (list, np.ndarray)):
            # If it's already a list/array, convert to list and filter valid values
            cleaned = [
                str(x).strip()
                for x in val
                if x is not None
                and str(x).strip()
                and str(x).strip().lower() not in ("none", "nan")
            ]
            return cleaned if cleaned else [unknown_label]

        # For scalars, check if it's NaN
        if pd.isna(val):
            return [unknown_label]

        # Handle string representation of lists
        val_str = str(val).strip()
        if not val_str or val_str == "nan" or val_str == "None":
            return [unknown_label]

        # Check if it's a list-like string
        if val_str.startswith("[") and val_str.endswith("]"):
            try:
                # Parse the string as a list
                import ast

                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Filter out empty strings and None values
                    cleaned = [
                        str(x).strip()
                        for x in parsed
                        if x and str(x).strip() and str(x).strip().lower() != "none"
                    ]
                    return cleaned if cleaned else [unknown_label]
                else:
                    return [unknown_label]
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single value
                return [val_str]
        else:
            # Single value, not a list
            return [val_str]

    df[field] = df[field].apply(parse_list_field)

    # Explode the list into separate rows, keeping the original index
    df_exploded = df.explode(field)

    # Sanitize the exploded values
    df_exploded[field] = sanitize_series(df_exploded[field], unknown_label)

    return df_exploded


def reorder_categories(
    values: Sequence[str], preferred_order: Optional[Sequence[str]] = None
) -> List[str]:
    unique_vals = [v for v in values if pd.notna(v)]
    seen = []
    for val in unique_vals:
        if val not in seen:
            seen.append(val)
    if not preferred_order:
        return sorted(seen)
    ordered = [item for item in preferred_order if item in seen]
    remaining = [item for item in seen if item not in ordered]
    return ordered + sorted(remaining)


def save_heatmap(
    data: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = "YlOrRd",
) -> None:
    """
    Save a heatmap with adaptive sizing based on the number of categories.

    Args:
        data: DataFrame with prevalence data
        title: Plot title
        output_path: Path to save the figure
        cmap: Colormap to use
    """
    n_rows, n_cols = data.shape

    # Adaptive sizing: smaller cells for larger heatmaps
    if n_rows <= 10 and n_cols <= 10:
        # Small heatmap: larger cells
        cell_width, cell_height = 0.8, 0.6
        font_size = 9
    elif n_rows <= 20 and n_cols <= 20:
        # Medium heatmap: medium cells
        cell_width, cell_height = 0.6, 0.5
        font_size = 8
    else:
        # Large heatmap: smaller cells
        cell_width, cell_height = 0.5, 0.4
        font_size = 7

    # Calculate figure size with min/max bounds
    fig_width = max(6, min(20, cell_width * n_cols + 2))
    fig_height = max(4, min(16, cell_height * n_rows + 1.5))

    plt.figure(figsize=(fig_width, fig_height))

    # Adjust annotation based on size
    show_annot = n_rows * n_cols <= 300  # Only show numbers if not too many cells

    sns.heatmap(
        data,
        annot=show_annot,
        fmt=".1f" if show_annot else "",
        cmap=cmap,
        linewidths=0.5 if n_rows * n_cols <= 400 else 0.2,
        linecolor="white",
        cbar_kws={"label": "Prevalence (%)"},
        annot_kws={"fontsize": font_size} if show_annot else {},
    )

    plt.title(title, fontsize=11, fontweight="bold", pad=10)
    plt.xlabel(data.columns.name or "", fontsize=10)
    plt.ylabel(data.index.name or "", fontsize=10)

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


def run_prevalence_cross_sections(
    meta: pd.DataFrame,
    csv_path_fn,
    should_skip_fn,
    feature_names: Dict[str, str],
    output_dir: Path,
) -> None:
    """
    Generate cross-sectional prevalence heatmaps for all feature pairs.
    Automatically consolidates features with many categories.
    Handles multilabel fields (like origin_pred) by exploding them before counting.
    """
    sentinel = "prevalence_cross_sections.done"
    if should_skip_fn(sentinel, "prevalence cross-sections"):
        return
    available_features = [col for col in feature_names if col in meta.columns]
    if len(available_features) < 2:
        logging.warning("Not enough metadata features for cross-prevalence plots.")
        return
    prevalence_dir = Path(output_dir) / "prevalence"
    prevalence_dir.mkdir(parents=True, exist_ok=True)

    # Features that should be consolidated if they have many categories
    features_to_consolidate = ["body_region_pred", "origin_pred"]

    # Multilabel features that need to be exploded before counting
    multilabel_features = ["origin_pred"]

    for a, b in itertools.combinations(available_features, 2):
        # Check if either feature is multilabel
        a_is_multilabel = a in multilabel_features
        b_is_multilabel = b in multilabel_features

        # If at least one feature is multilabel, we need special handling
        if a_is_multilabel or b_is_multilabel:
            logging.info("Processing multilabel prevalence for %s vs %s", a, b)

            if a_is_multilabel and b_is_multilabel:
                # Both are multilabel - create all combinations per row
                a_exploded = explode_multilabel_field(meta, a)
                b_exploded = explode_multilabel_field(meta, b)

                # Create all combinations for each original row
                rows = []
                for idx in meta.index:
                    a_vals = a_exploded.loc[a_exploded.index == idx, a].tolist()
                    b_vals = b_exploded.loc[b_exploded.index == idx, b].tolist()
                    for a_val in a_vals:
                        for b_val in b_vals:
                            rows.append({a: a_val, b: b_val})
                work_df = pd.DataFrame(rows)
                series_a = work_df[a].reset_index(drop=True)
                series_b = work_df[b].reset_index(drop=True)

            elif a_is_multilabel:
                # Only a is multilabel
                a_exploded = explode_multilabel_field(meta, a)
                # Align b with the exploded a using the index
                b_aligned = meta.loc[a_exploded.index, b]
                series_a = a_exploded[a].reset_index(drop=True)
                series_b = sanitize_series(b_aligned).reset_index(drop=True)

            else:  # b_is_multilabel
                # Only b is multilabel
                b_exploded = explode_multilabel_field(meta, b)
                # Align a with the exploded b using the index
                a_aligned = meta.loc[b_exploded.index, a]
                series_a = sanitize_series(a_aligned).reset_index(drop=True)
                series_b = b_exploded[b].reset_index(drop=True)

        else:
            # Neither is multilabel, use normal handling
            series_a = sanitize_series(meta[a])
            series_b = sanitize_series(meta[b])

        # Consolidate if feature has many categories (including after exploding multilabel)
        if a in features_to_consolidate:
            n_cats_a = series_a[series_a != "Unknown"].nunique()
            if n_cats_a > MAX_CATEGORIES_DISPLAY:
                logging.info(
                    "Consolidating %s (%d categories) for cross-section with %s",
                    a,
                    n_cats_a,
                    b,
                )
                series_a = consolidate_categories(series_a)

        if b in features_to_consolidate:
            n_cats_b = series_b[series_b != "Unknown"].nunique()
            if n_cats_b > MAX_CATEGORIES_DISPLAY:
                logging.info(
                    "Consolidating %s (%d categories) for cross-section with %s",
                    b,
                    n_cats_b,
                    a,
                )
                series_b = consolidate_categories(series_b)

        series_a.name = feature_names.get(a, a)
        series_b.name = feature_names.get(b, b)
        valid_mask = (series_a != "Unknown") & (series_b != "Unknown")
        if valid_mask.sum() == 0:
            continue
        table = pd.crosstab(series_a[valid_mask], series_b[valid_mask], dropna=False)
        if table.empty:
            continue

        # Define preferred ordering for specific features
        preferred_orders = {
            "age_pred_bin": AGE_LABELS + ["Unknown", "Other"],
        }

        row_order = reorder_categories(
            table.index.tolist(),
            preferred_orders.get(a, None),
        )
        col_order = reorder_categories(
            table.columns.tolist(),
            preferred_orders.get(b, None),
        )
        table = table.reindex(index=row_order, columns=col_order, fill_value=0)
        prevalence = (table / max(1, table.to_numpy().sum())) * 100.0
        prefix = f"prevalence_{a}_vs_{b}"
        stacked = table.stack().rename("count").reset_index()
        stacked = stacked.rename(
            columns={
                stacked.columns[0]: feature_names.get(a, a),
                stacked.columns[1]: feature_names.get(b, b),
            }
        )
        stacked["prevalence_share"] = stacked["count"] / max(1, table.values.sum())
        stacked.to_csv(prevalence_dir / f"{prefix}.csv", index=False)
        heatmap_path = prevalence_dir / f"{prefix}.svg"
        title = f"{feature_names.get(a, a)} vs {feature_names.get(b, b)}"
        save_heatmap(prevalence, title, heatmap_path)
    Path(csv_path_fn(sentinel)).touch()


def summarize_mode(series: pd.Series) -> str:
    if series.empty:
        return ""
    clean = sanitize_series(series)
    counts = clean.value_counts()
    if counts.empty:
        return ""
    return counts.idxmax()


def main():
    p = argparse.ArgumentParser(description="Run analysis for a model's embeddings")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_col", default="dataset")
    p.add_argument("--year_col", default="release_year")
    p.add_argument("--condition_col", default="condition")
    p.add_argument(
        "--icd_mapping",
        default="results/condition_icd_mapping.csv",
        help="Path to condition→ICD mapping CSV (set to '' to disable ICD analysis).",
    )
    p.add_argument(
        "--icd_orphan_gap",
        type=int,
        default=3,
        help="Years of inactivity required to flag an ICD code as orphaned.",
    )
    p.add_argument(
        "--icd_min_group_samples",
        type=int,
        default=200,
        help="Minimum samples required for a chapter/block novelty curve.",
    )
    p.add_argument(
        "--icd_max_blocks_for_shift",
        type=int,
        default=8,
        help="Max number of ICD blocks (by volume) to compute cross-dataset divergence for.",
    )
    p.add_argument(
        "--icd_min_shift_samples",
        type=int,
        default=300,
        help="Minimum samples within a block required before computing cross-dataset shift.",
    )
    p.add_argument("--k", type=int, default=100)
    p.add_argument(
        "--novelty_bootstrap_reps",
        type=int,
        default=200,
        help="Number of bootstrap resamples for yearly novelty confidence intervals (0 disables).",
    )
    p.add_argument(
        "--novelty_bootstrap_sample_size",
        type=int,
        default=10000,
        help="Optional cap on sample count per bootstrap draw; defaults to using all samples for a year.",
    )
    p.add_argument(
        "--novelty_max_baseline_queries",
        type=int,
        default=100000,
        help="Optional cap on total queries used for the yearly novelty baseline (set to 0 to disable).",
    )
    p.add_argument(
        "--novelty_alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for yearly novelty confidence intervals.",
    )
    p.add_argument(
        "--novelty_seed",
        type=int,
        default=0,
        help="Random seed for yearly novelty bootstrap.",
    )
    p.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of all results, ignoring existing CSV files",
    )
    args = p.parse_args()

    model_dir = Path("assets") / args.model_name
    meta_path = model_dir / "embeddings" / "dataframe.csv"
    embeddings_path = model_dir / "embeddings" / "embeddings.npz"
    if not embeddings_path.exists():
        alt_path = embeddings_path.with_suffix(".npy")
        if alt_path.exists():
            embeddings_path = alt_path
    atlas_path = model_dir / "atlas_input.parquet"
    outdir_path = model_dir / "analysis"

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)
    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Required metadata file not found: {atlas_path}. "
            "atlas_input.parquet is produced by create_skinmap.py with --use_atlas."
        )
    outdir_path.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(atlas_path)
    if meta is None:
        raise ValueError(
            f"Failed to load metadata from {atlas_path}; the parquet is unreadable "
            "or missing the required 'img_path' column. Regenerate it with "
            "create_skinmap.py --use_atlas."
        )
    meta = meta.rename(columns={"image": "thumbnail_rel_path"})
    if "thumbnail_rel_path" not in meta.columns:
        meta["thumbnail_rel_path"] = ""
    else:
        meta["thumbnail_rel_path"] = meta["thumbnail_rel_path"].fillna("")
    assign_age_bins(meta)
    map_origin_to_continent(meta)

    X = load_embeddings(str(embeddings_path))
    Xn = l2_normalize(X)

    year_series = coerce_year_series(meta, args.year_col)
    year_int = year_series.round().astype("Int64") if year_series is not None else None

    icd_mapping_df = None
    if args.icd_mapping:
        icd_mapping_df = load_icd_mapping(args.icd_mapping)
        if icd_mapping_df is not None:
            if args.condition_col not in meta.columns:
                logging.warning(
                    "Condition column '%s' missing; cannot merge ICD mapping.",
                    args.condition_col,
                )
            else:
                meta = annotate_icd(meta, icd_mapping_df, args.condition_col)

    def csv_path(name: str) -> str:
        return str(outdir_path / name)

    def should_skip(name, label: Optional[str] = None):
        target = csv_path(name)
        display = label or name
        skip = not args.force_recompute and os.path.exists(target)
        if skip:
            logging.info("Skipping %s (already exists)", display)
        else:
            logging.info("Computing %s", display)
        return skip

    # Yearly Novelty Analysis
    if year_int is not None and not should_skip("yearly_novelty.csv"):
        valid_mask = year_int.notna()
        if valid_mask.sum() < len(meta):
            logging.info(
                "Ignoring %d samples without %s",
                len(meta) - int(valid_mask.sum()),
                args.year_col,
            )
        if valid_mask.sum() < 2:
            logging.warning(
                "Not enough samples with %s to compute yearly novelty", args.year_col
            )
        else:
            yearly_novelty(
                Xn[valid_mask.to_numpy()],
                year_int[valid_mask].astype(int).to_numpy(),
                k=args.k,
                n_bootstrap=args.novelty_bootstrap_reps,
                alpha=args.novelty_alpha,
                random_state=args.novelty_seed,
                bootstrap_sample_size=args.novelty_bootstrap_sample_size,
                max_bootstrap_queries=(
                    None
                    if args.novelty_max_baseline_queries == 0
                    else args.novelty_max_baseline_queries
                ),
            ).to_csv(csv_path("yearly_novelty.csv"), index=False)

    # Domain Shift Analysis
    if args.dataset_col in meta.columns and not should_skip("domain_shift_frechet.csv"):
        dataset_domain_shift(Xn, meta[args.dataset_col]).to_csv(
            csv_path("domain_shift_frechet.csv"), index=False
        )

    if (
        icd_mapping_df is not None
        and args.condition_col in meta.columns
        and year_int is not None
    ):
        run_icd_analysis(meta, year_int, Xn, args, csv_path, should_skip=should_skip)

        # Also run ICD analysis grouped by modality
        run_icd_analysis_by_modality(
            meta, year_int, Xn, args, outdir_path, should_skip=should_skip
        )

        # Also run ICD analysis grouped by FST (Fitzpatrick Skin Type groups: 1-2, 3-4, 5-6)
        run_icd_analysis_by_fst(
            meta, year_int, Xn, args, outdir_path, should_skip=should_skip
        )

    run_prevalence_cross_sections(
        meta, csv_path, should_skip, PREVALENCE_FEATURES, outdir_path
    )

    print("Finished run_analysis. Outputs in:", outdir_path)
    print("To generate figures, use: notebooks/08_Generate_analysis_figures.ipynb")


if __name__ == "__main__":
    main()
