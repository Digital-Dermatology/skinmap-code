"""
Data preparation pipeline extracted from notebooks/01_Playground.ipynb.

The goal is to keep the original notebook logic intact while making it
callable from a CLI so we can reuse and test it. The steps mirror the
notebook:
1) Load/merge datasets (optional, if an existing CSV is provided skip this).
2) Merge concise captions when available.
3) Harmonize body_location into body_region + laterality.
4) Clean condition strings.
5) Merge small ISIC splits.
6) Annotate ICD using the existing mapping CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Single source of truth for ICD mapping/annotation (re-exported below for the
# rebuild pipeline and tests that import them from skinmap.pipeline.prep_data).
from skinmap.icd_mapping import annotate_icd, load_icd_mapping  # noqa: F401

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------
# Mappers copied from the notebook
# --------------------------
try:
    from core.src.datasets.helper import DatasetName, get_dataset
except Exception:  # pragma: no cover - only used when loading datasets directly
    DatasetName = None  # type: ignore
    get_dataset = None  # type: ignore


release_mapper = (
    {
        DatasetName.ISIC_2024: 2024,
        DatasetName.FITZPATRICK17K: 2021,
        DatasetName.HAM10000: 2018,
        DatasetName.SD_128: 2016,
        DatasetName.PASSION: 2024,
        DatasetName.DERM7PT: 2019,
        DatasetName.PH2: 2013,
        DatasetName.MED_NODE: 2015,
        DatasetName.SKINCAP: 2024,
        DatasetName.SCIN: 2024,
        DatasetName.DERM1M: 2025,
        DatasetName.HIBA: 2018,
        DatasetName.MSKCC: 2020,
        DatasetName.DERMNET_CSV: 2020,
        DatasetName.PAD_UFES_20: 2020,
        DatasetName.DDI: 2022,
        DatasetName.DAFFODIL: 2024,
        DatasetName.DERMACON_IN: 2025,
        DatasetName.MILK10K: 2025,
        DatasetName.BALD: 2025,
        DatasetName.DERM12345: 2024,
        DatasetName.MIDAS: 2024,
    }
    if DatasetName is not None
    else {}
)

release_mapper_name = {"DermNet": 2020, "CAN2000": 2023, "CAN5600": 2023}

modality_mapper = (
    {
        DatasetName.ISIC_2024: "TBP",
        DatasetName.FITZPATRICK17K: "clinical",
        DatasetName.HAM10000: "dermoscopy",
        DatasetName.SD_128: "clinical",
        DatasetName.PASSION: "clinical",
        DatasetName.DERM7PT: "dermoscopy",
        DatasetName.PH2: "dermoscopy",
        DatasetName.MED_NODE: "clinical",
        DatasetName.SKINCAP: "clinical",
        DatasetName.SCIN: "clinical",
        DatasetName.ALTMEYERS: "clinical",
        DatasetName.DAFFODIL: "clinical",
        DatasetName.LESION130K: "clinical",
        DatasetName.MM_SKINQA: None,  # Will be taken from metadata
        DatasetName.HIBA: "dermoscopy",
        DatasetName.MSKCC: "dermoscopy",
        DatasetName.DERMNET_CSV: "clinical",
        DatasetName.PAD_UFES_20: "clinical",
        DatasetName.DDI: "clinical",
        DatasetName.DAFFODIL: "dermoscopy",
        DatasetName.DERMACON_IN: "clinical",
        # MILK10k and MIDAS are multimodal; the loader sets a per-image
        # `modality` column (read first by build_dataframe_from_datasets).
        # These entries are only the fallback when that column is absent.
        DatasetName.MILK10K: "dermoscopy",
        DatasetName.BALD: "dermoscopy",
        DatasetName.DERM12345: "dermoscopy",
        DatasetName.MIDAS: "clinical",
    }
    if DatasetName is not None
    else {}
)

modality_mapper_name = {
    "DermNet": "clinical",
    "CAN2000": "clinical",
    "CAN5600": "clinical",
    "DermaCompass": "clinical",
}


def parse_isic_attribution(attribution: object) -> Optional[List[str]]:
    """Parse ISIC attribution field to extract country of origin."""
    if pd.isna(attribution) or attribution == "Anonymous":
        return None

    attribution = str(attribution).lower()

    patterns = {
        "United States": [
            "memorial sloan kettering",
            "msk",
            "sloan kettering",
            "university of pittsburgh",
            "pittsburgh",
        ],
        "Spain": [
            "hospital clínic de barcelona",
            "hospital clinic de barcelona",
            "barcelona",
            "clínic de barcelona",
        ],
        "Australia": [
            "university of queensland",
            "queensland",
            "frazer institute",
            "diamantina institute",
            "sydney melanoma",
            "royal prince alfred",
        ],
        "Austria": [
            "medical university of vienna",
            "university of vienna",
            "vidir group",
            "vienna",
        ],
        "Greece": [
            "university of athens",
            "athens",
            "andreas syggros",
            "konstantinos liopyris",
        ],
        "Switzerland": [
            "university hospital of basel",
            "university hospital basel",
            "hospital basel",
            "basel",
        ],
        "United Kingdom": [
            "imperial college london",
            "imperial college",
        ],
        "Brazil": [
            "federal university of espírito santo",
            "ufes",
        ],
        "Argentina": [
            "hospital italiano de buenos aires",
            "buenos aires",
        ],
        "France": [
            "julien anriot",
        ],
    }

    countries: List[str] = []
    for country, keywords in patterns.items():
        if any(keyword in attribution for keyword in keywords):
            if country not in countries:
                countries.append(country)

    return countries if countries else None


def parse_country_field(country: object) -> Optional[List[str]]:
    """Parse country field from dataset metadata (e.g., PASSION dataset).

    Args:
        country: Country value from metadata (can be string or NaN)

    Returns:
        List containing the country name, or None if invalid/missing
    """
    if pd.isna(country):
        return None

    country_str = str(country).strip()
    if not country_str or country_str.lower() in ["unknown", "none", "nan", ""]:
        return None

    return [country_str]


def parse_skincap_source(source: object) -> Optional[List[str]]:
    """Parse SkinCAP source field to determine origin based on source dataset.

    SkinCAP is a composite dataset from fitzpatrick17k and DDI.

    Args:
        source: Source dataset name from metadata

    Returns:
        List containing origin countries based on source dataset, or None
    """
    if pd.isna(source):
        return None

    source_str = str(source).strip().lower()

    # Map source datasets to their known origins
    source_origin_map = {
        "ddi": ["United States"],
        "fitzpatrick17k": None,  # Web-sourced, unknown origin
    }

    return source_origin_map.get(source_str, None)


def create_isic_attribution_lookup(
    isic_meta_data: pd.DataFrame,
) -> Dict[str, Optional[List[str]]]:
    """Create a lookup dictionary mapping ISIC image_id to parsed origin.

    Args:
        isic_meta_data: DataFrame with ISIC metadata including image_id and attribution columns

    Returns:
        Dictionary mapping image_id to list of origin countries (or None)
    """
    if "attribution" not in isic_meta_data.columns:
        logger.warning(
            "ISIC metadata missing 'attribution' column; cannot create origin lookup"
        )
        return {}

    # Determine the image ID column name (could be img_path, image_id, etc.)
    img_id_col = None
    for col in ["img_path", "image_id", "isic_id"]:
        if col in isic_meta_data.columns:
            img_id_col = col
            break

    if img_id_col is None:
        logger.warning(
            "ISIC metadata missing image ID column; cannot create origin lookup"
        )
        return {}

    # Extract just the ISIC ID part (e.g., "ISIC_0027419" from full path)
    def extract_isic_id(path: str) -> str:
        """Extract ISIC_XXXXXXX from path or filename."""
        if pd.isna(path):
            return None
        path_str = str(path)
        # Look for ISIC_XXXXXXX pattern
        import re

        match = re.search(r"(ISIC_\d+)", path_str)
        return match.group(1) if match else None

    # Create lookup
    lookup = {}
    for idx, row in isic_meta_data.iterrows():
        isic_id = extract_isic_id(row[img_id_col])
        if isic_id:
            origin = parse_isic_attribution(row.get("attribution"))
            lookup[isic_id] = origin

    logger.info(f"Created ISIC attribution lookup with {len(lookup)} entries")
    return lookup


origin_mapper = (
    {
        DatasetName.ISIC_2024: "parse_attribution",  # Will parse from attribution
        DatasetName.ISIC: "parse_attribution",  # Will parse from attribution
        DatasetName.FITZPATRICK17K: None,  # Web-sourced from online atlases
        DatasetName.HAM10000: "parse_from_isic",  # Look up origin in ISIC archive by image_id
        DatasetName.SD_128: None,  # DermQuest atlas, mixed/unclear sources
        DatasetName.PASSION: "parse_country",  # Will parse from country field
        DatasetName.DERM7PT: ["Italy"],
        DatasetName.PH2: ["Portugal"],
        DatasetName.MED_NODE: ["Netherlands"],
        DatasetName.SKINCAP: "parse_source",  # Composite dataset: parse from source field
        DatasetName.SCIN: ["United States"],
        DatasetName.ALTMEYERS: ["Germany"],
        DatasetName.DERM1M: None,  # Web-sourced educational resources
        DatasetName.DERMACOMPASS: ["Switzerland"],
        DatasetName.PAD_UFES_20: ["Brazil"],
        DatasetName.DDI: ["United States"],
        DatasetName.DAFFODIL: None,  # Web-sourced, unknown origin
        DatasetName.LESION130K: ["South Korea"],  # Created by Han Seung Seog, Korea
        DatasetName.MM_SKINQA: None,  # Web-sourced medical textbook images
        DatasetName.HIBA: ["Argentina"],  # Hospital Italiano de Buenos Aires
        DatasetName.MSKCC: [
            "United States"
        ],  # Memorial Sloan Kettering Cancer Center, New York
        DatasetName.DERMNET_CSV: ["New Zealand"],  # DermNet NZ
        DatasetName.DERMACON_IN: ["India"],  # DermaCon-IN dataset from India
        # MILK10k: 5 ISIC-archive centers (Vienna, Ankara, Queensland, MSKCC, Skopje)
        DatasetName.MILK10K: [
            "Austria",
            "Turkey",
            "Australia",
            "United States",
            "North Macedonia",
        ],
        # BALD: 7 academic centers across Austria, Greece, France, Italy, Japan
        DatasetName.BALD: ["Austria", "Greece", "France", "Italy", "Japan"],
        DatasetName.DERM12345: ["Turkey"],  # Turkish institutions (2008-2021)
        DatasetName.MIDAS: ["United States"],  # Stanford / MRA-MIDAS
    }
    if DatasetName is not None
    else {}
)

origin_mapper_name = {
    "DermNet": ["New Zealand"],
    "CAN2000": None,  # Web-sourced from ~80 countries
    "CAN5600": None,  # Web-sourced from ~80 countries
    "PubmedNoisy": None,  # Scientific literature, global
    "Altmeyers": ["Germany"],
}


DEFAULT_L_DATASETS = (
    [
        (DatasetName.ISIC_2024, Path("/data/"), "ISIC-challenge-2024"),
        (DatasetName.ISIC, Path("/data/"), "ISIC"),
        (DatasetName.FITZPATRICK17K, Path("/data/"), "Fitzpatrick17k"),
        (DatasetName.HAM10000, Path("/data/"), "HAM10000"),
        (DatasetName.SD_128, Path("/data/"), "SD128"),
        (
            DatasetName.PASSION,
            Path("/data/PASSION/PASSION_collection_2020_2023"),
            "PASSION",
        ),
        (DatasetName.DERMACOMPASS, Path("/data/"), "DermaCompass"),
        (DatasetName.DERMACON_IN, Path("/data/"), "DermaCon-IN"),
        (DatasetName.DERM7PT, Path("/data/"), "Derm7pt"),
        (DatasetName.PH2, Path("/data/"), "PH2"),
        (DatasetName.MED_NODE, Path("/data/"), "MED-NODE"),
        (DatasetName.SKINCAP, Path("/data/"), "SkinCAP"),
        (DatasetName.SCIN, Path("/data/"), "SCIN"),
        (DatasetName.GENERIC, Path("/data/CAN2000"), "CAN2000"),
        (DatasetName.GENERIC, Path("/data/CAN5600"), "CAN5600"),
        (DatasetName.GENERIC, Path("/data/DermNet"), "DermNet"),
        (DatasetName.DERMNET_CSV, Path("/data/"), "DermNet"),
        (DatasetName.PUBMED_NOISY, Path("/data/pubmed_noisy"), "PubMed"),
        (DatasetName.ALTMEYERS, Path("/data/Altmeyers"), "Altmeyers"),
        (DatasetName.DERM1M, Path("/data/Derm1M"), "Derm1M"),
        (DatasetName.PAD_UFES_20, Path("/data/"), "PAD-UFES-20"),
        (DatasetName.DDI, Path("/data/"), "DDI"),
        (DatasetName.DAFFODIL, Path("/data/Daffodil"), "Daffodil"),
        (DatasetName.HIBA, Path("/data/"), "HIBA"),
        (DatasetName.MSKCC, Path("/data/"), "MSKCC"),
        (DatasetName.MILK10K, Path("/data/"), "MILK10k"),
        (DatasetName.BALD, Path("/data/"), "BALD"),
        (DatasetName.DERM12345, Path("/data/"), "DERM12345"),
        (DatasetName.MIDAS, Path("/data/"), "MIDAS"),
        (DatasetName.MM_SKINQA, Path("/data/MM-SkinQA"), "MM-SkinQA"),
    ]
    if DatasetName is not None
    else []
)


# --------------------------
# Body location harmonization (verbatim from the notebook)
# --------------------------

UNKNOWN_TOKENS = {
    "unknown",
    "none",
    "unspecified",
    "not specified",
    "na",
    "n/a",
    "",
    "nan",
}

CORRECTIONS = [
    (r"\bower extremity\b", "lower extremity"),  # typo
    (r"\beylid\b", "eyelid"),  # typo
    (r"\bdorsum feet\b", "dorsum foot"),  # plural variant
    (r"\blowerlegs\b", "lower legs"),  # concatenated variant
]


def _normalize(s: object) -> str:
    s = str(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = (
        s.replace("dorsumhand", "dorsum hand")
        .replace("dorsumfoot", "dorsum foot")
        .replace("glutealcleft", "gluteal cleft")
        .replace("antecubitalfossa", "antecubital fossa")
    )
    s = re.sub(r"[^a-z0-9\s\/,-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, rep in CORRECTIONS:
        s = re.sub(pat, rep, s)
    return s


def extract_laterality(raw: object) -> str:
    s = _normalize(raw)
    if re.search(r"\bbilateral|both sides\b", s):
        return "bilateral"
    if re.search(r"\bleft\b", s):
        return "left"
    if re.search(r"\bright\b", s):
        return "right"
    return np.nan


CATS: Dict[str, List[str]] = {
    "head_neck": [
        "head",
        "face",
        "scalp",
        "forehead",
        "temple",
        "eye",
        "eyelid",
        "ear",
        "nose",
        "cheek",
        "lip",
        "mouth",
        "chin",
        "jaw",
        "mandible",
        "maxilla",
        "neck",
        "hair",
        "hairbearingarea",
        "perioral",
        # merged from mucosa (oral/mucosal sites are head anatomy)
        "oral",
        "buccal",
        "tongue",
        "gingiva",
        "palate",
        "conjunctiva",
        "nasal",
        "mucosa",
        "lip mucosa",
        "mucosalmembranes",
        "lips",
    ],
    "trunk_front": [
        "torso front",
        "anterior trunk",
        "anterior torso",
        "chest",
        "breast",
        "sternum",
        "clavicle",
        "rib",
        "abdomen",
        "abdominal",
        "suprapubic",
        # merged from shoulder_axilla (axilla is a trunk fold)
        "axilla",
        "armpit",
        "axillae",
        # merged from trunk_unspecified (default to front when unspecified)
        "torso",
        "trunk",
        "lateral torso",
        "lateral trunk",
    ],
    "trunk_back": [
        "torso back",
        "posterior trunk",
        "posterior torso",
        "back",
        "scapular",
        "paraspinal",
        "lumbar",
    ],
    "upper_limb": [
        "upper extremity",
        "upper extremities",
        "upper limb",
        "upper limbs",
        "upper arm",
        "brachium",
        "elbow",
        "antecubital",
        "forearm",
        "wrist",
        "arm",
        "arms",
        "wrists",
        # merged from shoulder_axilla (shoulder is proximal upper limb)
        "shoulder",
        "deltoid",
        "shoulders",
    ],
    "lower_limb": [
        "lower extremity",
        "lower extremities",
        "lower limb",
        "lower limbs",
        "leg",
        "thigh",
        "knee",
        "popliteal",
        "calf",
        "shin",
        "lower leg",
        "ankle",
        "legs",
        "lower legs",
        "ankles",
        "knees",
        "thighs",
    ],
    "acral": [
        # explicit acral markers (ISIC uses "acral NOS" / "acral palms or soles")
        "acral",
        # nail structures
        "nail",
        "periungual",
        "subungual",
        "matrix",
        "nail bed",
        # hands / fingers (from upper_limb)
        "hand",
        "palm",
        "dorsum hand",
        "finger",
        "thumb",
        "index",
        "middle",
        "ring",
        "little",
        "hands",
        "fingers",
        "palms",
        "palmar",
        "palmarplantar",
        # feet / toes (from lower_limb)
        "heel",
        "foot",
        "feet",
        "dorsum foot",
        "sole",
        "plantar",
        "toe",
        "hallux",
        "toes",
        "soles",
    ],
    "gluteal_buttocks": [
        "gluteal",
        "butt",
        "buttock",
        "buttocks",
        "gluteal cleft",
        "natal cleft",
    ],
    "groin_genital_perineum": [
        "groin",
        "inguinal",
        "genital",
        "penis",
        "scrotum",
        "testis",
        "vulva",
        "labia",
        "mons",
        "perineum",
        "pubic",
        "genitals",
    ],
}

PRIORITY: List[str] = [
    "acral",
    "groin_genital_perineum",
    "gluteal_buttocks",
    "trunk_back",
    "trunk_front",
    "head_neck",
    "upper_limb",
    "lower_limb",
]


def _make_pattern(tokens: List[str]) -> re.Pattern:
    escaped = [re.escape(t) for t in sorted(tokens, key=len, reverse=True)]
    return re.compile(rf"(?<![a-z0-9])(?:{'|'.join(escaped)})(?![a-z0-9])")


CAT_PATTERNS: Dict[str, re.Pattern] = {k: _make_pattern(v) for k, v in CATS.items()}


def _fallback_region(s: str) -> str | None:
    # Acral (nail / hands / feet) — checked first (most specific)
    if re.search(r"\b(nail|fingernails|periungual|subungual|matrix|nail bed)\b", s):
        return "acral"
    if re.search(r"\b(hand|hands|finger(s)?|palm(s)?|palmar|palmarplantar)\b", s):
        return "acral"
    if re.search(r"\b(foot|feet|toe(s)?|sole(s)?|plantar|heel|hallux)\b", s):
        return "acral"
    if re.search(r"\barm(s)?\b", s):
        return "upper_limb"
    if re.search(r"\b(wrist(s)?)\b", s):
        return "upper_limb"
    if re.search(r"\bleg(s)?\b", s):
        return "lower_limb"
    if re.search(
        r"\b(lower ?leg(s)?|thigh(s)?|knee(s)?|ankle(s)?)\b",
        s,
    ):
        return "lower_limb"
    if re.search(r"\b(shoulder(s)?|deltoid)\b", s):
        return "upper_limb"
    if re.search(r"\b(back|lumbar|scapular|paraspinal)\b", s):
        return "trunk_back"
    if re.search(
        r"\b(chest|abdomen|abdominal|sternum|clavicle|rib|anterior trunk|anterior torso|torso front|axillae?|armpit)\b",
        s,
    ):
        return "trunk_front"
    if re.search(r"\b(torso|trunk|lateral torso|lateral trunk)\b", s):
        return "trunk_front"
    if re.search(
        r"\b(face|scalp|neck|cheek|lip|mouth|perioral|nose|ear|eye|eyelid|hair|oral|buccal|tongue|mucosa)\b",
        s,
    ):
        return "head_neck"
    if re.search(r"\b(genital(s)?|groin|inguin|perineum|pubic)\b", s):
        return "groin_genital_perineum"
    return None


def harmonize_body_location(raw: object) -> str:
    s = _normalize(raw)
    if s in UNKNOWN_TOKENS:
        return "unknown"

    s_core = re.sub(r"\b(left|right|bilateral|both sides)\b", " ", s)
    s_core = re.sub(r"\s+", " ", s_core).strip()

    matches = [cat for cat, pat in CAT_PATTERNS.items() if pat.search(s_core)]

    if not matches and re.search(r"\bintertriginous\b", s_core):
        return "other"

    if matches:
        for cat in PRIORITY:
            if cat in matches:
                return cat

    fb = _fallback_region(s_core)
    if fb:
        return fb

    return "other"


def harmonize_dataframe(
    df: pd.DataFrame, col: str = "body_location"
) -> pd.DataFrame:
    out = df.copy()
    out["laterality"] = out[col].apply(extract_laterality)
    out["body_region"] = out[col].apply(harmonize_body_location)
    return out


def preprocess_body_location(series: pd.Series) -> pd.Series:
    """Mimic notebook cleanup before harmonization (split & sort multi-sites)."""
    out = series.apply(
        lambda x: (
            x.lower().replace("&", ",").replace("-", ",").split(",")
            if isinstance(x, str)
            else x
        )
    )
    out = out.apply(lambda x: [y.strip() for y in x] if isinstance(x, list) else x)
    out = out.apply(lambda x: sorted(x) if isinstance(x, list) else x)
    out = out.apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return out.where(lambda s: s.notna(), np.nan)


# --------------------------
# Gender harmonization (matches notebook cell)
# --------------------------

GENDER_MAPPING = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
    "other_or_unspecified": np.nan,
}


def harmonize_gender(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.lower()
        .map(GENDER_MAPPING)
        .where(lambda s: s.notna(), np.nan)
    )
    return mapped


# --------------------------
# Condition cleaning and ISIC merging
# --------------------------


def merge_small_isic_data(dataset_desc: str) -> str:
    if "ISIC" not in dataset_desc:
        return dataset_desc
    if "challenge" not in dataset_desc:
        return "ISIC-others"
    return dataset_desc


def clean_condition_series(series: pd.Series) -> pd.Series:
    out = (
        series.str.replace("_", " ")
        .str.replace("-", " ")
        .str.lower()
        .apply(lambda x: re.sub(r"[\(\[].*?[\)\]]", "", x) if isinstance(x, str) else x)
        .str.strip()
    )
    return out.where(lambda x: x.notna(), np.nan)


def harmonize_fitzpatrick(series: pd.Series) -> pd.Series:
    cleaned = series.apply(
        lambda x: (
            x.replace("fitzpatrick skin type", "").strip() if isinstance(x, str) else x
        )
    )
    cleaned = cleaned.apply(
        lambda x: x.replace("FST", "").strip() if isinstance(x, str) else x
    )
    mapping = {
        -1: np.nan,
        56: 6,
        34: 4,
        12: 2,
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "NONE_IDENTIFIED": np.nan,
    }
    cleaned = cleaned.apply(lambda x: mapping.get(x, x))
    cleaned = cleaned.where(lambda x: x.notna(), np.nan)
    def _to_int_or_nan(x):
        if not isinstance(x, str):
            return x
        try:
            return int(x)
        except (ValueError, TypeError):
            return np.nan

    cleaned = cleaned.apply(_to_int_or_nan)
    return cleaned.where(lambda x: x.notna(), np.nan)


def harmonize_age(series: pd.Series) -> pd.Series:
    mapping = {
        "AGE_UNKNOWN": np.nan,
        None: np.nan,
        "AGE_18_TO_29": 25,
        "AGE_40_TO_49": 45,
        "AGE_50_TO_59": 55,
        "AGE_70_TO_79": 75,
        "AGE_60_TO_69": 65,
        "AGE_30_TO_39": 35,
        3060: np.nan,
    }
    cleaned = series.apply(lambda x: mapping.get(x, x))
    return cleaned.where(lambda x: x.notna(), np.nan)


def merge_concise_captions(
    df: pd.DataFrame, concise_captions_path: Path
) -> pd.DataFrame:
    if concise_captions_path.exists():
        logger.info("Loading existing concise captions from %s", concise_captions_path)
        df_concise = pd.read_csv(concise_captions_path)
        concise_mapping = df_concise.set_index("img_path")[
            "concise_description"
        ].to_dict()
        mask = df["description_short"] == df["description"]
        df.loc[mask, "description_short"] = (
            df.loc[mask, "img_path"]
            .map(concise_mapping)
            .fillna(df.loc[mask, "description_short"])
        )
        logger.info("Merged concise captions for %d rows", mask.sum())
    else:
        logger.info(
            "No concise captions found at %s; skipping merge", concise_captions_path
        )
    return df


# --------------------------
# Value-count diagnostics to mirror the notebook sanity checks
# --------------------------


def analyze_origin_distribution(df: pd.DataFrame) -> None:
    """Analyze and log statistics about origin distribution, including multi-origin samples."""
    if "origin" not in df.columns:
        return

    origin_series = df["origin"]

    # Count samples with different numbers of origins
    def count_origins(x):
        """Count number of origins (handles both strings and lists)."""
        if x is None:
            return 0
        try:
            # Check if it's NaN (scalar)
            if isinstance(x, (float, int)) and pd.isna(x):
                return 0
        except (TypeError, ValueError):
            pass
        if isinstance(x, list):
            return len(x)
        if isinstance(x, str):
            return 1
        return 0

    origin_counts = origin_series.apply(count_origins)

    # Statistics
    total_samples = len(df)
    samples_with_origin = (origin_counts > 0).sum()
    samples_no_origin = (origin_counts == 0).sum()
    samples_single_origin = (origin_counts == 1).sum()
    samples_multi_origin = (origin_counts > 1).sum()

    logger.info("=" * 60)
    logger.info("ORIGIN DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total_samples:,}")
    logger.info(
        f"Samples with origin: {samples_with_origin:,} ({samples_with_origin/total_samples*100:.1f}%)"
    )
    logger.info(
        f"Samples without origin: {samples_no_origin:,} ({samples_no_origin/total_samples*100:.1f}%)"
    )
    logger.info(
        f"  - Single origin: {samples_single_origin:,} ({samples_single_origin/total_samples*100:.1f}%)"
    )
    logger.info(
        f"  - Multiple origins: {samples_multi_origin:,} ({samples_multi_origin/total_samples*100:.1f}%)"
    )

    if samples_multi_origin > 0:
        logger.info("")
        logger.info("Multi-origin breakdown:")
        multi_origin_dist = origin_counts[origin_counts > 1].value_counts().sort_index()
        for num_origins, count in multi_origin_dist.items():
            logger.info(f"  - {num_origins} origins: {count:,} samples")

        # Show examples of multi-origin samples by dataset
        logger.info("")
        logger.info("Multi-origin samples by dataset:")
        multi_origin_df = df[origin_counts > 1]
        if "dataset_desc" in multi_origin_df.columns:
            dataset_multi_counts = multi_origin_df["dataset_desc"].value_counts()
            for dataset, count in dataset_multi_counts.items():
                logger.info(f"  - {dataset}: {count:,} samples")

    # Show all unique origins
    logger.info("")
    logger.info("All unique origins found:")
    all_origins = set()
    for origins in origin_series:
        if origins is None:
            continue
        # Skip NaN values
        try:
            if isinstance(origins, (float, int)) and pd.isna(origins):
                continue
        except (TypeError, ValueError):
            pass

        if isinstance(origins, list):
            all_origins.update(origins)
        elif isinstance(origins, str):
            all_origins.add(origins)

    for origin in sorted(all_origins):
        # Count how many samples have this origin
        def has_origin(x):
            if x is None:
                return False
            try:
                if isinstance(x, (float, int)) and pd.isna(x):
                    return False
            except (TypeError, ValueError):
                pass
            if isinstance(x, list):
                return origin in x
            elif isinstance(x, str):
                return x == origin
            return False

        count = sum(1 for x in origin_series if has_origin(x))
        logger.info(f"  - {origin}: {count:,} samples")

    logger.info("=" * 60)


def log_value_counts_summary(
    df: pd.DataFrame, condition_col: Optional[str] = None, top_n: int = 30
) -> Dict[str, pd.Series]:
    summaries: Dict[str, pd.Series] = {}

    def _make_hashable(x: object) -> object:
        if isinstance(x, list):
            return tuple(x)
        return x

    def _log_counts(label: str, series: pd.Series) -> None:
        counts = series.value_counts(dropna=False)
        if counts.empty:
            return
        summaries[label] = counts
        unique_count = (
            series.dropna().apply(_make_hashable).nunique()
            if not series.dropna().empty
            else 0
        )
        logger.info(
            "%s (top %d, total=%d, non-null=%d, unique=%d):\n%s",
            label,
            top_n,
            len(series),
            series.notna().sum(),
            unique_count,
            counts.head(top_n).to_string(),
        )

    if "dataset_desc" in df.columns:
        _log_counts("dataset_desc", df["dataset_desc"])
        isic_mask = df["dataset_desc"].astype(str).str.contains("ISIC", na=False)
        if isic_mask.any():
            _log_counts("dataset_desc (ISIC only)", df.loc[isic_mask, "dataset_desc"])

    if "modality" in df.columns:
        _log_counts("modality", df["modality"])
        missing_modality = df.loc[df["modality"].isna(), "dataset_desc"]
        if not missing_modality.empty:
            _log_counts("dataset_desc (missing modality)", missing_modality)

    if "release_year" in df.columns:
        missing_year = df.loc[df["release_year"].isna(), "dataset_desc"]
        if not missing_year.empty:
            _log_counts("dataset_desc (missing release_year)", missing_year)

    for col in ["laterality", "body_region", "gender", "fitzpatrick", "age", "origin"]:
        if col in df.columns:
            _log_counts(col, df[col])

    if condition_col and condition_col in df.columns:
        _log_counts(condition_col, df[condition_col])
        dermatitis = df.loc[
            df[condition_col].astype(str).str.contains("dermatitis", na=False),
            "dataset_desc",
        ]
        if not dermatitis.empty:
            _log_counts("dataset_desc (condition contains 'dermatitis')", dermatitis)

    return summaries


# --------------------------
# Severity features
# --------------------------

# ICD category code → severity tier mapping.
# Each entry is (prefix, tier_label).  Checked in order; first match wins.
_ICD_SEVERITY_RULES: List[Tuple[str, str]] = [
    # Tier 4 – malignant neoplasms
    ("C", "malignant"),
    # Tier 3 – in-situ neoplasms & actinic keratosis
    ("D00", "pre_malignant"),
    ("D01", "pre_malignant"),
    ("D02", "pre_malignant"),
    ("D03", "pre_malignant"),
    ("D04", "pre_malignant"),
    ("D05", "pre_malignant"),
    ("D06", "pre_malignant"),
    ("D07", "pre_malignant"),
    ("D08", "pre_malignant"),
    ("D09", "pre_malignant"),
    ("L57", "pre_malignant"),
    # Uncertain / unspecified-behaviour neoplasms (D37-D49): genuinely neither
    # benign nor malignant (e.g. keratoacanthoma D48.5, mastocytosis D47,
    # freckle D49.81). Previously lumped into "inflammatory" which was wrong.
    ("D37", "uncertain"),
    ("D38", "uncertain"),
    ("D39", "uncertain"),
    ("D4", "uncertain"),
    # Tier 2 – inflammatory / infections (incl. ICD chapters A & B: infectious
    # diseases were previously unhandled -> silently mislabelled "healthy").
    ("A", "inflammatory"),  # A00-A99 infectious (syphilis A53, lyme A69, cutaneous TB A18.4)
    ("B", "inflammatory"),  # B00-B99 infectious (herpes B00/B02, warts B07, molluscum B08.1, tinea B35)
    ("L0", "inflammatory"),  # L00-L08 infections
    ("L1", "inflammatory"),  # L10-L14 bullous
    ("L2", "inflammatory"),  # L20-L30 dermatitis/eczema
    ("L3", "inflammatory"),  # L30-L39 (cont.)
    ("L4", "inflammatory"),  # L40-L45 papulosquamous
    ("L5", "inflammatory"),  # L50-L54 urticaria/erythema (excl. L57 handled above)
    # Tier 1 – benign
    ("D1", "benign"),  # D10-D36 benign neoplasms
    ("D2", "benign"),
    ("D3", "benign"),  # D30-D36 benign neoplasms (e.g. D22 = melanocytic nevi)
    ("L6", "benign"),  # L60-L75 skin appendage disorders
    ("L7", "benign"),
    ("L8", "benign"),  # L80-L99 other skin disorders
    ("L9", "benign"),
    # Dermatologic vascular lesions in the circulatory chapter are benign:
    # spider angioma / telangiectases I78.x, etc. (previously fell through to
    # "healthy"). NB: stasis dermatitis I87.2 is really inflammatory, but is a
    # rare edge case among these.
    ("I", "benign"),
]


def compute_severity_icd(icd_category: str) -> Optional[str]:
    """Map an ICD category code (e.g. 'C43', 'L20', 'D22') to a severity tier.

    Returns one of: 'healthy', 'benign', 'inflammatory', 'pre_malignant',
    'malignant', or ``None`` when the code is empty / unmapped.
    """
    code = str(icd_category).strip()
    if not code or code.lower() in ("", "unmapped", "nan", "none"):
        return None
    for prefix, label in _ICD_SEVERITY_RULES:
        if code.startswith(prefix):
            return label
    # Codes that don't match any pathological prefix → healthy
    return "healthy"


def compute_severity_symptom(meta_data: pd.DataFrame) -> pd.Series:
    """Compute a symptom-burden severity score from PAD-UFES-20 clinical attributes.

    Expected columns in *meta_data*: ``diameter_1``, ``diameter_2``,
    ``bleed``, ``hurt``, ``itch``, ``grew``, ``changed``, ``elevation``.

    Returns a :class:`pd.Series` with ordinal labels:
    ``'none'``, ``'low'``, ``'moderate'``, ``'high'``.
    """
    required = [
        "diameter_1",
        "diameter_2",
        "bleed",
        "hurt",
        "itch",
        "grew",
        "changed",
        "elevation",
    ]
    missing = [c for c in required if c not in meta_data.columns]
    if missing:
        logger.warning(
            "Cannot compute severity_symptom – missing columns: %s", ", ".join(missing)
        )
        return pd.Series(None, index=meta_data.index, dtype=object)

    d1 = pd.to_numeric(meta_data["diameter_1"], errors="coerce")
    d2 = pd.to_numeric(meta_data["diameter_2"], errors="coerce")
    mean_diam = (d1 + d2) / 2.0
    # z-score normalise diameters within the PAD dataset
    diam_mean = mean_diam.mean()
    diam_std = mean_diam.std()
    if diam_std == 0 or pd.isna(diam_std):
        z_diam = pd.Series(0.0, index=meta_data.index)
    else:
        z_diam = (mean_diam - diam_mean) / diam_std

    binary_cols = ["bleed", "hurt", "itch", "grew", "changed", "elevation"]
    binary_sum = pd.Series(0.0, index=meta_data.index)
    for col in binary_cols:
        binary_sum += (
            pd.to_numeric(meta_data[col], errors="coerce").fillna(0).clip(0, 1)
        )

    raw_score = z_diam + binary_sum

    labels = ["none", "low", "moderate", "high"]
    try:
        binned = pd.qcut(raw_score, q=4, labels=labels, duplicates="drop")
    except ValueError:
        # Fallback if too few unique values for 4 bins
        binned = pd.cut(raw_score, bins=4, labels=labels, duplicates="drop")
    return binned.astype(str).replace("nan", None)


# --------------------------
# Core pipeline
# --------------------------


def _load_dataset_config(path: Path) -> List[Tuple[object, Path, str]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "datasets" in data:
            data = data["datasets"]
    else:
        df = pd.read_csv(path)
        data = df.to_dict(orient="records")

    resolved: List[Tuple[object, Path, str]] = []
    for row in data:
        name_val = row["dataset_name"]
        if DatasetName is not None and not isinstance(name_val, DatasetName):
            name_val = (
                DatasetName[name_val]
                if name_val in DatasetName.__members__
                else DatasetName(name_val)
            )
        resolved.append((name_val, Path(row["dataset_path"]), str(row["dataset_desc"])))
    return resolved


def build_dataframe_from_datasets(
    l_datasets: Sequence[Tuple[object, Path, str]],
) -> pd.DataFrame:
    if get_dataset is None or DatasetName is None:
        raise ImportError(
            "Dataset loader is unavailable; ensure core.src.datasets.helper is importable."
        )

    # Create ISIC attribution lookup for datasets that need it (e.g., HAM10000)
    isic_attribution_lookup: Dict[str, Optional[List[str]]] = {}

    l_dataset = []
    for dataset_name, dataset_path, dataset_desc in l_datasets:
        logger.info("Loading dataset %s from %s", dataset_name, dataset_path)
        try:
            dataset = get_dataset(
                dataset_name=dataset_name, dataset_path=dataset_path, return_loader=False
            )
        except (FileNotFoundError, ValueError) as e:
            # Skip-if-absent: a dataset not staged on this machine (e.g. MIDAS,
            # gated behind the Redivis DUA) must not abort the whole atlas rebuild.
            # The loaders signal a missing dataset two ways: FileNotFoundError (a
            # missing metadata CSV) or ValueError("Image path must exist", raised by
            # GenericImageDataset when the image directory is absent). Skip those
            # and continue; the dataset is picked up automatically once its data is
            # staged. A ValueError that is NOT a missing-path signal is a real
            # loader bug -> re-raise it.
            msg = str(e).lower()
            missing = isinstance(e, FileNotFoundError) or any(
                k in msg for k in ("must exist", "no such file", "does not exist", "not found")
            )
            if not missing:
                raise
            logger.warning(
                "Dataset %s not staged at %s (%s) -- SKIPPING. "
                "Stage it to include it in the rebuild.",
                dataset_name, dataset_path, e,
            )
            continue
        if dataset_name == DatasetName.PAD_UFES_20:
            image_paths = dataset.meta_data[dataset.IMG_COL].apply(
                lambda x: os.path.join(dataset.root_dir, x)
            )
        elif dataset_name == DatasetName.ISIC:
            dataset.meta_data = dataset.meta_data[
                dataset.meta_data.dataset_desc != "ISIC-challenge-2024"
            ]
            dataset.meta_data = dataset.meta_data[
                dataset.meta_data.dataset_desc != "ISIC-ham10000"
            ]
            dataset.meta_data.reset_index(drop=True, inplace=True)
            image_paths = dataset.meta_data["img_path"].values.tolist()

            # Create attribution lookup for other datasets to use
            if not isic_attribution_lookup:
                isic_attribution_lookup = create_isic_attribution_lookup(
                    dataset.meta_data
                )
        else:
            image_paths = dataset.meta_data[dataset.IMG_COL].values.tolist()

        year = None
        if dataset_name in release_mapper.keys():
            year = release_mapper[dataset_name]
        if dataset_desc in release_mapper_name.keys():
            year = release_mapper_name[dataset_desc]

        # Modality resolution order:
        #   1. per-image `modality` column on the dataset (multimodal datasets
        #      such as MILK10k / MIDAS / MM-SkinQA set this from image_type);
        #   2. dataset-level modality_mapper / modality_mapper_name fallback;
        #   3. ISIC datasets default to dermoscopy.
        modality = None
        if dataset_name in modality_mapper.keys():
            modality = modality_mapper[dataset_name]
        if dataset_desc in modality_mapper_name.keys():
            modality = modality_mapper_name[dataset_desc]
        if modality is None and "ISIC" in dataset_desc:
            modality = "dermoscopy"

        if "modality" in dataset.meta_data.columns:
            # Honor a per-image modality column, filling gaps with the
            # dataset-level fallback resolved above.
            modality_values = (
                dataset.meta_data["modality"]
                .where(dataset.meta_data["modality"].notna(), modality)
                .tolist()
            )
        else:
            modality_values = len(dataset) * [modality]

        origin = None
        if dataset_name in origin_mapper.keys():
            origin = origin_mapper[dataset_name]
        if dataset_desc in origin_mapper_name.keys():
            origin = origin_mapper_name[dataset_desc]

        if origin == "parse_attribution" and "attribution" in dataset.meta_data.columns:
            logger.info("Parsing attribution for %s...", dataset_desc)
            origin_list = (
                dataset.meta_data["attribution"].apply(parse_isic_attribution).tolist()
            )
        elif origin == "parse_country" and "country" in dataset.meta_data.columns:
            logger.info("Parsing country field for %s...", dataset_desc)
            origin_list = (
                dataset.meta_data["country"].apply(parse_country_field).tolist()
            )
        elif origin == "parse_source" and "source" in dataset.meta_data.columns:
            logger.info("Parsing source field for %s...", dataset_desc)
            origin_list = (
                dataset.meta_data["source"].apply(parse_skincap_source).tolist()
            )
        elif origin == "parse_from_isic" and "image_id" in dataset.meta_data.columns:
            logger.info("Looking up origin from ISIC archive for %s...", dataset_desc)
            if not isic_attribution_lookup:
                logger.warning(
                    "ISIC dataset not loaded yet; cannot look up attribution for %s. "
                    "Ensure ISIC is loaded before HAM10000 in dataset list.",
                    dataset_desc,
                )
                origin_list = len(dataset) * [["Austria"]]  # Fallback to Austria only
            else:
                # Look up each image_id in ISIC attribution lookup
                origin_list = []
                for image_id in dataset.meta_data["image_id"]:
                    isic_id = str(image_id).strip()
                    origin = isic_attribution_lookup.get(isic_id, ["Austria"])
                    origin_list.append(origin)
                found_count = sum(1 for o in origin_list if o != ["Austria"])
                logger.info(
                    "Looked up %d/%d HAM10000 images in ISIC archive",
                    found_count,
                    len(origin_list),
                )
        else:
            origin_list = len(dataset) * [origin]

        if dataset_name == DatasetName.GENERIC:
            dataset.meta_data = dataset.meta_data.rename(
                columns={"diagnosis": "condition"}
            )

        # Standardize condition column name across all datasets
        # Different datasets use different column names: disease, diagnosis, diagnostic_name, condition
        if "condition" not in dataset.meta_data.columns:
            # Check for alternative column names and rename to "condition"
            if "disease" in dataset.meta_data.columns:
                dataset.meta_data = dataset.meta_data.rename(
                    columns={"disease": "condition"}
                )
                logger.debug(f"Renamed 'disease' to 'condition' for {dataset_desc}")
            elif "diagnostic_name" in dataset.meta_data.columns:
                dataset.meta_data = dataset.meta_data.rename(
                    columns={"diagnostic_name": "condition"}
                )
                logger.debug(
                    f"Renamed 'diagnostic_name' to 'condition' for {dataset_desc}"
                )
            elif "diagnosis" in dataset.meta_data.columns:
                dataset.meta_data = dataset.meta_data.rename(
                    columns={"diagnosis": "condition"}
                )
                logger.debug(f"Renamed 'diagnosis' to 'condition' for {dataset_desc}")

        # Compute severity_symptom for PAD-UFES-20, None for all others
        if dataset_name == DatasetName.PAD_UFES_20:
            severity_symptom_values = compute_severity_symptom(
                dataset.meta_data
            ).tolist()
        else:
            severity_symptom_values = len(dataset) * [None]

        columns = dataset.meta_data.columns
        l_dataset.append(
            list(
                zip(
                    image_paths,
                    dataset.meta_data["description"],
                    (
                        dataset.meta_data["dataset_desc"]
                        if "dataset_desc" in columns
                        else len(dataset) * [dataset_desc]
                    ),
                    (
                        dataset.meta_data["dataset_year"]
                        if "dataset_year" in columns
                        else len(dataset) * [year]
                    ),
                    modality_values,
                    (
                        dataset.meta_data["condition"]
                        if "condition" in columns
                        else len(dataset) * [None]
                    ),
                    (
                        dataset.meta_data["body_location"]
                        if "body_location" in columns
                        else len(dataset) * [None]
                    ),
                    (
                        dataset.meta_data["gender"]
                        if "gender" in columns
                        else len(dataset) * [None]
                    ),
                    (
                        dataset.meta_data["age"]
                        if "age" in columns
                        else len(dataset) * [None]
                    ),
                    (
                        dataset.meta_data["fitzpatrick"]
                        if "fitzpatrick" in columns
                        else len(dataset) * [None]
                    ),
                    (
                        dataset.meta_data["description_short"]
                        if "description_short" in columns
                        else dataset.meta_data["description"]
                    ),
                    origin_list,
                    severity_symptom_values,
                )
            )
        )

    df = pd.DataFrame(
        sum(l_dataset, []),
        columns=[
            "img_path",
            "description",
            "dataset_desc",
            "release_year",
            "modality",
            "condition",
            "body_location",
            "gender",
            "age",
            "fitzpatrick",
            "description_short",
            "origin",
            "severity_symptom",
        ],
    )
    df["description"] = df.apply(
        lambda row: " ".join(row["description"].split()), axis=1
    )
    df["description_short"] = df.apply(
        lambda row: " ".join(row["description_short"].split()), axis=1
    )

    # Simplify single-origin lists to single strings
    def simplify_origin(x):
        """Convert single-element origin lists to strings, keep multi-origin as lists."""
        if x is None:
            return None
        try:
            if isinstance(x, (float, int)) and pd.isna(x):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(x, list):
            if len(x) == 1:
                return x[0]  # Single origin: convert to string
            elif len(x) == 0:
                return None
            else:
                return x  # Multiple origins: keep as list
        return x

    df["origin"] = df["origin"].apply(simplify_origin)
    single_origin_count = df["origin"].apply(lambda x: isinstance(x, str)).sum()
    multi_origin_count = df["origin"].apply(lambda x: isinstance(x, list)).sum()
    logger.info(
        "Simplified origins: %d single-origin (strings), %d multi-origin (lists)",
        single_origin_count,
        multi_origin_count,
    )

    return df


def run_prep(
    input_csv: Optional[Path],
    dataset_config: Optional[Path],
    concise_captions: Path,
    icd_mapping_path: Optional[Path],
    condition_col: str,
    output_path: Path,
    log_counts: bool = True,
    value_counts_top: int = 30,
) -> Path:
    if input_csv is None and dataset_config is None:
        raise ValueError("Provide either --input-csv or --dataset-config.")

    if input_csv is not None:
        logger.info("Loading input CSV %s", input_csv)
        df = pd.read_csv(input_csv)
    else:
        if dataset_config is None:
            l_datasets = DEFAULT_L_DATASETS
        elif dataset_config == Path("default"):
            l_datasets = DEFAULT_L_DATASETS
        else:
            l_datasets = _load_dataset_config(dataset_config)
        # Honour $DATA_ROOT (default /data) so the pipeline runs on any machine,
        # not only one with data mounted at /data. rebuild_atlas.sh and the
        # downloader already use DATA_ROOT; previously prep_data ignored it and
        # silently read /data regardless. Only re-roots the /data-prefixed paths.
        data_root = os.environ.get("DATA_ROOT", "/data").rstrip("/") or "/data"
        if data_root != "/data":
            def _reroot(p: Path) -> Path:
                s = str(p)
                if s == "/data" or s.startswith("/data/"):
                    return Path(data_root + s[len("/data"):])
                return p
            l_datasets = [(n, _reroot(p), d) for (n, p, d) in l_datasets]
            logger.info("Re-rooted dataset paths under DATA_ROOT=%s", data_root)
        df = build_dataframe_from_datasets(l_datasets)

    df["dataset_desc"] = df["dataset_desc"].apply(
        lambda x: merge_small_isic_data(str(x))
    )
    df = merge_concise_captions(df, concise_captions)
    if "body_location" in df.columns:
        df["body_location"] = preprocess_body_location(df["body_location"])
    df = harmonize_dataframe(df)
    df["body_region"] = df["body_region"].apply(
        lambda x: np.nan if x in ["unknown", "other"] else x
    )
    if condition_col in df.columns:
        df[condition_col] = clean_condition_series(df[condition_col])
    if "gender" in df.columns:
        df["gender"] = harmonize_gender(df["gender"])
    if "fitzpatrick" in df.columns:
        df["fitzpatrick"] = harmonize_fitzpatrick(df["fitzpatrick"])
    if "age" in df.columns:
        df["age"] = harmonize_age(df["age"])

    if icd_mapping_path is not None:
        icd_mapping_df = load_icd_mapping(str(icd_mapping_path))
        if icd_mapping_df is not None:
            df = annotate_icd(df, icd_mapping_df, condition_col)
            # Derive severity from ICD category codes
            if "icd_category" in df.columns:
                df["severity_icd"] = df["icd_category"].apply(compute_severity_icd)
        else:
            logger.warning("ICD mapping not loaded; skipping ICD annotation.")

    if log_counts:
        log_value_counts_summary(
            df,
            condition_col=condition_col if condition_col in df.columns else None,
            top_n=value_counts_top,
        )
        # Analyze origin distribution including multi-origin samples
        analyze_origin_distribution(df)

    # Check for and remove duplicate img_path entries
    if "img_path" in df.columns:
        n_before = len(df)
        n_duplicates = df.duplicated(subset=["img_path"]).sum()

        if n_duplicates > 0:
            logger.warning(
                f"Found {n_duplicates} duplicate img_path entries ({n_duplicates/n_before*100:.2f}%)"
            )

            # Show examples of duplicates
            duplicate_paths = df[df.duplicated(subset=["img_path"], keep=False)][
                "img_path"
            ].unique()[:5]
            logger.warning(f"Example duplicate paths: {list(duplicate_paths)}")

            # Check if duplicates have identical metadata (safe to drop) or different metadata (data issue)
            dup_df = df[df.duplicated(subset=["img_path"], keep=False)]
            grouped = dup_df.groupby("img_path")

            # Check a sample of duplicates to see if they're identical
            identical_duplicates = 0
            different_duplicates = 0

            for path, group in list(grouped)[:100]:  # Check first 100 duplicate groups
                if len(group) > 1:
                    # Compare all rows for this path (excluding img_path itself)
                    cols_to_compare = [c for c in group.columns if c != "img_path"]
                    if cols_to_compare:
                        first_row = group.iloc[0][cols_to_compare]
                        all_identical = all(
                            group.iloc[i][cols_to_compare].equals(first_row)
                            for i in range(1, len(group))
                        )
                        if all_identical:
                            identical_duplicates += 1
                        else:
                            different_duplicates += 1

            if different_duplicates > 0:
                logger.warning(
                    f"WARNING: {different_duplicates}/{len(list(grouped)[:100])} sampled duplicate groups "
                    "have DIFFERENT metadata. This may indicate a data quality issue!"
                )
                logger.warning("Keeping FIRST occurrence of each duplicate.")
            else:
                logger.info(
                    "All sampled duplicates have identical metadata - safely dropping duplicates."
                )

            # Drop duplicates, keeping first occurrence
            df = df.drop_duplicates(subset=["img_path"], keep="first")
            n_after = len(df)
            logger.info(
                f"Removed {n_before - n_after} duplicate rows. Final count: {n_after}"
            )
        else:
            logger.info(
                f"No duplicate img_path entries found ({n_before} unique images)"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    logger.info("Wrote harmonized data to %s", output_path)
    logger.info(
        "Dataset summary: shape=%s, columns=%s",
        df.shape,
        list(df.columns),
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prep data pipeline (extracted from notebook)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Existing merged metadata CSV; skips dataset loading.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("default"),
        help="JSON/CSV listing dataset_name,dataset_path,dataset_desc. "
        "Use 'default' (default) to mirror the notebook list.",
    )
    parser.add_argument(
        "--concise-captions",
        type=Path,
        default=Path("assets/data_with_concise_captions.csv"),
        help="Path to concise captions CSV.",
    )
    parser.add_argument(
        "--icd-mapping",
        type=Path,
        default=Path("results/condition_icd_mapping.csv"),
        help="Condition to ICD mapping CSV. ICD columns are skipped if the file is missing.",
    )
    parser.add_argument(
        "--condition-col",
        type=str,
        default="condition",
        help="Condition column to use for ICD merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/data.csv"),
        help="Output file (.csv or .parquet).",
    )
    parser.add_argument(
        "--no-value-counts",
        action="store_true",
        help="Skip logging notebook-style value-count diagnostics.",
    )
    parser.add_argument(
        "--value-counts-top",
        type=int,
        default=30,
        help="How many rows to show per value-count summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_prep(
        input_csv=args.input_csv,
        dataset_config=args.dataset_config,
        concise_captions=args.concise_captions,
        icd_mapping_path=args.icd_mapping,
        condition_col=args.condition_col,
        output_path=args.output,
        log_counts=not args.no_value_counts,
        value_counts_top=args.value_counts_top,
    )


if __name__ == "__main__":
    main()
