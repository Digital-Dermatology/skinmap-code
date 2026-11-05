from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")

# Chapters that contain primary dermatology codes, ordered by preference.
# 12 = Diseases of the skin (L), 2 = Neoplasms (C/D), 17 = Congenital (Q).
_DERM_PREFERRED_CHAPTERS = {"12", "2", "17"}

# Qualifier pairs that indicate semantic conflicts between query and match.
_CONFLICTING_QUALIFIERS = [
    ("hypopigmentation", "hyperpigmentation"),
    ("hypopigmented", "hyperpigmented"),
    ("hypo", "hyper"),
    ("bullous", "nonbullous"),
    ("acute", "subacute"),
    ("benign", "malignant"),
    ("localized", "generalized"),
]

# Generic ICD codes that should not be assigned via fuzzy matching.
# These are overly broad "unspecified" codes that fuzzy matching tends to
# pick up as false positives. Codes like L30.9 (dermatitis unspecified) are
# excluded because they match any condition containing "dermatitis" — the
# exact match path handles cases where the condition name is literally
# "dermatitis unspecified".
_GENERIC_ICD_CODES = {
    "L98.9",  # Other specified disorders of skin, unspecified
    "R69",    # Illness, unspecified
    "C95.90", # Leukemia, unspecified (not derm)
}


def _has_qualifier_conflict(query: str, match_desc: str) -> bool:
    """Check if query and match description contain conflicting qualifiers."""
    q = query.lower()
    m = match_desc.lower()
    for a, b in _CONFLICTING_QUALIFIERS:
        if (a in q and b in m) or (b in q and a in m):
            return True
    return False


def _semantic_overlap(query_norm: str, match_text: str) -> bool:
    """Check if query and match share meaningful lexical overlap.

    Checks both exact word matches and substring containment to handle
    medical term morphology (e.g., "angioma" in "hemangioma",
    "lipoma" in "lipomatous", "neurofibroma" in "neurofibromatosis").
    """
    q_words = set(query_norm.lower().split())
    m_norm = normalize_condition(match_text)
    m_words = set(m_norm.split())
    # Remove very short words (< 3 chars) as they cause false positives
    q_words = {w for w in q_words if len(w) >= 3}
    m_words = {w for w in m_words if len(w) >= 3}
    # Check exact word overlap
    if q_words & m_words:
        return True
    # Check substring containment for medical morphology, but only when
    # the match word STARTS with the query word (prefix match).
    # This handles: angioma→hemangioma, lipoma→lipomatous,
    # neurofibroma→neurofibromatosis, but avoids: scar→nonscarring,
    # milia→familial.
    for qw in q_words:
        if len(qw) < 5:
            continue  # Avoid short substring false positives
        for mw in m_words:
            # Query is a suffix of match word (e.g., "angioma" in "hemangioma")
            if len(mw) > len(qw) and mw.endswith(qw):
                return True
            # Match word starts with query (e.g., "lipoma" → "lipomatous")
            if len(mw) > len(qw) and mw.startswith(qw):
                return True
            # Query starts with match word (e.g., "neurofibromatosis" query,
            # "neurofibroma" in text)
            if len(qw) > len(mw) and qw.startswith(mw):
                return True
    return False


def normalize_condition(text: str) -> str:
    """Lowercase + strip punctuation for robust exact matches."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.lower()
    text = NORMALIZE_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _split_pipe(value: str | float | None) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


@dataclass
class ICDStringEntry:
    code: str
    description: str
    chapter: str
    chapter_title: str
    chapter_range: str
    section_id: str
    section_desc: str
    match_text: str
    match_source: str
    has_children: bool
    priority: int


class ConditionICDMapper:
    def __init__(
        self,
        icd_table: pd.DataFrame,
        *,
        min_score: float = 90.0,
        max_suggestions: int = 5,
        allowed_chapters: Sequence[str] | None = None,
    ) -> None:
        self.full_table = icd_table.copy()
        self.full_table["description"] = (
            self.full_table["description"].fillna("").astype(str)
        )
        self.full_table["synonyms"] = self.full_table["synonyms"].fillna("")
        self.full_table["notes"] = self.full_table["notes"].fillna("")
        self.full_table["section_desc"] = self.full_table["section_desc"].fillna("")
        self.full_table["chapter_title"] = self.full_table["chapter_title"].fillna("")
        self.full_table["chapter_range"] = self.full_table["chapter_range"].fillna("")
        self.full_table["section_id"] = self.full_table["section_id"].fillna("")
        self.full_table["chapter"] = self.full_table["chapter"].fillna("")
        self.full_table["has_children"] = (
            self.full_table["has_children"].astype(str).fillna("0")
        )

        table = self.full_table
        if allowed_chapters:
            allowed = {str(ch).strip() for ch in allowed_chapters if str(ch).strip()}
            table = table[table["chapter"].astype(str).isin(allowed)]
            if table.empty:
                raise ValueError("Filtering by allowed chapters removed all ICD rows.")
        self.icd_table = table.copy()

        self.code_lookup = {
            str(row["code"]): row for _, row in self.icd_table.iterrows()
        }
        self.full_code_lookup = {
            str(row["code"]): row for _, row in self.full_table.iterrows()
        }
        self.min_score = float(min_score)
        self.max_suggestions = int(max_suggestions)
        self.choice_texts: list[str] = []
        self.choice_meta: list[ICDStringEntry] = []
        self.exact_lookup: dict[str, list[ICDStringEntry]] = {}
        self._build_index()

    def _base_entry(self, row: pd.Series, text: str, source: str) -> ICDStringEntry:
        has_children = str(row["has_children"]).strip() == "1"
        priority = 0
        if has_children:
            priority += 2
        if source != "description":
            priority += 1
        return ICDStringEntry(
            code=str(row["code"]),
            description=row["description"],
            chapter=row["chapter"],
            chapter_title=row["chapter_title"],
            chapter_range=row["chapter_range"],
            section_id=row["section_id"],
            section_desc=row["section_desc"],
            match_text=text,
            match_source=source,
            has_children=has_children,
            priority=priority,
        )

    def _register_entry(self, entry: ICDStringEntry) -> None:
        text = entry.match_text.strip()
        if not text:
            return
        self.choice_texts.append(text)
        self.choice_meta.append(entry)
        norm = normalize_condition(text)
        self.exact_lookup.setdefault(norm, []).append(entry)

    def _build_index(self) -> None:
        for _, row in self.icd_table.iterrows():
            desc_entry = self._base_entry(row, row["description"], "description")
            self._register_entry(desc_entry)
            for synonym in _split_pipe(row.get("synonyms", "")):
                syn_entry = self._base_entry(row, synonym, "synonym")
                self._register_entry(syn_entry)
        # Build a separate index for derm-preferred chapters (fallback search)
        self._derm_preferred_texts: list[str] = []
        self._derm_preferred_entries: list[ICDStringEntry] = []
        for text, entry in zip(self.choice_texts, self.choice_meta):
            if str(entry.chapter) in _DERM_PREFERRED_CHAPTERS:
                self._derm_preferred_texts.append(text)
                self._derm_preferred_entries.append(entry)

    def _pick_exact(self, norm: str) -> ICDStringEntry | None:
        if norm not in self.exact_lookup:
            return None
        candidates = sorted(self.exact_lookup[norm], key=lambda e: e.priority)
        return candidates[0] if candidates else None

    def _load_override_entry(self, condition: str, code: str) -> ICDStringEntry | None:
        if not code:
            return None
        original_code = code
        row = self.full_code_lookup.get(code)
        # If exact code not found, try base code (e.g., W57.XXXA -> W57)
        if row is None and "." in code:
            base_code = code.split(".")[0]
            row = self.full_code_lookup.get(base_code)
        if row is None:
            return None
        entry = self._base_entry(row, str(condition), "override")
        # Preserve the original override code (e.g., W57.XXXA instead of W57)
        entry.code = original_code
        return entry

    def load_overrides(self, path: Path | None) -> dict[str, ICDStringEntry]:
        overrides: dict[str, ICDStringEntry] = {}
        if path is None or not path.exists():
            return overrides
        df = pd.read_csv(path)
        if "condition_raw" not in df.columns or "icd_code" not in df.columns:
            return overrides
        for _, row in df.iterrows():
            condition = str(row["condition_raw"]).strip()
            code = str(row["icd_code"]).strip()
            if not condition or not code:
                continue
            entry = self._load_override_entry(condition, code)
            if entry is None:
                continue
            overrides[normalize_condition(condition)] = entry
        return overrides

    def match_condition(
        self,
        condition: str,
        *,
        overrides: dict[str, ICDStringEntry] | None = None,
    ) -> dict[str, str | float | int]:
        raw = (condition or "").strip()
        norm = normalize_condition(raw)
        result = {
            "condition_raw": raw,
            "condition_normalized": norm,
            "icd_code": "",
            "icd_description": "",
            "chapter": "",
            "chapter_title": "",
            "chapter_range": "",
            "section_id": "",
            "section_desc": "",
            "match_type": "empty" if not raw else "unmapped",
            "match_source": "",
            "match_text": "",
            "match_score": np.nan,
            "suggestions": "",
        }
        if not raw:
            return result

        overrides = overrides or {}
        entry = overrides.get(norm)
        if entry is not None:
            match_type = "override"
            score = 100.0
            suggestions: list[str] = []
            return self._populate_result(result, entry, match_type, score, suggestions)

        entry = self._pick_exact(norm)
        if entry is not None:
            match_type = f"exact_{entry.match_source}"
            score = 100.0
            suggestions = []
            return self._populate_result(result, entry, match_type, score, suggestions)

        # Fetch more candidates than needed so we can filter bad ones
        candidates = process.extract(
            raw,
            self.choice_texts,
            scorer=fuzz.WRatio,
            processor=None,
            limit=max(self.max_suggestions, 30),
        )
        suggestions_meta: list[dict[str, str | float]] = []
        valid_candidates: list[tuple[ICDStringEntry, float]] = []
        for _, score, idx in candidates:
            meta = self.choice_meta[idx]
            score = float(score)
            if len(suggestions_meta) < self.max_suggestions:
                suggestions_meta.append(
                    {
                        "code": meta.code,
                        "text": meta.match_text,
                        "source": meta.match_source,
                        "score": score,
                    }
                )
            if score < self.min_score:
                continue
            if meta.code in _GENERIC_ICD_CODES:
                continue
            if _has_qualifier_conflict(norm, meta.match_text):
                continue
            if not _semantic_overlap(norm, meta.match_text):
                continue
            valid_candidates.append((meta, score))

        # Among valid candidates, prefer derm-specific chapters (12, 2, 17)
        # when scores are within 5 points of the best score
        best_entry: ICDStringEntry | None = None
        best_score: float | None = None
        if valid_candidates:
            top_score = valid_candidates[0][1]
            # Candidates within 5 points of top are considered equivalent
            equivalent = [
                (e, s) for e, s in valid_candidates if s >= top_score - 5.0
            ]
            # Prefer derm chapters among equivalent candidates
            derm_equiv = [
                (e, s) for e, s in equivalent
                if str(e.chapter) in _DERM_PREFERRED_CHAPTERS
            ]
            if derm_equiv:
                best_entry, best_score = derm_equiv[0]
            else:
                best_entry, best_score = equivalent[0]

        # If no valid candidate found, try derm-preferred index as fallback
        if best_entry is None and self._derm_preferred_entries:
            derm_candidates = process.extract(
                raw,
                self._derm_preferred_texts,
                scorer=fuzz.WRatio,
                processor=None,
                limit=10,
            )
            for _, score, idx in derm_candidates:
                meta = self._derm_preferred_entries[idx]
                score = float(score)
                if score < self.min_score:
                    break
                if meta.code in _GENERIC_ICD_CODES:
                    continue
                if _has_qualifier_conflict(norm, meta.match_text):
                    continue
                if not _semantic_overlap(norm, meta.match_text):
                    continue
                best_entry = meta
                best_score = score
                break

        suggestions_str = ";".join(
            f"{s['code']}|{s['text']}|{s['score']:.1f}" for s in suggestions_meta
        )
        if best_entry is None or best_score is None:
            result["suggestions"] = suggestions_str
            return result
        return self._populate_result(
            result,
            best_entry,
            "fuzzy",
            best_score,
            suggestions_meta,
        )

    def _populate_result(
        self,
        base_result: dict[str, str | float | int],
        entry: ICDStringEntry,
        match_type: str,
        score: float,
        suggestions: List[dict[str, str | float]],
    ) -> dict[str, str | float | int]:
        base_result.update(
            {
                "icd_code": entry.code,
                "icd_description": entry.description,
                "chapter": entry.chapter,
                "chapter_title": entry.chapter_title,
                "chapter_range": entry.chapter_range,
                "section_id": entry.section_id,
                "section_desc": entry.section_desc,
                "match_type": match_type,
                "match_source": entry.match_source,
                "match_text": entry.match_text,
                "match_score": float(score),
                "suggestions": ";".join(
                    f"{s['code']}|{s['text']}|{s['score']:.1f}" for s in suggestions
                ),
            }
        )
        return base_result

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        min_score: float = 90.0,
        max_suggestions: int = 5,
        allowed_chapters: Sequence[str] | None = None,
    ) -> "ConditionICDMapper":
        df = pd.read_csv(path)
        return cls(
            df,
            min_score=min_score,
            max_suggestions=max_suggestions,
            allowed_chapters=allowed_chapters,
        )

    def map_conditions(
        self,
        conditions: Sequence[str],
        *,
        overrides: dict[str, ICDStringEntry] | None = None,
    ) -> list[dict[str, str | float | int]]:
        overrides = overrides or {}
        return [self.match_condition(cond, overrides=overrides) for cond in conditions]


# ---------------------------------------------------------------------------
# DataFrame annotation
#
# Loads a condition->ICD mapping CSV and annotates a metadata frame with the derived
# ICD columns, for both ``skinmap.pipeline.run_analysis`` and ``skinmap.pipeline.prep_data``.
# ---------------------------------------------------------------------------


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


__all__ = ["ConditionICDMapper", "normalize_condition"]
