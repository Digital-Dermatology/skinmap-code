from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


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

        candidates = process.extract(
            raw,
            self.choice_texts,
            scorer=fuzz.WRatio,
            processor=None,
            limit=self.max_suggestions,
        )
        suggestions_meta: list[dict[str, str | float]] = []
        best_entry: ICDStringEntry | None = None
        best_score: float | None = None
        for text, score, idx in candidates:
            meta = self.choice_meta[idx]
            score = float(score)
            suggestions_meta.append(
                {
                    "code": meta.code,
                    "text": meta.match_text,
                    "source": meta.match_source,
                    "score": score,
                }
            )
            if best_entry is None and score >= self.min_score:
                best_entry = meta
                best_score = score
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


__all__ = ["ConditionICDMapper", "normalize_condition"]
