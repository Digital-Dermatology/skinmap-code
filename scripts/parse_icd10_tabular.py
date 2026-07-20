#!/usr/bin/env python3
"""
Convert the official ICD-10-CM tabular XML into a flat CSV table.

This script focuses on extracting enough structure for downstream condition
mapping:
    - chapter metadata (number, title, range)
    - section (block) identifiers and descriptions
    - each diagnosis code with optional synonyms/notes
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator, Iterable

DEFAULT_OUTPUT = Path("assets/icd10/icd10cm_tabular_2026.csv")


def extract_text(node: ET.Element | None) -> list[str]:
    """Return all stripped text chunks under a node (preorder traversal)."""
    if node is None:
        return []
    parts: list[str] = []
    text = (node.text or "").strip()
    if text:
        parts.append(text)
    for child in node:
        parts.extend(extract_text(child))
    tail = (node.tail or "").strip()
    if tail:
        parts.append(tail)
    return parts


def extract_terms(diag: ET.Element, tag_names: Iterable[str]) -> list[str]:
    """Collect inclusion terms/notes under the given tag names."""
    terms: list[str] = []
    for tag in tag_names:
        for node in diag.findall(tag):
            for text in extract_text(node):
                if text:
                    terms.append(text)
    return terms


def split_chapter_desc(desc: str) -> tuple[str, str]:
    """Split 'Title (Range)' -> ('Title', 'Range')."""
    match = re.match(r"^(.*?)(?:\\s*\\(([^)]+)\\))?$", desc.strip())
    if not match:
        return desc.strip(), ""
    title = match.group(1).strip()
    rng = (match.group(2) or "").strip()
    return title, rng


def iter_diag(
    diag: ET.Element,
    *,
    chapter_num: str,
    chapter_title: str,
    chapter_range: str,
    section_id: str,
    section_desc: str,
    parent_code: str | None,
) -> Generator[dict[str, str], None, None]:
    code = (diag.findtext("name") or "").strip()
    if not code:
        return
    desc = (diag.findtext("desc") or "").strip()
    children = diag.findall("diag")
    synonyms = extract_terms(diag, ["inclusionTerm"])
    notes = extract_terms(
        diag,
        [
            "note",
            "includes",
            "excludes1",
            "excludes2",
            "useAdditionalCode",
            "codeFirst",
            "codeAlso",
            "sevenChrNote",
        ],
    )
    yield {
        "code": code,
        "description": desc,
        "chapter": chapter_num,
        "chapter_title": chapter_title,
        "chapter_range": chapter_range,
        "section_id": section_id,
        "section_desc": section_desc,
        "parent_code": parent_code or "",
        "has_children": "1" if children else "0",
        "synonyms": "|".join(dict.fromkeys(synonyms)),
        "notes": "|".join(dict.fromkeys(notes)),
    }
    for child in children:
        yield from iter_diag(
            child,
            chapter_num=chapter_num,
            chapter_title=chapter_title,
            chapter_range=chapter_range,
            section_id=section_id,
            section_desc=section_desc,
            parent_code=code,
        )


def parse_tabular(
    xml_path: Path, chapter_filter: str | None = None
) -> list[dict[str, str]]:
    root = ET.parse(xml_path).getroot()
    rows: list[dict[str, str]] = []
    for chapter in root.findall("chapter"):
        chapter_num = (chapter.findtext("name") or "").strip()
        chapter_desc = (chapter.findtext("desc") or "").strip()
        if chapter_filter:
            if chapter_filter not in (chapter_num, chapter_desc):
                # Allow filtering via substring match on description/range too.
                if chapter_filter not in chapter_desc:
                    continue
        chapter_title, chapter_range = split_chapter_desc(chapter_desc)
        for section in chapter.findall("section"):
            section_id = section.attrib.get("id", "")
            section_desc = (section.findtext("desc") or "").strip()
            for diag in section.findall("diag"):
                rows.extend(
                    iter_diag(
                        diag,
                        chapter_num=chapter_num,
                        chapter_title=chapter_title,
                        chapter_range=chapter_range,
                        section_id=section_id,
                        section_desc=section_desc,
                        parent_code=None,
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten ICD-10-CM tabular XML to CSV."
    )
    parser.add_argument("xml_path", type=Path, help="Path to icd10cm_tabular_*.xml")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--chapter",
        help="Optional chapter ID or substring filter (e.g., '12' or 'Dermatology').",
    )
    args = parser.parse_args()

    rows = parse_tabular(args.xml_path, chapter_filter=args.chapter)
    if not rows:
        print("No rows produced; check chapter filter?", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "code",
        "description",
        "chapter",
        "chapter_title",
        "chapter_range",
        "section_id",
        "section_desc",
        "parent_code",
        "has_children",
        "synonyms",
        "notes",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} codes to {args.output}")


if __name__ == "__main__":
    main()
