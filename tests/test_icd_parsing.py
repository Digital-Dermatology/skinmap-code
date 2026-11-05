"""Tests for ICD-10 tabular parsing helpers."""

import pytest

from scripts.parse_icd10_tabular import split_chapter_desc


@pytest.mark.parametrize(
    "desc, expected",
    [
        ("Neoplasms (C00-D49)", ("Neoplasms", "C00-D49")),
        (
            "Certain infectious and parasitic diseases (A00-B99)",
            ("Certain infectious and parasitic diseases", "A00-B99"),
        ),
        ("No range here", ("No range here", "")),
        ("  Trailing space (L00-L99)  ", ("Trailing space", "L00-L99")),
    ],
)
def test_split_chapter_desc(desc, expected):
    """The '(Range)' suffix must be split out (regression: over-escaped regex left
    every chapter_range empty and the title polluted with the parenthetical)."""
    assert split_chapter_desc(desc) == expected
