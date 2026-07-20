import pandas as pd

from src.prep_data import (
    annotate_icd,
    clean_condition_series,
    harmonize_age,
    harmonize_dataframe,
    harmonize_fitzpatrick,
    harmonize_gender,
    log_value_counts_summary,
    merge_concise_captions,
    preprocess_body_location,
)


def test_harmonize_body_location_extracts_laterality_and_region():
    df = pd.DataFrame({"body_location": ["Left arm"]})
    out = harmonize_dataframe(df)
    assert out.loc[0, "laterality"] == "left"
    assert out.loc[0, "body_region"] == "upper_limb"


def test_merge_concise_captions_updates_matching_rows(tmp_path):
    concise_path = tmp_path / "concise.csv"
    pd.DataFrame([{"img_path": "a.jpg", "concise_description": "short text"}]).to_csv(
        concise_path, index=False
    )
    df = pd.DataFrame(
        [
            {
                "img_path": "a.jpg",
                "description": "Long text",
                "description_short": "Long text",
            },
            {
                "img_path": "b.jpg",
                "description": "Long text B",
                "description_short": "already short",
            },
        ]
    )
    out = merge_concise_captions(df, concise_path)
    assert out.loc[0, "description_short"] == "short text"
    assert out.loc[1, "description_short"] == "already short"


def test_clean_condition_series_strips_noise():
    series = pd.Series(["eczema_(hand)", "foo-bar", None])
    cleaned = clean_condition_series(series)
    result = [None if pd.isna(x) else x for x in cleaned]
    assert result == ["eczema", "foo bar", None]


def test_annotate_icd_adds_expected_columns():
    meta = pd.DataFrame({"condition": ["acne"]})
    mapping = pd.DataFrame(
        [
            {
                "condition_raw": "acne",
                "icd_code": "L70.0",
                "icd_description": "Acne vulgaris",
                "chapter": "12",
                "chapter_title": "Diseases of the skin",
                "chapter_range": "L00-L99",
                "section_id": "L70-L75",
                "section_desc": "Acne and related disorders",
            }
        ]
    )
    out = annotate_icd(meta, mapping, "condition")
    assert out.loc[0, "icd_code"] == "L70.0"
    assert out.loc[0, "icd_category"] == "L70"
    assert out.loc[0, "icd_block"] == "L70-L75"


def test_preprocess_body_location_splits_and_sorts():
    series = pd.Series(["Arm & leg-left", None])
    cleaned = preprocess_body_location(series)
    result = [None if pd.isna(x) else x for x in cleaned.tolist()]
    assert result == ["arm, left, leg", None]


def test_harmonize_fitzpatrick_maps_tokens_to_ints():
    series = pd.Series(["fitzpatrick skin type II", "FSTV", -1, 34, "NONE_IDENTIFIED"])
    mapped = harmonize_fitzpatrick(series)
    result = [None if pd.isna(x) else int(x) for x in mapped]
    assert result == [2, 5, None, 4, None]


def test_harmonize_age_bins_ranges_and_unknowns():
    series = pd.Series(["AGE_18_TO_29", "AGE_UNKNOWN", 42, None, 3060])
    mapped = harmonize_age(series)
    result = [None if pd.isna(x) else x for x in mapped]
    assert result == [25, None, 42, None, None]


def test_harmonize_gender_collapses_to_binary():
    series = pd.Series(["M", "female", "Other_or_unspecified", "unknown", None, "F"])
    mapped = harmonize_gender(series)
    expected = ["male", "female", None, None, None, "female"]
    assert mapped.where(pd.notna(mapped), None).tolist() == expected


def test_log_value_counts_summary_reports_key_groups(caplog):
    df = pd.DataFrame(
        {
            "dataset_desc": ["A", "A", "ISIC-foo"],
            "modality": ["clinical", None, "dermoscopy"],
            "release_year": [2024, None, 2023],
            "body_location": ["Left arm", "Right leg", "Unknown"],
            "gender": ["M", "F", "other_or_unspecified"],
            "fitzpatrick": ["II", "FSTV", None],
            "age": ["AGE_18_TO_29", "AGE_UNKNOWN", 42],
            "origin": [["US"], None, ["DE"]],
            "condition": ["dermatitis foo", "acne", "dermatitis bar"],
        }
    )
    df = harmonize_dataframe(df)
    df["gender"] = harmonize_gender(df["gender"])
    df["fitzpatrick"] = harmonize_fitzpatrick(df["fitzpatrick"])
    df["age"] = harmonize_age(df["age"])

    with caplog.at_level("INFO"):
        summaries = log_value_counts_summary(df, condition_col="condition", top_n=5)

    # gender collapsed to two categories plus NaN
    gender_counts = summaries["gender"].copy()
    gender_counts.index = ["nan" if pd.isna(x) else x for x in gender_counts.index]
    assert gender_counts.to_dict() == {"male": 1, "female": 1, "nan": 1}

    assert "dataset_desc (missing modality)" in summaries
    assert "dataset_desc (missing release_year)" in summaries
    assert "dataset_desc (ISIC only)" in summaries
    assert "dataset_desc (condition contains 'dermatitis')" in summaries
    assert "dataset_desc (missing modality)" in caplog.text
    assert "dataset_desc (top 5, total=3" in caplog.text
    # Extra harmonizations covered
    assert "fitzpatrick" in summaries
    assert "age" in summaries
    assert "origin" in summaries
