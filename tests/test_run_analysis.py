"""
Tests for src/run_analysis.py
Testing new features: continent mapping, category consolidation, and improved heatmaps.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.run_analysis import (
    COUNTRY_TO_CONTINENT,
    MAX_CATEGORIES_DISPLAY,
    consolidate_categories,
    explode_multilabel_field,
    map_origin_to_continent,
    reorder_categories,
    run_prevalence_cross_sections,
    sanitize_series,
    save_heatmap,
)


class TestMapOriginToContinent:
    """Test the country-to-continent mapping functionality."""

    def test_maps_known_countries_to_continents(self):
        """Test that known countries are correctly mapped to continents."""
        df = pd.DataFrame(
            {
                "origin_pred": [
                    "united states",
                    "germany",
                    "china",
                    "brazil",
                    "australia",
                ]
            }
        )
        map_origin_to_continent(df)

        assert df.loc[0, "origin_continent"] == "North America"
        assert df.loc[1, "origin_continent"] == "Europe"
        assert df.loc[2, "origin_continent"] == "Asia"
        assert df.loc[3, "origin_continent"] == "South America"
        assert df.loc[4, "origin_continent"] == "Oceania"

    def test_handles_unknown_countries(self):
        """Test that unknown countries are mapped to 'Other'."""
        df = pd.DataFrame(
            {"origin_pred": ["atlantis", "made_up_place", "unknown_country"]}
        )
        map_origin_to_continent(df)

        assert all(df["origin_continent"] == "Other")

    def test_handles_missing_values(self):
        """Test that NaN and empty values are handled correctly."""
        df = pd.DataFrame({"origin_pred": [None, "", "  ", "unknown", np.nan]})
        map_origin_to_continent(df)

        assert all(df["origin_continent"] == "Unknown")

    def test_case_insensitive_mapping(self):
        """Test that country names are case-insensitive."""
        df = pd.DataFrame({"origin_pred": ["FRANCE", "France", "france", "FrAnCe"]})
        map_origin_to_continent(df)

        assert all(df["origin_continent"] == "Europe")

    def test_handles_whitespace(self):
        """Test that leading/trailing whitespace is handled."""
        df = pd.DataFrame({"origin_pred": ["  japan  ", "japan", " japan"]})
        map_origin_to_continent(df)

        assert all(df["origin_continent"] == "Asia")

    def test_missing_source_column(self):
        """Test graceful handling when source column is missing."""
        df = pd.DataFrame({"some_other_column": ["value"]})
        map_origin_to_continent(df, source_col="origin_pred")

        # Should not crash and should not create the column
        assert "origin_continent" not in df.columns

    def test_comprehensive_country_coverage(self):
        """Test that mapping includes major countries from all continents."""
        # Africa
        assert COUNTRY_TO_CONTINENT["egypt"] == "Africa"
        assert COUNTRY_TO_CONTINENT["south africa"] == "Africa"

        # Asia
        assert COUNTRY_TO_CONTINENT["india"] == "Asia"
        assert COUNTRY_TO_CONTINENT["japan"] == "Asia"

        # Europe
        assert COUNTRY_TO_CONTINENT["united kingdom"] == "Europe"
        assert COUNTRY_TO_CONTINENT["italy"] == "Europe"

        # North America
        assert COUNTRY_TO_CONTINENT["canada"] == "North America"
        assert COUNTRY_TO_CONTINENT["mexico"] == "North America"

        # South America
        assert COUNTRY_TO_CONTINENT["argentina"] == "South America"
        assert COUNTRY_TO_CONTINENT["chile"] == "South America"

        # Oceania
        assert COUNTRY_TO_CONTINENT["new zealand"] == "Oceania"
        assert COUNTRY_TO_CONTINENT["fiji"] == "Oceania"


class TestConsolidateCategories:
    """Test the category consolidation functionality."""

    def test_consolidates_when_exceeding_max_categories(self):
        """Test that categories are consolidated when exceeding limit."""
        # Create series with 20 categories
        categories = [f"cat_{i}" for i in range(20)]
        values = categories * 5  # Each category appears 5 times
        series = pd.Series(values)

        result = consolidate_categories(series, max_categories=10)

        # Should have top 10 categories + "Other"
        unique_values = result.unique()
        assert len(unique_values) <= 11  # 10 top + "Other"
        assert "Other" in unique_values

    def test_preserves_unknown_category(self):
        """Test that 'Unknown' category is preserved separately."""
        categories = [f"cat_{i}" for i in range(20)] + ["Unknown"] * 10
        series = pd.Series(categories)

        result = consolidate_categories(series, max_categories=5)

        # Unknown should be separate from Other
        assert "Unknown" in result.unique()
        assert "Other" in result.unique()
        assert (result == "Unknown").sum() == 10

    def test_keeps_top_categories_by_frequency(self):
        """Test that consolidation keeps the most frequent categories."""
        # Create series where some categories are more frequent
        series = pd.Series(
            ["common1"] * 100
            + ["common2"] * 80
            + ["common3"] * 60
            + [f"rare_{i}" for i in range(20)]  # 20 rare categories with 1 each
        )

        result = consolidate_categories(series, max_categories=3)

        # Top 3 should be preserved
        assert (result == "common1").sum() == 100
        assert (result == "common2").sum() == 80
        assert (result == "common3").sum() == 60
        # Rare ones should be "Other"
        assert (result == "Other").sum() == 20

    def test_no_consolidation_when_under_limit(self):
        """Test that no consolidation occurs when under the limit."""
        series = pd.Series(["cat_a", "cat_b", "cat_c"] * 10)

        result = consolidate_categories(series, max_categories=5)

        # Should be unchanged
        assert "Other" not in result.unique()
        assert len(result.unique()) == 3

    def test_handles_empty_series(self):
        """Test graceful handling of empty series."""
        series = pd.Series([], dtype=str)

        result = consolidate_categories(series)

        assert len(result) == 0

    def test_respects_custom_max_categories(self):
        """Test that custom max_categories parameter is respected."""
        categories = [f"cat_{i}" for i in range(30)]
        series = pd.Series(categories)

        result = consolidate_categories(series, max_categories=8)

        # Should have at most 8 + "Other" + "Unknown" (if present)
        unique_no_other = [v for v in result.unique() if v not in ["Other", "Unknown"]]
        assert len(unique_no_other) <= 8


class TestSaveHeatmap:
    """Test the improved heatmap saving functionality."""

    def test_creates_heatmap_file(self, tmp_path):
        """Test that heatmap file is created successfully."""
        data = pd.DataFrame(
            np.random.rand(5, 5) * 100,
            index=[f"row_{i}" for i in range(5)],
            columns=[f"col_{i}" for i in range(5)],
        )
        output_path = tmp_path / "test_heatmap.png"

        save_heatmap(data, "Test Heatmap", output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_handles_small_heatmap(self, tmp_path):
        """Test handling of small heatmaps (≤10x10)."""
        data = pd.DataFrame(np.random.rand(5, 5) * 100)
        output_path = tmp_path / "small_heatmap.png"

        save_heatmap(data, "Small Heatmap", output_path)

        assert output_path.exists()

    def test_handles_medium_heatmap(self, tmp_path):
        """Test handling of medium heatmaps (≤20x20)."""
        data = pd.DataFrame(np.random.rand(15, 15) * 100)
        output_path = tmp_path / "medium_heatmap.png"

        save_heatmap(data, "Medium Heatmap", output_path)

        assert output_path.exists()

    def test_handles_large_heatmap(self, tmp_path):
        """Test handling of large heatmaps (>20x20)."""
        data = pd.DataFrame(np.random.rand(30, 25) * 100)
        output_path = tmp_path / "large_heatmap.png"

        save_heatmap(data, "Large Heatmap", output_path)

        assert output_path.exists()

    def test_custom_colormap(self, tmp_path):
        """Test that custom colormap is accepted."""
        data = pd.DataFrame(np.random.rand(5, 5) * 100)
        output_path = tmp_path / "custom_cmap.png"

        save_heatmap(data, "Custom Colormap", output_path, cmap="Blues")

        assert output_path.exists()

    def test_closes_figure_properly(self, tmp_path):
        """Test that matplotlib figures are closed to prevent memory leaks."""
        data = pd.DataFrame(np.random.rand(5, 5) * 100)
        output_path = tmp_path / "test.png"

        # Get initial figure count
        initial_figs = len(plt.get_fignums())

        save_heatmap(data, "Test", output_path)

        # Figure count should return to initial
        assert len(plt.get_fignums()) == initial_figs


class TestRunPrevalenceCrossSections:
    """Test the cross-section prevalence analysis."""

    def test_creates_output_directory(self, tmp_path):
        """Test that prevalence directory is created."""
        meta = pd.DataFrame(
            {
                "fitzpatrick_pred": ["I", "II", "III"] * 10,
                "gender_pred": ["male", "female", "male"] * 10,
            }
        )

        feature_names = {
            "fitzpatrick_pred": "Fitzpatrick Skin Type",
            "gender_pred": "Gender",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        prevalence_dir = tmp_path / "prevalence"
        assert prevalence_dir.exists()
        assert prevalence_dir.is_dir()

    def test_generates_csv_and_png_files(self, tmp_path):
        """Test that CSV and PNG files are generated for each pair."""
        meta = pd.DataFrame(
            {
                "fitzpatrick_pred": ["I", "II", "III"] * 10,
                "gender_pred": ["male", "female", "male"] * 10,
            }
        )

        feature_names = {
            "fitzpatrick_pred": "Fitzpatrick Skin Type",
            "gender_pred": "Gender",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        prevalence_dir = tmp_path / "prevalence"
        csv_file = prevalence_dir / "prevalence_fitzpatrick_pred_vs_gender_pred.csv"
        svg_file = prevalence_dir / "prevalence_fitzpatrick_pred_vs_gender_pred.svg"

        assert csv_file.exists()
        assert svg_file.exists()

    def test_consolidates_large_categories(self, tmp_path):
        """Test that large categories are automatically consolidated."""
        # Create feature with many categories
        many_categories = [f"category_{i}" for i in range(30)]
        meta = pd.DataFrame(
            {
                "body_region_pred": np.random.choice(many_categories, 300),
                "gender_pred": np.random.choice(["male", "female"], 300),
            }
        )

        feature_names = {
            "body_region_pred": "Body Region",
            "gender_pred": "Gender",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        csv_file = (
            tmp_path / "prevalence" / "prevalence_body_region_pred_vs_gender_pred.csv"
        )
        assert csv_file.exists()

        # Read the CSV and check that consolidation occurred
        result_df = pd.read_csv(csv_file)
        unique_regions = result_df["Body Region"].unique()

        # Should have at most MAX_CATEGORIES_DISPLAY + "Other" + "Unknown"
        assert len(unique_regions) <= MAX_CATEGORIES_DISPLAY + 2

    def test_skips_when_should_skip_returns_true(self, tmp_path):
        """Test that analysis is skipped when should_skip returns True."""
        meta = pd.DataFrame(
            {
                "fitzpatrick_pred": ["I", "II"] * 5,
                "gender_pred": ["male", "female"] * 5,
            }
        )

        feature_names = {
            "fitzpatrick_pred": "Fitzpatrick Skin Type",
            "gender_pred": "Gender",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return True  # Always skip

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        prevalence_dir = tmp_path / "prevalence"
        # Directory should not be created if skipped
        assert not prevalence_dir.exists()


class TestHelperFunctions:
    """Test helper functions."""

    def test_sanitize_series_replaces_empty_with_unknown(self):
        """Test that sanitize_series handles empty and NaN values."""
        series = pd.Series(["valid", "", None, "  ", "nan", np.nan])

        result = sanitize_series(series)

        assert result[0] == "valid"
        assert result[1] == "Unknown"
        assert result[2] == "Unknown"
        assert result[3] == "Unknown"
        assert result[4] == "Unknown"
        assert result[5] == "Unknown"

    def test_reorder_categories_with_preferred_order(self):
        """Test that reorder_categories respects preferred order."""
        values = ["c", "a", "b", "d"]
        preferred = ["a", "b", "c"]

        result = reorder_categories(values, preferred)

        # Should start with preferred order, then remaining sorted
        assert result == ["a", "b", "c", "d"]

    def test_reorder_categories_without_preferred_order(self):
        """Test that reorder_categories sorts when no preferred order."""
        values = ["zebra", "apple", "monkey", "banana"]

        result = reorder_categories(values, preferred_order=None)

        assert result == ["apple", "banana", "monkey", "zebra"]


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_end_to_end_with_origin_continent(self, tmp_path):
        """Test complete workflow with origin continent mapping."""
        # Create sample data
        meta = pd.DataFrame(
            {
                "origin_pred": ["united states", "germany", "china"] * 10,
                "fitzpatrick_pred": ["I", "III", "V"] * 10,
            }
        )

        # Apply continent mapping
        map_origin_to_continent(meta)

        # Verify mapping worked
        assert "origin_continent" in meta.columns
        assert set(meta["origin_continent"].unique()) == {
            "North America",
            "Europe",
            "Asia",
        }

        # Run prevalence analysis
        feature_names = {
            "origin_continent": "Origin (Continent)",
            "fitzpatrick_pred": "Fitzpatrick Skin Type",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        # Verify outputs
        csv_file = (
            tmp_path
            / "prevalence"
            / "prevalence_origin_continent_vs_fitzpatrick_pred.csv"
        )
        svg_file = (
            tmp_path
            / "prevalence"
            / "prevalence_origin_continent_vs_fitzpatrick_pred.svg"
        )

        assert csv_file.exists()
        assert svg_file.exists()

        # Verify CSV content
        result_df = pd.read_csv(csv_file)
        assert "Origin (Continent)" in result_df.columns
        assert "Fitzpatrick Skin Type" in result_df.columns
        assert "count" in result_df.columns
        assert "prevalence_share" in result_df.columns


class TestExplodeMultilabelField:
    """Test the multilabel field explosion functionality."""

    def test_explodes_list_string(self):
        """Test that string representations of lists are exploded correctly."""
        meta = pd.DataFrame(
            {
                "origin_pred": [
                    "['usa', 'canada']",
                    "['germany']",
                    "['china', 'japan', 'korea']",
                ]
            }
        )

        result = explode_multilabel_field(meta, "origin_pred")

        # Should have 6 rows total (2 + 1 + 3)
        assert len(result) == 6
        assert result[result["origin_pred"] == "usa"].shape[0] == 1
        assert result[result["origin_pred"] == "canada"].shape[0] == 1
        assert result[result["origin_pred"] == "germany"].shape[0] == 1
        assert result[result["origin_pred"] == "china"].shape[0] == 1
        assert result[result["origin_pred"] == "japan"].shape[0] == 1
        assert result[result["origin_pred"] == "korea"].shape[0] == 1

    def test_handles_single_values(self):
        """Test that single values (non-lists) are handled correctly."""
        meta = pd.DataFrame({"origin_pred": ["usa", "canada", "germany"]})

        result = explode_multilabel_field(meta, "origin_pred")

        # Should have same number of rows (no explosion)
        assert len(result) == 3
        assert result["origin_pred"].tolist() == ["usa", "canada", "germany"]

    def test_handles_empty_lists(self):
        """Test that empty lists are converted to 'Unknown'."""
        meta = pd.DataFrame({"origin_pred": ["[]", "['usa']", "[]"]})

        result = explode_multilabel_field(meta, "origin_pred")

        # First and third should be Unknown
        assert result.iloc[0]["origin_pred"] == "Unknown"
        assert result.iloc[1]["origin_pred"] == "usa"
        assert result.iloc[2]["origin_pred"] == "Unknown"

    def test_handles_nan_values(self):
        """Test that NaN values are converted to 'Unknown'."""
        meta = pd.DataFrame({"origin_pred": [np.nan, "['usa']", None]})

        result = explode_multilabel_field(meta, "origin_pred")

        assert result.iloc[0]["origin_pred"] == "Unknown"
        assert result.iloc[1]["origin_pred"] == "usa"
        assert result.iloc[2]["origin_pred"] == "Unknown"

    def test_preserves_index_for_alignment(self):
        """Test that original indices are preserved for proper alignment."""
        meta = pd.DataFrame(
            {"origin_pred": ["['usa', 'canada']", "['germany']", "['china', 'japan']"]},
            index=[10, 20, 30],
        )

        result = explode_multilabel_field(meta, "origin_pred")

        # Check that indices are preserved (duplicated for multiple values)
        assert 10 in result.index
        assert 20 in result.index
        assert 30 in result.index

        # Index 10 should appear twice (usa, canada)
        assert (result.index == 10).sum() == 2
        # Index 20 should appear once (germany)
        assert (result.index == 20).sum() == 1
        # Index 30 should appear twice (china, japan)
        assert (result.index == 30).sum() == 2

    def test_filters_none_values_from_lists(self):
        """Test that None values within lists are filtered out."""
        meta = pd.DataFrame(
            {"origin_pred": ["['usa', None, 'canada']", "['germany', None]"]}
        )

        result = explode_multilabel_field(meta, "origin_pred")

        # Should only have usa, canada, germany (None filtered out)
        assert len(result) == 3
        assert "usa" in result["origin_pred"].values
        assert "canada" in result["origin_pred"].values
        assert "germany" in result["origin_pred"].values
        assert "None" not in result["origin_pred"].values

    def test_handles_invalid_list_strings(self):
        """Test that invalid list strings are treated as single values."""
        meta = pd.DataFrame({"origin_pred": ["[usa, canada", "not a list", "germany]"]})

        result = explode_multilabel_field(meta, "origin_pred")

        # Should treat each as a single value (no explosion)
        assert len(result) == 3

    def test_missing_field_returns_unchanged(self):
        """Test graceful handling when field doesn't exist."""
        meta = pd.DataFrame({"other_field": ["value1", "value2"]})

        result = explode_multilabel_field(meta, "origin_pred")

        # Should return original dataframe
        assert result.equals(meta)


class TestMultilabelPrevalence:
    """Test prevalence calculations with multilabel fields."""

    def test_prevalence_with_multilabel_origin(self, tmp_path):
        """Test that multilabel origin field is properly exploded for prevalence."""
        meta = pd.DataFrame(
            {
                "origin_pred": [
                    "['usa', 'canada']",  # 2 origins
                    "['germany']",  # 1 origin
                    "['usa']",  # 1 origin
                    "['china', 'japan']",  # 2 origins
                    "['usa']",  # 1 origin
                ],
                "gender_pred": ["male", "female", "male", "female", "male"],
            }
        )

        feature_names = {
            "origin_pred": "Origin",
            "gender_pred": "Gender",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        csv_file = tmp_path / "prevalence" / "prevalence_origin_pred_vs_gender_pred.csv"
        assert csv_file.exists()

        result_df = pd.read_csv(csv_file)

        # Check that USA appears correctly (should be 3 times)
        usa_rows = result_df[result_df["Origin"] == "usa"]
        assert usa_rows["count"].sum() == 3  # usa appears 3 times in total

        # Check that each origin is counted separately
        unique_origins = result_df["Origin"].unique()
        assert "usa" in unique_origins
        assert "canada" in unique_origins
        assert "germany" in unique_origins
        assert "china" in unique_origins
        assert "japan" in unique_origins

    def test_multilabel_with_single_label_cross_section(self, tmp_path):
        """Test cross-section between multilabel and single-label fields."""
        meta = pd.DataFrame(
            {
                "origin_pred": [
                    "['usa', 'canada']",
                    "['usa']",
                    "['germany', 'france']",
                ],
                "fitzpatrick_pred": ["I", "II", "III"],
            }
        )

        feature_names = {
            "origin_pred": "Origin",
            "fitzpatrick_pred": "Fitzpatrick",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        csv_file = (
            tmp_path / "prevalence" / "prevalence_origin_pred_vs_fitzpatrick_pred.csv"
        )
        assert csv_file.exists()

        result_df = pd.read_csv(csv_file)

        # First row has ['usa', 'canada'] and FST I
        # Should create 2 entries: (usa, I) and (canada, I)
        usa_fst1 = result_df[
            (result_df["Origin"] == "usa") & (result_df["Fitzpatrick"] == "I")
        ]
        assert len(usa_fst1) == 1
        assert usa_fst1["count"].values[0] == 1

        canada_fst1 = result_df[
            (result_df["Origin"] == "canada") & (result_df["Fitzpatrick"] == "I")
        ]
        assert len(canada_fst1) == 1
        assert canada_fst1["count"].values[0] == 1

    def test_both_fields_multilabel(self, tmp_path):
        """Test cross-section when both fields are multilabel."""
        # Note: In current implementation, only origin_pred is multilabel
        # This test documents the behavior if we had two multilabel fields
        meta = pd.DataFrame(
            {
                "origin_pred": ["['usa', 'canada']", "['germany']"],
                "gender_pred": ["male", "female"],
            }
        )

        feature_names = {
            "origin_pred": "Origin",
            "gender_pred": "Gender",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        csv_file = tmp_path / "prevalence" / "prevalence_origin_pred_vs_gender_pred.csv"
        assert csv_file.exists()

        result_df = pd.read_csv(csv_file)

        # First row should create: (usa, male) and (canada, male)
        # Crosstab creates all combinations, so we check non-zero counts
        male_nonzero = result_df[
            (result_df["Gender"] == "male") & (result_df["count"] > 0)
        ]
        female_nonzero = result_df[
            (result_df["Gender"] == "female") & (result_df["count"] > 0)
        ]

        assert len(male_nonzero) == 2  # usa and canada with male
        assert len(female_nonzero) == 1  # germany with female

    def test_multilabel_consolidation(self, tmp_path):
        """Test that multilabel fields can be consolidated when many categories."""
        # Create many origin values
        origins = [f"['country_{i}']" for i in range(30)]
        meta = pd.DataFrame(
            {
                "origin_pred": origins,
                "gender_pred": ["male", "female"] * 15,
            }
        )

        feature_names = {
            "origin_pred": "Origin",
            "gender_pred": "Gender",
        }

        def csv_path_fn(name):
            return str(tmp_path / name)

        def should_skip_fn(name, label=None):
            return False

        run_prevalence_cross_sections(
            meta, csv_path_fn, should_skip_fn, feature_names, tmp_path
        )

        csv_file = tmp_path / "prevalence" / "prevalence_origin_pred_vs_gender_pred.csv"
        assert csv_file.exists()

        result_df = pd.read_csv(csv_file)

        # Should have consolidated to top N + "Other"
        unique_origins = result_df["Origin"].unique()
        assert (
            "Other" in unique_origins
            or len(unique_origins) <= MAX_CATEGORIES_DISPLAY + 2
        )


class TestMultilabelEdgeCases:
    """Test edge cases for multilabel handling."""

    def test_empty_dataframe(self):
        """Test that empty dataframe is handled gracefully."""
        meta = pd.DataFrame({"origin_pred": []})

        result = explode_multilabel_field(meta, "origin_pred")

        assert len(result) == 0

    def test_all_unknown_values(self):
        """Test when all values are unknown/empty."""
        meta = pd.DataFrame({"origin_pred": [None, np.nan, "", "[]", "nan"]})

        result = explode_multilabel_field(meta, "origin_pred")

        # All should become "Unknown"
        assert all(result["origin_pred"] == "Unknown")
        assert len(result) == 5

    def test_mixed_list_and_single_values(self):
        """Test handling of mixed list and single value formats."""
        meta = pd.DataFrame(
            {
                "origin_pred": [
                    "['usa', 'canada']",  # list
                    "germany",  # single value
                    "['china']",  # list with one item
                    "france",  # single value
                ]
            }
        )

        result = explode_multilabel_field(meta, "origin_pred")

        # Should have 5 rows total
        assert len(result) == 5

        # Check all values are present
        values = result["origin_pred"].tolist()
        assert "usa" in values
        assert "canada" in values
        assert "germany" in values
        assert "china" in values
        assert "france" in values

    def test_whitespace_handling_in_lists(self):
        """Test that whitespace is properly handled in list values."""
        meta = pd.DataFrame(
            {"origin_pred": ["['  usa  ', ' canada ']", "['  germany  ']"]}
        )

        result = explode_multilabel_field(meta, "origin_pred")

        # Whitespace should be stripped
        values = result["origin_pred"].tolist()
        assert "usa" in values
        assert "canada" in values
        assert "germany" in values
        assert "  usa  " not in values
