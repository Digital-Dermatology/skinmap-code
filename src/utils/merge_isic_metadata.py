#!/usr/bin/env python3
"""
Script to merge and harmonize additional ISIC metadata files with the main dataset.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ISICMetadataMerger:
    """Class to handle merging and harmonizing ISIC metadata from multiple sources."""

    def __init__(self, main_metadata_path: str, additional_metadata_dir: str):
        self.main_metadata_path = Path(main_metadata_path)
        self.additional_metadata_dir = Path(additional_metadata_dir)
        self.main_df = None
        self.additional_dfs = {}
        self.merged_df = None

        # Dataset year mapping - fill in actual years as needed
        self.dataset_years = {
            "bcn20000": 2024,
            "ham10000": 2018,
            "mskcc": 2025,
            "derm12345": 2024,
            "prove-ai": 2022,
            "braaff-annotated-acral-lesions-dataset-bald": 2025,
            "consecutive-biopsies-for-melanoma-across-year-2020": 2020,
            "collection-for-isbi-2016-100-lesion-classification": 2016,
            "challenge-2016-test": 2016,
            "challenge-2016-training": 2016,
            "challenge-2017-test": 2017,
            "challenge-2017-training": 2017,
            "challenge-2017-validation": 2017,
            "challenge-2018-task-1-2-test": 2018,
            "challenge-2018-task-1-2-training": 2018,
            "challenge-2018-task-1-2-validation": 2018,
            "challenge-2018-task-3-test": 2018,
            "challenge-2018-task-3-training": 2018,
            "challenge-2018-task-3-validation": 2018,
            "challenge-2019-test": 2019,
            "challenge-2019-training": 2019,
            "challenge-2020-test": 2020,
            "challenge-2020-training": 2020,
            "challenge-2024-training": 2024,
        }

    def load_main_metadata(self) -> pd.DataFrame:
        """Load the main metadata CSV file."""
        logger.info(f"Loading main metadata from {self.main_metadata_path}")
        self.main_df = pd.read_csv(self.main_metadata_path)
        logger.info(
            f"Loaded main metadata with {len(self.main_df)} records and {len(self.main_df.columns)} columns"
        )
        return self.main_df

    def load_additional_metadata(self) -> Dict[str, pd.DataFrame]:
        """Load all additional metadata CSV files."""
        logger.info(f"Loading additional metadata from {self.additional_metadata_dir}")

        for csv_file in self.additional_metadata_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                dataset_name = csv_file.stem.replace("_metadata", "")
                self.additional_dfs[dataset_name] = df
                logger.info(
                    f"Loaded {dataset_name}: {len(df)} records, {len(df.columns)} columns"
                )
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")

        return self.additional_dfs

    def analyze_columns(self) -> Dict:
        """Analyze columns across all datasets to understand the schema differences."""
        logger.info("Analyzing column schemas across all datasets")

        # Get main dataset columns
        main_columns = set(self.main_df.columns)

        # Analyze additional datasets
        analysis = {
            "main_columns": main_columns,
            "additional_columns": {},
            "common_columns": set(),
            "unique_to_additional": set(),
            "column_coverage": {},
        }

        all_additional_columns = set()

        for dataset_name, df in self.additional_dfs.items():
            dataset_columns = set(df.columns)
            analysis["additional_columns"][dataset_name] = dataset_columns
            all_additional_columns.update(dataset_columns)

            # Count non-null values for each column
            coverage = {}
            for col in dataset_columns:
                coverage[col] = df[col].count() / len(df) * 100
            analysis["column_coverage"][dataset_name] = coverage

        # Find common and unique columns
        analysis["common_columns"] = main_columns.intersection(all_additional_columns)
        analysis["unique_to_additional"] = all_additional_columns - main_columns
        analysis["unique_to_main"] = main_columns - all_additional_columns

        logger.info(f"Main dataset has {len(main_columns)} columns")
        logger.info(
            f"Additional datasets collectively have {len(all_additional_columns)} unique columns"
        )
        logger.info(f"Common columns: {len(analysis['common_columns'])}")
        logger.info(
            f"Unique to additional datasets: {len(analysis['unique_to_additional'])}"
        )
        logger.info(f"Unique to main dataset: {len(analysis['unique_to_main'])}")

        return analysis

    def merge_metadata(self) -> pd.DataFrame:
        """Merge all additional metadata with the main metadata."""
        logger.info("Starting metadata merge process")

        # Start with the main dataset
        self.merged_df = self.main_df.copy()

        # Keep track of merge statistics
        merge_stats = {
            "initial_records": len(self.merged_df),
            "datasets_merged": 0,
            "records_added": 0,
            "columns_added": [],
        }

        for dataset_name, additional_df in self.additional_dfs.items():
            logger.info(f"Merging {dataset_name} dataset...")

            # Check if isic_id column exists
            if "isic_id" not in additional_df.columns:
                logger.warning(f"Skipping {dataset_name}: no 'isic_id' column found")
                continue

            # Add dataset year information
            additional_df_copy = additional_df.copy()
            if dataset_name in self.dataset_years:
                additional_df_copy["dataset_year"] = self.dataset_years[dataset_name]
                logger.info(
                    f"Adding dataset year {self.dataset_years[dataset_name]} for {dataset_name}"
                )
            else:
                logger.warning(
                    f"No year defined for dataset {dataset_name}, skipping year assignment"
                )

            # Add dataset source information
            additional_df_copy["dataset_source"] = dataset_name

            # Identify new columns that don't exist in the main dataset
            new_columns = set(additional_df_copy.columns) - set(self.merged_df.columns)
            if new_columns:
                logger.info(f"Adding new columns from {dataset_name}: {new_columns}")
                merge_stats["columns_added"].extend(new_columns)

            # Merge based on isic_id
            initial_count = len(self.merged_df)

            # Perform outer merge to keep all records
            self.merged_df = self.merged_df.merge(
                additional_df_copy,
                on="isic_id",
                how="outer",
                suffixes=("", f"_{dataset_name}"),
            )

            # Handle duplicate columns by combining information
            self._handle_duplicate_columns(dataset_name)

            records_added = len(self.merged_df) - initial_count
            merge_stats["records_added"] += records_added
            merge_stats["datasets_merged"] += 1

            logger.info(f"Merged {dataset_name}: {records_added} new records added")

        logger.info(
            f"Merge complete. Final dataset: {len(self.merged_df)} records, {len(self.merged_df.columns)} columns"
        )
        logger.info(
            f"Summary: {merge_stats['datasets_merged']} datasets merged, "
            f"{merge_stats['records_added']} total records added, "
            f"{len(set(merge_stats['columns_added']))} new columns added"
        )

        return self.merged_df

    def _handle_duplicate_columns(self, dataset_name: str):
        """Handle columns that appear in both datasets by combining the information."""
        # Find columns with suffixes (indicating duplicates)
        duplicate_cols = [
            col for col in self.merged_df.columns if col.endswith(f"_{dataset_name}")
        ]

        for dup_col in duplicate_cols:
            original_col = dup_col.replace(f"_{dataset_name}", "")

            if original_col in self.merged_df.columns:
                # Special handling for dataset_year and dataset_source
                if original_col == "dataset_year":
                    # For year: take the minimum (earliest release)
                    self.merged_df[original_col] = self.merged_df.apply(
                        lambda row: self._min_year(row[original_col], row[dup_col]),
                        axis=1,
                    )
                elif original_col == "dataset_source":
                    # For source: combine with semicolon separator (track all datasets)
                    self.merged_df[original_col] = self.merged_df.apply(
                        lambda row: self._combine_values(
                            row[original_col], row[dup_col]
                        ),
                        axis=1,
                    )
                else:
                    # Standard handling: use original if available, otherwise use the new one
                    self.merged_df[original_col] = self.merged_df[original_col].fillna(
                        self.merged_df[dup_col]
                    )

                # Drop the duplicate column
                self.merged_df.drop(columns=[dup_col], inplace=True)
                logger.debug(f"Combined duplicate column: {original_col}")

    def _combine_values(self, val1, val2):
        """Combine two values with semicolon separator, handling NaN values."""
        if pd.isna(val1) and pd.isna(val2):
            return np.nan
        elif pd.isna(val1):
            return str(val2)
        elif pd.isna(val2):
            return str(val1)
        else:
            # Convert to string and combine if different
            str1, str2 = str(val1), str(val2)
            if str1 == str2:
                return str1
            else:
                return f"{str1};{str2}"

    def _min_year(self, year1, year2):
        """Return the minimum year, handling NaN values."""
        if pd.isna(year1) and pd.isna(year2):
            return np.nan
        elif pd.isna(year1):
            return year2
        elif pd.isna(year2):
            return year1
        else:
            return min(int(year1), int(year2))

    def _add_first_dataset_info(self):
        """Add columns for first dataset and complete dataset history for each sample."""
        logger.info("Adding first dataset and history tracking...")

        # Reconstruct the full history by combining main and additional datasets
        full_history = []

        # Add main dataset records (assume they're from the base ISIC dataset)
        if hasattr(self, "main_df"):
            main_records = self.main_df.copy()
            main_records["dataset_source"] = "main_isic"
            main_records["dataset_year"] = 2024  # Default year for main ISIC dataset
            full_history.append(
                main_records[["isic_id", "dataset_source", "dataset_year"]]
            )

        # Add records from additional datasets
        for dataset_name, df in self.additional_dfs.items():
            if "isic_id" in df.columns:
                dataset_records = df[["isic_id"]].copy()
                dataset_records["dataset_source"] = dataset_name
                dataset_records["dataset_year"] = self.dataset_years.get(
                    dataset_name, np.nan
                )
                full_history.append(dataset_records)

        # Combine all historical records
        if full_history:
            history_df = pd.concat(full_history, ignore_index=True)
            history_df = history_df.dropna(subset=["isic_id"])

            # Sort by year to find chronological order
            history_df = history_df.sort_values(["isic_id", "dataset_year"])

            # Get the first occurrence of each sample
            first_occurrence = history_df.groupby("isic_id").first().reset_index()
            first_occurrence = first_occurrence[
                ["isic_id", "dataset_source", "dataset_year"]
            ].rename(
                columns={
                    "dataset_source": "first_dataset_introduced",
                    "dataset_year": "first_year_introduced",
                }
            )

            # Get the complete history of each sample
            def create_history(group):
                # Sort by year, then by dataset name for consistency
                sorted_group = group.sort_values(["dataset_year", "dataset_source"])
                datasets = sorted_group["dataset_source"].tolist()
                years = sorted_group["dataset_year"].tolist()

                # Create history string with years
                history_parts = []
                for dataset, year in zip(datasets, years):
                    if pd.notna(year):
                        history_parts.append(f"{dataset}({int(year)})")
                    else:
                        history_parts.append(dataset)

                return ";".join(history_parts)

            dataset_history = (
                history_df.groupby("isic_id").apply(create_history).reset_index()
            )
            dataset_history.columns = ["isic_id", "dataset_history"]

            # Merge first dataset info with the main dataset
            self.merged_df = self.merged_df.merge(
                first_occurrence, on="isic_id", how="left"
            )

            # Merge dataset history with the main dataset
            self.merged_df = self.merged_df.merge(
                dataset_history, on="isic_id", how="left"
            )

            # Log statistics
            first_dataset_counts = self.merged_df[
                "first_dataset_introduced"
            ].value_counts()
            first_year_counts = (
                self.merged_df["first_year_introduced"].value_counts().sort_index()
            )

            logger.info("First dataset distribution:")
            for dataset, count in first_dataset_counts.items():
                logger.info(f"  {dataset}: {count} samples")

            logger.info("First introduction year distribution:")
            for year, count in first_year_counts.items():
                logger.info(f"  {year}: {count} samples")

            # Log some examples of dataset history
            logger.info("Sample dataset histories:")
            sample_histories = self.merged_df[
                ["isic_id", "first_dataset_introduced", "dataset_history"]
            ].head(5)
            for _, row in sample_histories.iterrows():
                logger.info(
                    f"  {row['isic_id']}: first in {row['first_dataset_introduced']}, history: {row['dataset_history']}"
                )

        else:
            logger.warning("No historical data available for first dataset tracking")
            self.merged_df["first_dataset_introduced"] = np.nan
            self.merged_df["first_year_introduced"] = np.nan
            self.merged_df["dataset_history"] = np.nan

    def clean_merged_data(self):
        """Clean and validate the merged dataset."""
        logger.info("Cleaning merged dataset...")

        initial_records = len(self.merged_df)

        # Remove records without isic_id
        self.merged_df = self.merged_df[self.merged_df["isic_id"].notna()]

        # Remove duplicate isic_id entries (keep first occurrence)
        initial_unique = len(self.merged_df)
        self.merged_df = self.merged_df.drop_duplicates(
            subset=["isic_id"], keep="first"
        )
        duplicates_removed = initial_unique - len(self.merged_df)

        if duplicates_removed > 0:
            logger.warning(f"Removed {duplicates_removed} duplicate isic_id entries")

        # Add first dataset description and year tracking
        self._add_first_dataset_info()

        # Reset index
        self.merged_df.reset_index(drop=True, inplace=True)

        records_removed = initial_records - len(self.merged_df)
        logger.info(
            f"Cleaning complete. {records_removed} records removed. Final: {len(self.merged_df)} records"
        )

        return self.merged_df

    def save_merged_metadata(self, output_path: str):
        """Save the merged metadata to a new CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged metadata saved to {output_path}")

        # Save a summary report
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        self._save_summary_report(summary_path)

    def _save_summary_report(self, summary_path: Path):
        """Save a summary report of the merge process."""
        with open(summary_path, "w") as f:
            f.write("ISIC Metadata Merge Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Main dataset records: {len(self.main_df)}\n")
            f.write(f"Final merged records: {len(self.merged_df)}\n")
            f.write(
                f"Total columns in merged dataset: {len(self.merged_df.columns)}\n\n"
            )

            f.write("Additional datasets processed:\n")
            for name, df in self.additional_dfs.items():
                f.write(f"  - {name}: {len(df)} records, {len(df.columns)} columns\n")

            f.write("\nColumns in final dataset:\n")
            for col in sorted(self.merged_df.columns):
                non_null_count = self.merged_df[col].count()
                percentage = (non_null_count / len(self.merged_df)) * 100
                f.write(
                    f"  - {col}: {non_null_count}/{len(self.merged_df)} ({percentage:.1f}%)\n"
                )

            # Add first dataset information summary
            if "first_dataset_introduced" in self.merged_df.columns:
                f.write("\nFirst Dataset Introduction Summary:\n")
                f.write("-" * 40 + "\n")

                first_dataset_counts = self.merged_df[
                    "first_dataset_introduced"
                ].value_counts()
                f.write("Distribution by first dataset:\n")
                for dataset, count in first_dataset_counts.items():
                    percentage = (count / len(self.merged_df)) * 100
                    f.write(f"  - {dataset}: {count} samples ({percentage:.1f}%)\n")

                f.write("\n")
                first_year_counts = (
                    self.merged_df["first_year_introduced"].value_counts().sort_index()
                )
                f.write("Distribution by first introduction year:\n")
                for year, count in first_year_counts.items():
                    percentage = (count / len(self.merged_df)) * 100
                    f.write(f"  - {year}: {count} samples ({percentage:.1f}%)\n")

                # Add sample dataset histories
                f.write("\nSample Dataset Histories (first 10):\n")
                sample_histories = self.merged_df[
                    ["isic_id", "first_dataset_introduced", "dataset_history"]
                ].head(10)
                for _, row in sample_histories.iterrows():
                    f.write(
                        f"  {row['isic_id']}: first in {row['first_dataset_introduced']}, history: {row['dataset_history']}\n"
                    )

        logger.info(f"Summary report saved to {summary_path}")

    def export_first_dataset_info(self, output_path: str):
        """Export the first dataset and complete history for each sample."""
        if "first_dataset_introduced" not in self.merged_df.columns:
            logger.error(
                "First dataset information not available. Run merge process first."
            )
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select relevant columns including the new history column
        export_cols = [
            "isic_id",
            "first_dataset_introduced",
            "first_year_introduced",
            "dataset_history",
        ]

        # Add additional useful columns if available
        optional_cols = ["diagnosis_1", "sex", "age_approx", "anatom_site_general"]
        for col in optional_cols:
            if col in self.merged_df.columns:
                export_cols.append(col)

        export_df = self.merged_df[export_cols].copy()
        export_df = export_df.sort_values(
            ["first_year_introduced", "first_dataset_introduced", "isic_id"]
        )

        export_df.to_csv(output_path, index=False)
        logger.info(f"First dataset and history information exported to {output_path}")
        logger.info(
            f"Exported {len(export_df)} samples with first dataset and history tracking"
        )

        return export_df


def main():
    """Main function to execute the metadata merging process."""
    # Configuration
    main_metadata_path = "/data/ISIC/metadata.csv"
    additional_metadata_dir = "/data/ISIC/additional_metadata"
    output_path = "/data/ISIC/metadata_merged.csv"

    # Initialize the merger
    merger = ISICMetadataMerger(main_metadata_path, additional_metadata_dir)

    try:
        # Load all metadata
        merger.load_main_metadata()
        merger.load_additional_metadata()

        # Analyze the schemas
        analysis = merger.analyze_columns()

        # Print some analysis results
        print("\nColumn Analysis:")
        print(
            f"Columns unique to additional datasets: {sorted(analysis['unique_to_additional'])}"
        )
        print(f"Columns unique to main dataset: {sorted(analysis['unique_to_main'])}")

        # Perform the merge
        merger.merge_metadata()

        # Clean the merged data
        merger.clean_merged_data()

        # Save the results
        merger.save_merged_metadata(output_path)

        print("\nMerge completed successfully!")
        print(f"Original dataset: {len(merger.main_df)} records")
        print(f"Merged dataset: {len(merger.merged_df)} records")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during merge process: {e}")
        raise


if __name__ == "__main__":
    main()
