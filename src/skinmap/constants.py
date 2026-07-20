"""Constants used throughout the SkinMap package."""

# Dataset column names
IMG_PATH_COL = "img_path"
DESCRIPTION_COL = "description"
DATASET_DESC_COL = "dataset_desc"
TEXT_COL = "text"

# Common metadata columns
CONDITION_COL = "condition"
AGE_COL = "age"
ORIGIN_COL = "origin"

# Multilabel columns that need special handling
# Note: origin is now single-label, not multi-label
MULTILABEL_COLUMNS = set()  # Empty set - no multi-label columns currently
