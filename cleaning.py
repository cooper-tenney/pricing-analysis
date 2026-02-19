"""
Data cleaning utilities for messy business CSV data.
Never mutates the original dataframe; all operations work on copies.
"""

import numpy as np
import pandas as pd


# --- Column classification (explicit business rules) ---
DATE_COLUMNS = ["Close Date", "Contract Start Date", "Contract End Date"]
ID_COLUMNS = [
    "Quote Number",
    "Quote Line ID",
    "Opportunity : Account Name : Account ID",
]
# Columns that would leak the target or are derived from it; excluded from modeling.
LEAKAGE_COLUMNS = [
    "Customer Total",
    "Customer Amount",
    "Quote ARR",
    "Quote Line ARR",
    "Growth ARR",
]

# Thresholds for automatic column classification of object columns
NUMERIC_STRING_FRAC_THRESHOLD = 0.6  # If >60% of values parse as numeric, treat as quantitative
CATEGORICAL_NUNIQUE_THRESHOLD = 200  # If fewer unique values, treat as categorical


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Classify columns into quantitative, categorical, textual, temporal, and identifier.
    Uses explicit date/id lists and heuristics for object columns.
    """
    quantitative: list[str] = []
    categorical: list[str] = []
    textual: list[str] = []
    temporal: list[str] = []
    identifier: list[str] = []

    for c in df.columns:
        if c in DATE_COLUMNS:
            temporal.append(c)
            continue
        if c in ID_COLUMNS:
            identifier.append(c)
            continue

        if df[c].dtype == "object":
            # Check what fraction of values parse as numeric (after stripping commas and $)
            ser = df[c].astype(str)
            cleaned = ser.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
            # Also remove parentheses for negative numbers
            cleaned = cleaned.str.replace(r"^\(([^)]*)\)$", r"-\1", regex=True)
            numeric_parsed = pd.to_numeric(cleaned, errors="coerce")
            frac_num = numeric_parsed.notna().mean()

            nunique = df[c].nunique()

            if frac_num >= NUMERIC_STRING_FRAC_THRESHOLD:
                quantitative.append(c)
            elif nunique < CATEGORICAL_NUNIQUE_THRESHOLD:
                categorical.append(c)
            else:
                textual.append(c)
        else:
            quantitative.append(c)

    return {
        "quantitative": quantitative,
        "categorical": categorical,
        "textual": textual,
        "temporal": temporal,
        "identifier": identifier,
    }


def _to_numeric_series(ser: pd.Series) -> pd.Series:
    """
    Convert a series to numeric, handling commas, currency symbols, blanks, and (x) for negative.
    Returns a new series; does not mutate.
    """
    out = ser.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    # Treat (1,234) as -1234
    out = out.str.replace(r"^\(([^)]*)\)$", r"-\1", regex=True)
    # Coerce non-numeric to NaN
    return pd.to_numeric(out, errors="coerce")


def convert_numeric_like_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert numeric-like string columns to float. Works on a copy.
    """
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            continue
        if df[c].dtype == "object":
            df[c] = _to_numeric_series(df[c])
    return df


def parse_date_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Safely parse date columns to datetime. Non-parseable values become NaT.
    Works on a copy.
    """
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            continue
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def impute_missing(df: pd.DataFrame, quantitative: list[str], categorical: list[str]) -> pd.DataFrame:
    """
    Impute numeric columns with median and categorical with mode.
    Only imputes columns that exist and have missing values. Works on a copy.
    """
    df = df.copy()
    for c in quantitative:
        if c not in df.columns:
            continue
        if df[c].dtype in ("number", "float64", "int64") or np.issubdtype(df[c].dtype, np.number):
            median_val = df[c].median()
            if pd.isna(median_val):
                median_val = 0
            df[c] = df[c].fillna(median_val)
    for c in categorical:
        if c not in df.columns:
            continue
        mode_val = df[c].mode()
        if len(mode_val) > 0:
            df[c] = df[c].fillna(mode_val.iloc[0])
        else:
            df[c] = df[c].fillna("__MISSING__")
    return df


def drop_high_missing_columns(df: pd.DataFrame, missing_pct_threshold: float = 80.0) -> pd.DataFrame:
    """
    Drop columns with more than missing_pct_threshold % missing values.
    Returns a copy with those columns dropped.
    """
    pct = df.isna().mean() * 100
    to_drop = pct[pct > missing_pct_threshold].index.tolist()
    if not to_drop:
        return df.copy()
    return df.drop(columns=to_drop).copy()


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: classify columns, convert numeric-like strings,
    parse dates, drop very-high-missing columns, impute the rest.
    Never mutates df_raw; returns a new dataframe.
    """
    df = df_raw.copy(deep=True)
    roles = classify_columns(df)

    # Drop columns with >80% missing first (so we don't impute mostly-empty columns)
    df = drop_high_missing_columns(df, missing_pct_threshold=80.0)

    # Convert object columns that were classified as quantitative to numeric
    quant = [c for c in roles["quantitative"] if c in df.columns]
    df = convert_numeric_like_columns(df, quant)

    # Re-classify after conversion: former quantitative object cols may now be numeric
    roles = classify_columns(df)
    quant = [c for c in roles["quantitative"] if c in df.columns]
    cat = [c for c in roles["categorical"] if c in df.columns]
    temporal = [c for c in roles["temporal"] if c in df.columns]

    # Parse date columns
    df = parse_date_columns(df, temporal)

    # Impute: numeric with median, categorical with mode
    df = impute_missing(df, quant, cat)

    return df
