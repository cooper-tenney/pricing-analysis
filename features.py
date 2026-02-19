"""
Feature engineering: target inference, extraction, date expansion, numeric transforms.
All operations work on copies; original dataframe is never mutated.
"""

import numpy as np
import pandas as pd

from cleaning import DATE_COLUMNS, _to_numeric_series

# Pricing keywords (lowercase) for target inference
PRICING_KEYWORDS = [
    "unit price", "price", "rate", "amount", "total", "cost",
    "revenue", "arr", "mrr", "acv", "tcv", "value",
]
# ID-like substrings to penalize
ID_LIKE_KEYWORDS = ["id", "guid", "uuid", "key", "number"]
# Min non-null fraction after numeric conversion to consider a column
TARGET_NUMERIC_MIN_FRAC = 0.70
# Min score to accept an inferred target (heuristic)
TARGET_MIN_SCORE = 0.1


def infer_target_column(df: pd.DataFrame, user_target: str | None = None) -> str:
    """
    Infer or validate regression target column.
    If user_target is provided and exists, return it.
    Else use heuristic scoring. Raises ValueError if no suitable target found.
    """
    if user_target is not None and user_target in df.columns:
        return user_target

    nrows = len(df)
    candidates = []
    for col in df.columns:
        ser = df[col]
        # Skip datetime
        if pd.api.types.is_datetime64_any_dtype(ser):
            continue
        # Skip obvious bool
        if ser.dtype == bool:
            continue
        if ser.dtype == object and ser.nunique() <= 2:
            vals = set(ser.dropna().astype(str).str.strip().str.lower())
            if vals <= {"true", "false", "yes", "no", "0", "1", "y", "n", ""}:
                continue

        converted = _to_numeric_series(ser)
        non_null_frac = converted.notna().mean()
        if non_null_frac < TARGET_NUMERIC_MIN_FRAC:
            continue

        nunique = converted.nunique()
        name_lower = col.lower()

        # Penalize ID-like
        id_penalty = 0.0
        if nunique / max(nrows, 1) > 0.9:
            id_penalty += 0.5
        if any(kw in name_lower for kw in ID_LIKE_KEYWORDS):
            id_penalty += 0.3

        # Prefer pricing keywords
        keyword_bonus = 0.0
        for kw in PRICING_KEYWORDS:
            if kw in name_lower:
                keyword_bonus += 0.4
                break

        # Penalize year-like (e.g. Close Year)
        if "year" in name_lower and nunique <= 10:
            id_penalty += 0.2

        score = non_null_frac * 0.3 + keyword_bonus - id_penalty
        score = max(0, score)
        candidates.append((col, score, non_null_frac))

    if not candidates:
        raise ValueError(
            "No suitable target column found. Need at least one column with >= 70% "
            "numeric-convertible values. Use --target to specify explicitly."
        )

    candidates.sort(key=lambda x: -x[1])
    best_col, best_score, best_frac = candidates[0]
    if best_score < TARGET_MIN_SCORE:
        top10 = "\n  ".join(f"{c[0]} (frac={c[2]:.2f})" for c in candidates[:10])
        raise ValueError(
            f"No target scored above threshold. Best candidates:\n  {top10}\n"
            "Use --target to specify explicitly."
        )
    return best_col


def extract_target(series: pd.Series) -> pd.Series:
    """
    Convert target series to numeric: strip commas, currency, blanks; coerce errors to NaN.
    Returns a new series.
    """
    return _to_numeric_series(series)


def expand_dates_to_features(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """
    Expand each date column into _year and _month numeric features, then drop the
    original date columns. Works on a copy. Datetime columns only.
    """
    df = df.copy()
    for c in date_columns:
        if c not in df.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}_year"] = df[c].dt.year
        df[f"{c}_month"] = df[c].dt.month
    cols_to_drop = [c for c in date_columns if c in df.columns]
    return df.drop(columns=cols_to_drop)


def add_numeric_engineered_features(
    X: pd.DataFrame,
    numeric_cols: list[str],
    mode: str = "basic",
    nrows: int | None = None,
) -> tuple[pd.DataFrame, list[str], list[tuple[str, str]]]:
    """
    Add lightweight nonlinear features for numeric columns. No leakage: transforms
    are per-column without using target.
    numeric_cols: explicit list of columns to consider (no auto-detection).
    mode: "none" -> no change; "basic" -> log1p, sqrt, winsorized.
    Applies log1p/sqrt only when min>=0 and column is not constant.
    Skips winsorize for columns with nunique > 0.5*nrows (likely IDs).
    Returns (X_modified, engineered_base_cols, skipped_with_reasons).
    engineered_base_cols = base column names we added features for.
    skipped_with_reasons = [(col, reason)] for cols we skipped (technical reasons).
    """
    if mode != "basic" or not numeric_cols:
        return X, [], []
    X = X.copy()
    nrows = nrows or len(X)
    engineered_base: list[str] = []
    skipped: list[tuple[str, str]] = []
    for col in numeric_cols:
        if col not in X.columns:
            continue
        ser = pd.to_numeric(X[col], errors="coerce")
        if ser.isna().all():
            skipped.append((col, "all NaN"))
            continue
        nunique = ser.nunique()
        if nunique <= 1:
            skipped.append((col, "constant"))
            continue
        added_any = False
        # log1p/sqrt only when min>=0 and not constant
        nonneg = (ser >= 0) & ser.notna()
        if nonneg.all():
            X[f"{col}_log1p"] = np.log1p(np.maximum(0, ser))
            X[f"{col}_sqrt"] = np.sqrt(np.maximum(0, ser))
            added_any = True
        # Winsorize: skip if nunique > 0.5*nrows (likely IDs)
        if nunique <= 0.5 * nrows:
            lo = float(ser.quantile(0.01))
            hi = float(ser.quantile(0.99))
            X[f"{col}_winsor"] = ser.clip(lower=lo, upper=hi)
            added_any = True
        if added_any:
            engineered_base.append(col)
        else:
            reasons = []
            if not nonneg.all():
                reasons.append("contains negatives")
            if nunique > 0.5 * nrows:
                reasons.append("high unique ratio (likely ID)")
            skipped.append((col, "; ".join(reasons) if reasons else "skipped"))
    return X, engineered_base, skipped


def prepare_model_data(
    df_clean: pd.DataFrame,
    target_name: str,
    numeric_feature_engineering: str = "basic",
    safe_numeric_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict, dict]:
    """
    From a cleaned dataframe, build feature matrix X and target y for regression.
    - Drops rows where target is missing or non-numeric after conversion.
    - Leakage and identifier columns should be dropped before calling (via audit recommend_drops).
    - Expands temporal columns (DATE_COLUMNS + any datetime dtype) into year/month.
    - Numeric feature engineering applies only to safe_numeric_cols (from audit).
    Returns (X, y, column_info, numeric_eng_info). Never mutates df_clean.
    numeric_eng_info: {"engineered_numeric_cols": [...], "skipped_numeric_cols": {col: reason}}
    """
    df = df_clean.copy(deep=True)

    if target_name not in df.columns:
        raise ValueError(f"Target column '{target_name}' not found in dataframe.")
    y_processed = extract_target(df[target_name])
    valid_mask = y_processed.notna()
    y = y_processed.loc[valid_mask].copy()
    df = df.loc[valid_mask].copy()

    X = df.drop(columns=[target_name]).copy()

    # Expand date columns: named in DATE_COLUMNS + any datetime dtype
    date_cols_to_expand = [c for c in DATE_COLUMNS if c in X.columns]
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for c in dt_cols:
        if c not in date_cols_to_expand:
            date_cols_to_expand.append(c)
    X = expand_dates_to_features(X, date_cols_to_expand)

    # Drop any remaining datetime columns
    dt_remain = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    if dt_remain:
        X = X.drop(columns=dt_remain)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Only engineer columns in safe_numeric_cols that exist in X
    cols_to_engineer = [c for c in (safe_numeric_cols or []) if c in X.columns]
    X, engineered_base, skipped_tech = add_numeric_engineered_features(
        X, cols_to_engineer, mode=numeric_feature_engineering, nrows=len(X)
    )
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_eng_info = {
        "engineered_numeric_cols": engineered_base,
        "skipped_numeric_cols": dict(skipped_tech),
    }

    column_info = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
    }
    return X, y, column_info, numeric_eng_info
