"""
Data audit: flag problematic columns before modeling.
Does not auto-drop; only flags and reports. Dropping stays in prepare_model_data.
"""

from typing import Optional

import numpy as np
import pandas as pd

# Keywords suggesting ID-like columns
ID_LIKE_KEYWORDS = ["id", "guid", "uuid", "key", "number", "hash"]
# Keywords suggesting potential target leakage (excluding target itself)
LEAKAGE_KEYWORDS = [
    "close", "won", "booked", "revenue", "arr", "price", "amount",
    "billed", "invoice", "contract_end", "total", "customer_total",
    "customer_amount", "quote_arr", "growth_arr",
]
# Robust z-score threshold (using median and MAD)
ROBUST_Z_THRESHOLD = 3.5
# Dominant category threshold for near_constant
DOMINANT_CATEGORY_PCT = 95
HIGH_MISSING_PCT = 40
NEAR_CONSTANT_NUNIQUE = 2
HIGH_CARDINALITY_NUNIQUE = 200
# nunique / nrows ratio above which column looks like an ID
ID_LIKE_UNIQUE_RATIO = 0.85


def _robust_z_score(ser: pd.Series) -> tuple[int, float]:
    """
    Count of extreme outliers (|robust z| > threshold) and max magnitude.
    Robust z = (x - median) / MAD, MAD = median(|x - median|).
    Returns (count, max_magnitude).
    """
    valid = ser.dropna()
    if len(valid) < 4:
        return 0, 0.0
    med = valid.median()
    mad = (valid - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return 0, 0.0
    z = (valid - med) / (1.4826 * mad)
    extreme = (z.abs() > ROBUST_Z_THRESHOLD).sum()
    max_mag = float(z.abs().max()) if len(z) else 0.0
    return int(extreme), max_mag


def _is_near_constant(ser: pd.Series) -> tuple[bool, str]:
    """Check if column is near-constant. Returns (is_near_constant, reason)."""
    nunique = ser.nunique(dropna=True)
    if nunique <= NEAR_CONSTANT_NUNIQUE:
        return True, f"nunique={nunique}"
    counts = ser.value_counts(dropna=True)
    if len(counts) == 0:
        return False, ""
    top_pct = counts.iloc[0] / len(ser.dropna()) * 100
    if top_pct >= DOMINANT_CATEGORY_PCT:
        return True, f"dominant category {top_pct:.0f}%"
    return False, ""


def _values_look_like_ids(ser: pd.Series, nrows: int) -> tuple[bool, str]:
    """Check if numeric values look like identifiers. Returns (looks_like_id, reason)."""
    if not pd.api.types.is_numeric_dtype(ser):
        return False, ""
    valid = ser.dropna()
    if len(valid) < 3:
        return False, ""
    nunique = valid.nunique()
    ratio = nunique / max(nrows, 1)
    # Mostly unique
    if ratio > ID_LIKE_UNIQUE_RATIO:
        return True, f"nunique/nrows={ratio:.2f}"
    # Very large ints (typical of IDs)
    vals = valid.astype(np.float64)
    abs_vals = np.abs(vals)
    p99_abs = np.percentile(abs_vals, 99)
    if p99_abs > 1e12 and nunique > 0.5 * len(valid):
        return True, "very large values, mostly unique"
    # Low repeated values: each value appears 1-2 times
    counts = valid.value_counts()
    single_or_pair = (counts <= 2).sum() / max(len(counts), 1)
    if single_or_pair > 0.9 and nunique > 100:
        return True, "low repeat rate, high cardinality"
    return False, ""


def _is_id_like(col_name: str, ser: pd.Series, nrows: int) -> tuple[bool, str]:
    """Check if column looks like an identifier. Returns (is_id_like, reason)."""
    name_lower = col_name.lower()
    has_id_keyword = any(kw in name_lower for kw in ID_LIKE_KEYWORDS)
    nunique = ser.nunique(dropna=True)
    ratio = nunique / max(nrows, 1)
    # nunique/nrows > 0.85
    if ratio > ID_LIKE_UNIQUE_RATIO:
        return True, f"nunique≈nrows ({nunique}/{nrows})"
    # name contains id, uuid, guid, number, key, hash
    if has_id_keyword and ratio > 0.5:
        return True, f"name suggests id, nunique={nunique}"
    if has_id_keyword and nunique > 500:
        return True, f"name suggests id, high cardinality {nunique}"
    # values look like identifiers
    looks_like, val_reason = _values_look_like_ids(ser, nrows)
    if looks_like:
        return True, val_reason
    return False, ""


def _is_potential_leakage(col_name: str, target_col: Optional[str]) -> tuple[bool, str]:
    """Check if column name suggests target leakage. Returns (is_leakage, reason)."""
    if target_col and col_name == target_col:
        return False, ""
    name_lower = col_name.lower()
    matches = [kw for kw in LEAKAGE_KEYWORDS if kw in name_lower]
    if matches:
        return True, f"name contains: {', '.join(matches[:3])}"
    return False, ""


def audit_columns(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    """
    Generate audit report with flags: high_missing, extreme_outliers, near_constant,
    high_cardinality, suspicious_id_like, potential_leakage.
    Each flag has severity 0-3 and a short reason. Does not modify or drop any columns.
    """
    nrows = len(df)
    rows = []

    for col in df.columns:
        ser = df[col]
        missing_pct = ser.isna().mean() * 100
        nunique = ser.nunique(dropna=True)
        flags = []
        max_severity = 0
        reasons = []

        # high_missing
        if missing_pct > HIGH_MISSING_PCT:
            sev = 3 if missing_pct > 80 else (2 if missing_pct > 60 else 1)
            flags.append("high_missing")
            reasons.append(f"missing_pct={missing_pct:.0f}")
            max_severity = max(max_severity, sev)

        # extreme_outliers (numeric only)
        if pd.api.types.is_numeric_dtype(ser):
            count, max_mag = _robust_z_score(ser)
            if count > 0:
                sev = 2 if count > 10 else 1
                flags.append("extreme_outliers")
                reasons.append(f"count={count}, max_z={max_mag:.1f}")
                max_severity = max(max_severity, sev)

        # near_constant
        is_nc, nc_reason = _is_near_constant(ser)
        if is_nc:
            flags.append("near_constant")
            reasons.append(nc_reason)
            max_severity = max(max_severity, 1)

        # high_cardinality (categorical)
        if ser.dtype == "object" or str(ser.dtype) == "category":
            if nunique > HIGH_CARDINALITY_NUNIQUE:
                sev = 2 if nunique > 500 else 1
                flags.append("high_cardinality")
                reasons.append(f"nunique={nunique}")
                max_severity = max(max_severity, sev)

        # suspicious_id_like
        is_id, id_reason = _is_id_like(col, ser, nrows)
        if is_id:
            flags.append("suspicious_id_like")
            reasons.append(id_reason)
            max_severity = max(max_severity, 2)

        # potential_leakage
        is_leak, leak_reason = _is_potential_leakage(col, target_col)
        if is_leak:
            flags.append("potential_leakage")
            reasons.append(leak_reason)
            max_severity = max(max_severity, 3)

        reason_str = "; ".join(reasons) if reasons else ""
        rows.append({
            "column": col,
            "dtype": str(ser.dtype),
            "missing_pct": round(missing_pct, 1),
            "nunique": nunique,
            "flags": " | ".join(flags) if flags else "",
            "severity": max_severity,
            "reason": reason_str,
        })

    return pd.DataFrame(rows)


def _is_numeric_dtype(dtype_str: str) -> bool:
    """True if dtype string indicates numeric (int, float)."""
    d = str(dtype_str).lower()
    return "int" in d or "float" in d


def recommend_drops(
    audit_df: pd.DataFrame,
    target_col: str,
    drop_review_leakage: bool = False,
) -> dict[str, list[str]]:
    """
    Recommend columns to drop for modeling based on audit flags.
    definite_drop: suspicious_id_like (id-like columns - safe to auto-drop).
    review_drop: potential_leakage (do NOT auto-add to "all" unless drop_review_leakage).
    all: definite_drop + (review_drop if drop_review_leakage else []).
    safe_numeric_cols: numeric columns NOT suspicious_id_like, NOT potential_leakage, NOT target.
    unsafe_numeric_cols: numeric columns that ARE suspicious_id_like OR potential_leakage.
    Never drop the target column.
    Returns {"definite_drop", "review_drop", "all", "safe_numeric_cols", "unsafe_numeric_cols",
             "leakage", "id_like"}.
    """
    definite_drop: list[str] = []
    review_drop: list[str] = []
    unsafe_numeric_cols: list[str] = []
    safe_numeric_cols: list[str] = []
    for _, row in audit_df.iterrows():
        col = row["column"]
        if col == target_col:
            continue
        flags = (row.get("flags") or "").split(" | ")
        severity = int(row.get("severity", 0))
        is_numeric = _is_numeric_dtype(str(row.get("dtype", "")))
        is_id_like = "suspicious_id_like" in flags
        is_leakage = "potential_leakage" in flags
        if severity >= 2:
            if is_id_like:
                definite_drop.append(col)
            if is_leakage:
                review_drop.append(col)
        # For numeric classification: id-like or leakage = unsafe
        if is_numeric and (is_id_like or is_leakage):
            unsafe_numeric_cols.append(col)
        elif is_numeric:
            safe_numeric_cols.append(col)
    all_drops = list(dict.fromkeys(definite_drop + (review_drop if drop_review_leakage else [])))
    return {
        "definite_drop": definite_drop,
        "review_drop": review_drop,
        "all": all_drops,
        "leakage": review_drop,
        "id_like": definite_drop,
        "safe_numeric_cols": safe_numeric_cols,
        "unsafe_numeric_cols": unsafe_numeric_cols,
    }


def print_audit_summary(audit_df: pd.DataFrame) -> None:
    """Print compact table of flagged columns sorted by severity, plus summary counts."""
    flagged = audit_df[audit_df["flags"] != ""].copy()
    if flagged.empty:
        print("  No columns flagged.")
        return

    flagged = flagged.sort_values("severity", ascending=False)
    for _, row in flagged.iterrows():
        print(f"  [{row['severity']}] {row['column']}: {row['flags']} — {row['reason'][:60]}")
    counts = audit_df[audit_df["flags"] != ""]["flags"].str.split(r" \| ").explode().value_counts()
    print(f"\n  Flag counts: {counts.to_dict()}")
