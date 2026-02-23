"""
Streamlit dashboard for Column Audit Summary.
Loads artifacts/runs/{run_id}/ui_summary.csv (latest run).
Clean table + expandable detail cards for worst columns.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ARTIFACTS = Path("artifacts")
RUNS = ARTIFACTS / "runs"

FLAG_FILTERS = {
    "potential_leakage": "Potential Leakage",
    "extreme_outliers": "Extreme Outliers",
    "negative_values": "Negative Values",
    "suspicious_id_like": "Suspicious Id Like",
    "high_cardinality": "High Cardinality",
    "near_constant": "Near Constant",
}


def _find_run_dir() -> Path | None:
    """Return path to run dir: latest from artifacts/latest.txt or most recent run."""
    latest_file = ARTIFACTS / "latest.txt"
    if latest_file.exists():
        run_id = latest_file.read_text(encoding="utf-8").strip()
        run_dir = RUNS / run_id
        if run_dir.exists() and (run_dir / "ui_summary.csv").exists():
            return run_dir
    if not RUNS.exists():
        return None
    candidates = []
    for d in RUNS.iterdir():
        if d.is_dir() and (d / "ui_summary.csv").exists():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def _safe(s) -> str:
    out = str(s).strip() if pd.notna(s) else ""
    return "" if not out or str(out).lower() == "nan" else out


STREAMLIT_CSS = """
<style>
:root { --primary: #1a56db; --primary-light: #e8f0fe; --chip-bg: #e8f0fe; --chip-border: #bfdbfe; }
.stMetric { background: #f8fafc; border-radius: 10px; padding: 0.5rem; }
.stDataFrame { border-radius: 10px; }
.expander { border: 1px solid #e2e8f0; border-radius: 10px; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Column Audit", layout="wide")
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.title("Column Audit Summary")

    run_dir = _find_run_dir()
    if not run_dir:
        st.error("No audit run found. Run: python main.py <data.csv>")
        return

    csv_path = run_dir / "ui_summary.csv"
    df = pd.read_csv(csv_path)
    # Backfill for older schema
    if "flags_pretty" not in df.columns and "flags" in df.columns:
        df["flags_pretty"] = df["flags"].fillna("").apply(lambda s: str(s).replace("_", " ").title())
    if "key_issue" not in df.columns and "reasons" in df.columns:
        df["key_issue"] = df["reasons"].fillna("").apply(lambda s: _safe(s)[:90])
    if "detail_reason" not in df.columns and "reasons" in df.columns:
        df["detail_reason"] = df["reasons"].fillna("").apply(_safe)
    if "detail_stats" not in df.columns and "stats" in df.columns:
        df["detail_stats"] = df["stats"].fillna("").apply(_safe)
    # Normalize NaNs to empty strings
    for c in ["flags_pretty", "key_issue", "detail_reason", "detail_stats", "reason_pretty", "stats_pretty", "reasons"]:
        if c in df.columns:
            df[c] = df[c].fillna("").apply(_safe)
    df["severity"] = df["severity"].fillna(0).astype(int)
    df["missing_pct"] = df["missing_pct"].fillna(0)
    df["nunique"] = df["nunique"].fillna(0).astype(int)

    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        severity_min, severity_max = st.slider("Severity", 0, 3, (0, 3), 1)
    with col2:
        st.markdown("**Flag types**")
        flag_checkboxes = {}
        for key, label in FLAG_FILTERS.items():
            flag_checkboxes[key] = st.checkbox(label, value=True, key=f"flag_{key}")
    with col3:
        search = st.text_input("Search column name", placeholder="Filter by column...")

    # Build display dataframe
    display_cols = ["severity", "column", "dtype", "missing_pct", "nunique", "flags_pretty", "key_issue"]
    display_cols = [c for c in display_cols if c in df.columns]
    ui_display = df[display_cols].copy()

    # Apply filters
    filtered = ui_display[
        (ui_display["severity"] >= severity_min) &
        (ui_display["severity"] <= severity_max)
    ]
    if search:
        search_lower = search.lower()
        filtered = filtered[filtered["column"].fillna("").str.lower().str.contains(search_lower, na=False)]
    for key, checked in flag_checkboxes.items():
        if not checked:
            substring = FLAG_FILTERS[key]
            filtered = filtered[~filtered["flags_pretty"].fillna("").str.contains(substring, case=False, na=False)]

    # Sort
    filtered = filtered.sort_values(
        ["severity", "missing_pct", "nunique"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    st.subheader("Audit Table")
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "severity": st.column_config.NumberColumn(
                "Severity",
                format="%d",
                help="0=clean, 1=minor, 2=warning, 3=critical",
            ),
            "missing_pct": st.column_config.NumberColumn("Miss %", format="%.1f%%"),
            "nunique": st.column_config.NumberColumn("Nunique", format="%d"),
            "key_issue": st.column_config.TextColumn("Key Issue", width="large"),
            "flags_pretty": st.column_config.TextColumn("Flags", width="medium"),
        },
    )

    st.subheader("Details (worst columns)")
    worst_n = 10
    worst = filtered.head(worst_n)
    badges = {3: "🔴", 2: "🟠", 1: "🟡", 0: "⚪"}

    for idx, row in worst.iterrows():
        col_name = _safe(row.get("column", ""))
        sev = int(row.get("severity", 0))
        badge = badges.get(sev, "⚪")
        full_row = df[df["column"] == col_name].iloc[0] if col_name in df["column"].values else row
        with st.expander(f"{badge} [{sev}] {col_name}"):
            flags = _safe(full_row.get("flags_pretty", ""))
            key_issue = _safe(full_row.get("key_issue", ""))
            detail_reason = _safe(full_row.get("detail_reason", full_row.get("reason_pretty", "")))
            detail_stats = _safe(full_row.get("detail_stats", full_row.get("stats_pretty", "")))
            if flags:
                st.markdown("**Flags:**")
                for f in flags.split(" • "):
                    if f.strip():
                        st.markdown(f"- {f.strip()}")
            st.markdown(f"**Key issue:** {key_issue or '—'}")
            if detail_reason:
                st.markdown("**Why flagged:**")
                st.markdown(detail_reason)
            if detail_stats:
                st.markdown("**Stats:**")
                st.code(detail_stats, language=None)
    st.caption(f"Run: {run_dir.name}")


if __name__ == "__main__":
    main()
