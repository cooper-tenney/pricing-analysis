"""
Audit visualization: matplotlib PDF report summarizing faulty columns.
Consistent style with the feature report.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _shorten(s: str, max_len: int = 25) -> str:
    s = str(s).strip()
    return s[: max_len - 3] + "..." if len(s) > max_len else s


def create_audit_report_pdf(
    ui_df: pd.DataFrame,
    output_path: str | Path = "artifacts/audit_report.pdf",
    plots_dir: str | Path = "artifacts/audit_plots",
    engineered_numeric_cols: list | None = None,
    skipped_numeric_cols: dict | None = None,
) -> Path:
    """
    Create multipage PDF summarizing audit results.
    Pages: Overview, Top 15 table, Missingness, High cardinality, Severity,
    Numeric engineering (if engineered_numeric_cols/skipped_numeric_cols provided).
    """
    output_path = Path(output_path)
    plots_dir = Path(plots_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    worst = ui_df.sort_values(
        ["severity", "missing_pct", "nunique"],
        ascending=[False, False, False],
    )

    with PdfPages(output_path) as pdf:
        # Page 1: Overview
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Audit Overview", fontsize=12, fontweight="bold")

        total = len(ui_df)
        sev3 = (ui_df["severity"] == 3).sum()
        sev2 = (ui_df["severity"] == 2).sum()
        sev1 = (ui_df["severity"] == 1).sum()
        sev0 = (ui_df["severity"] == 0).sum()

        ax0 = axes[0]
        ax0.axis("off")
        ax0.text(0.1, 0.9, f"Total columns: {total}", fontsize=10, transform=ax0.transAxes)
        ax0.text(0.1, 0.75, f"Severity 3: {sev3}", fontsize=10, transform=ax0.transAxes)
        ax0.text(0.1, 0.6, f"Severity 2: {sev2}", fontsize=10, transform=ax0.transAxes)
        ax0.text(0.1, 0.45, f"Severity 1: {sev1}", fontsize=10, transform=ax0.transAxes)
        ax0.text(0.1, 0.3, f"Severity 0: {sev0}", fontsize=10, transform=ax0.transAxes)

        ax1 = axes[1]
        flag_counts = ui_df[ui_df["flags"] != ""]["flags"].str.split(r" \| ").explode().value_counts()
        if not flag_counts.empty:
            bars = ax1.barh(range(len(flag_counts)), flag_counts.values, alpha=0.8)
            ax1.set_yticks(range(len(flag_counts)))
            ax1.set_yticklabels([_shorten(str(s), 30) for s in flag_counts.index], fontsize=8)
            ax1.set_xlabel("Count")
            ax1.set_title("Flag counts")
        else:
            ax1.text(0.5, 0.5, "No flags", ha="center", va="center", transform=ax1.transAxes)
        ax1.grid(axis="x", alpha=0.2)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "page1_overview.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Page 2: Top 15 worst columns table
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        tbl_data = worst.head(15)
        cols_display = ["column", "dtype", "missing_pct", "nunique", "severity", "flags", "reasons", "description"]
        cols_avail = [c for c in cols_display if c in tbl_data.columns]
        tbl_data = tbl_data[cols_avail].copy()
        tbl_data["column"] = tbl_data["column"].apply(lambda s: _shorten(str(s), 22))
        tbl_data["flags"] = tbl_data["flags"].apply(lambda s: _shorten(str(s), 28))
        tbl_data["reasons"] = tbl_data["reasons"].apply(lambda s: _shorten(str(s), 35))
        if "description" in tbl_data.columns:
            tbl_data["description"] = tbl_data["description"].fillna("").apply(lambda s: _shorten(str(s), 60))
        tbl_data["missing_pct"] = tbl_data["missing_pct"].fillna(0).round(1)
        tbl_data["nunique"] = tbl_data["nunique"].fillna(0).astype(int)

        table = ax.table(
            cellText=tbl_data.values,
            colLabels=tbl_data.columns,
            loc="center",
            cellLoc="left",
            colColours=["#e0e0e0"] * len(tbl_data.columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(range(len(tbl_data.columns)))
        ax.set_title("Top 15 worst columns (by severity, missing_pct, nunique)", fontsize=10, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "page2_table.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Page 3: Missingness
        fig, ax = plt.subplots(figsize=(8, 5))
        miss = ui_df.nlargest(15, "missing_pct")
        miss = miss[miss["missing_pct"] > 0]
        if not miss.empty:
            y_pos = range(len(miss))
            ax.barh(y_pos, miss["missing_pct"].values, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_shorten(str(s), 28) for s in miss["column"]], fontsize=8)
            ax.set_xlabel("Missing %")
            ax.set_title("Top 15 columns by missing_pct")
        else:
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "page3_missingness.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Page 4: High cardinality
        fig, ax = plt.subplots(figsize=(8, 5))
        card = ui_df.nlargest(15, "nunique")
        card = card[card["nunique"] > 0]
        if not card.empty:
            y_pos = range(len(card))
            ax.barh(y_pos, card["nunique"].values.astype(int), alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_shorten(str(s), 28) for s in card["column"]], fontsize=8)
            ax.set_xlabel("Unique count")
            ax.set_title("Top 15 columns by nunique")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "page4_cardinality.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Page 5: Severity (top 15 worst)
        fig, ax = plt.subplots(figsize=(8, 5))
        sev_tbl = worst.head(15)
        if not sev_tbl.empty:
            y_pos = range(len(sev_tbl))
            ax.barh(y_pos, sev_tbl["severity"].values, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_shorten(str(s), 28) for s in sev_tbl["column"]], fontsize=8)
            ax.set_xlabel("Severity (0-3)")
            ax.set_xlim(0, 3.5)
            ax.set_title("Top 15 worst columns by severity")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "page5_severity.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Page 6: Descriptions (Full) - top 30 columns by severity, missing_pct
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis("off")
        desc_tbl = worst.head(30)
        if "description" in desc_tbl.columns and not desc_tbl.empty:
            desc_tbl = desc_tbl[["column", "description"]].copy()
            desc_tbl["description"] = desc_tbl["description"].fillna("")
            lines = ["Descriptions (Full)", "=" * 50, ""]
            for _, row in desc_tbl.iterrows():
                col_name = str(row["column"])
                desc = str(row["description"]).strip()
                lines.append(col_name)
                for para in textwrap.wrap(desc, width=70):
                    lines.append("  " + para)
                lines.append("")
            body = "\n".join(lines)
            ax.text(0.02, 0.98, body, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", fontfamily="monospace")
        else:
            ax.text(0.5, 0.5, "No descriptions available.", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "page6_descriptions.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # Page 7: Numeric engineering (if provided)
        if engineered_numeric_cols is not None or skipped_numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axis("off")
            lines = ["Numeric Feature Engineering", "=" * 50, ""]
            eng = engineered_numeric_cols or []
            skp = skipped_numeric_cols or {}
            if eng:
                lines.append("Applied to:")
                for c in eng:
                    lines.append(f"  • {c}")
                lines.append("")
            if skp:
                lines.append("Skipped (with reasons):")
                for col, reason in sorted(skp.items(), key=lambda x: x[0]):
                    lines.append(f"  • {col}: {reason}")
            else:
                lines.append("Skipped: (none)")
            body = "\n".join(lines)
            ax.text(0.02, 0.98, body, transform=ax.transAxes, fontsize=9,
                    verticalalignment="top", fontfamily="monospace")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(plots_dir / "page7_numeric_engineering.png", bbox_inches="tight", dpi=100)
            plt.close(fig)

    return output_path
