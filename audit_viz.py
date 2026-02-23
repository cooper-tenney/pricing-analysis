"""
Audit visualization: matplotlib PDF report summarizing faulty columns.
Uses theme.py for blue-toned styling. GridSpec, thin accent bar, full-width pills.
"""

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from theme import (
    apply_mpl_theme,
    CARD_BORDER,
    CARD_FILL,
    FIG_SIZE,
    PRIMARY,
    SEVERITY as COLORS,
    TABLE_ALT_ROW,
    TABLE_HEADER,
)

# Local constants (detail page typography)
DPI = 150
TITLE = 16
H2 = 12
BODY = 10
SMALL = 9
DETAIL_TITLE = 15
DETAIL_H2 = 12
DETAIL_BODY = 10
DETAIL_SMALL = 9

# Pill backgrounds — light blue tones
PILL_COLORS = [
    "#e0f2fe", "#dbeafe", "#e8f0fe",
    "#eff6ff", "#f0f9ff", "#ecfeff",
]


def _shorten(s: str, max_len: int = 25) -> str:
    s = str(s).strip() if pd.notna(s) else ""
    if not s or str(s).lower() == "nan":
        return ""
    return textwrap.shorten(s, width=max_len, placeholder="...") if len(s) > max_len else s


def _wrap_text(s: str, width: int = 65, max_lines: int | None = 5) -> str:
    s = str(s).strip() if pd.notna(s) else ""
    if not s or str(s).lower() == "nan":
        return ""
    lines = textwrap.wrap(s, width=width)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = textwrap.shorten(lines[-1], width=width + 10, placeholder="...")
    return "\n".join(lines)


def _wrap_text_fill(s: str, width_chars: int = 50) -> list[str]:
    """Wrap text to list of lines. No truncation."""
    s = str(s).strip() if pd.notna(s) else ""
    if not s or str(s).lower() == "nan":
        return []
    return textwrap.wrap(s, width=width_chars)


def _parse_stats(stats_str: str) -> list[tuple[str, str]]:
    """Parse stats string into (label, value) pairs. Handles key=value and key=value; key=value."""
    s = str(stats_str).strip() if pd.notna(stats_str) else ""
    if not s or s.lower() == "nan":
        return []
    pairs = []
    for part in re.split(r";\s*", s):
        part = part.strip()
        for m in re.finditer(r"(\w+)=([-\d.eE+]+|[-\d]+\.?\d*|\S+)", part):
            label = m.group(1).replace("_", " ").title()
            pairs.append((label, str(m.group(2))))
        if not re.search(r"\w+=", part) and part:
            pairs.append(("Note", part))
    return pairs


# ---- Drawing helpers ----

def _draw_detail_header(ax, column: str, severity: int) -> None:
    """Thin accent bar + severity chip + column name. Compact ~64px feel."""
    ax.axis("off")
    color = COLORS.get(severity, COLORS[0])["main"]
    # Thin top accent bar (4% of axis)
    ax.add_patch(mpatches.Rectangle((0, 0.96), 1, 0.04, facecolor=color, edgecolor="none",
                                    transform=ax.transAxes))
    # Severity chip
    chip = mpatches.FancyBboxPatch((0.02, 0.5), 0.12, 0.36, boxstyle="round,pad=0.02,rounding_size=0.03",
                                   facecolor=color, edgecolor="none", transform=ax.transAxes)
    ax.add_patch(chip)
    ax.text(0.08, 0.68, f"Severity {severity}", fontsize=DETAIL_H2, color="white", fontweight="bold",
            transform=ax.transAxes, va="center", ha="center")
    # Column name right-aligned
    col_short = textwrap.shorten(column, width=50, placeholder="...")
    ax.text(0.98, 0.68, col_short, fontsize=DETAIL_TITLE, fontweight="bold", color="#212121",
            transform=ax.transAxes, va="center", ha="right")


def _draw_detail_pills(ax, pills: list[str]) -> None:
    """Full flag names, no truncation. Pills sized to text, wrap to next line."""
    ax.axis("off")
    if not pills:
        return
    pad_x, pad_y = 0.008, 0.008
    char_width = 0.012
    row_h = 0.28
    x0, y = 0.02, 0.88
    max_w = 0.96
    for i, p in enumerate(pills):
        p = str(p).strip()
        if not p:
            continue
        w = min(0.95, max(0.08, len(p) * char_width + 0.04))
        if x0 + w > max_w and x0 > 0.02:
            x0 = 0.02
            y -= row_h + pad_y
        color = PILL_COLORS[i % len(PILL_COLORS)]
        rect = mpatches.FancyBboxPatch((x0, y), w, row_h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                       facecolor=color, edgecolor="#bfdbfe", linewidth=0.3,
                                       transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y + row_h / 2, p, fontsize=DETAIL_SMALL, ha="center", va="center",
                transform=ax.transAxes)
        x0 += w + pad_x


def _draw_detail_card(ax, title: str, lines: list[str], fontsize: int = DETAIL_BODY) -> None:
    """Draw card with title and bullet lines. Lines are pre-wrapped."""
    ax.axis("off")
    pad = 0.04
    rect = mpatches.FancyBboxPatch((pad, pad), 1 - 2 * pad, 1 - 2 * pad,
                                   boxstyle="round,pad=0.01,rounding_size=0.02",
                                   facecolor=CARD_FILL, edgecolor=CARD_BORDER, linewidth=0.5,
                                   transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(pad + 0.02, 1 - pad - 0.04, title, fontsize=DETAIL_H2, fontweight="bold", color="#424242",
            transform=ax.transAxes, va="top")
    if not lines:
        return
    line_h = 0.065
    y = 1 - pad - 0.1
    for line in lines[:8]:
        ax.text(pad + 0.05, y, f"  {line}" if not line.startswith("-") else line,
                fontsize=fontsize, va="top", ha="left", transform=ax.transAxes,
                wrap=True)
        y -= line_h


def _draw_detail_stats_table(ax, pairs: list[tuple[str, str]], title: str = "Stats") -> None:
    """Stats table: title above, two-column key/value, zebra rows, right-aligned values."""
    ax.axis("off")
    if not pairs:
        return
    pairs = pairs[:8]
    # Title above table, no overlap
    ax.text(0.02, 0.96, title, fontsize=DETAIL_H2, fontweight="bold", color="#424242",
            transform=ax.transAxes, va="top")
    # Table starts below title
    table_top = 0.88
    n = len(pairs)
    row_h = (table_top - 0.04) / max(n, 1)
    for i, (label, value) in enumerate(pairs):
        yy = table_top - (i + 1) * row_h
        bg = TABLE_ALT_ROW if i % 2 else "white"
        ax.add_patch(mpatches.Rectangle((0.02, yy), 0.96, row_h - 0.01, facecolor=bg,
                                        edgecolor=CARD_BORDER, linewidth=0.3, transform=ax.transAxes))
        ax.text(0.05, yy + (row_h - 0.01) / 2, str(label), fontsize=DETAIL_SMALL, va="center", ha="left",
                transform=ax.transAxes, color="#616161")
        ax.text(0.95, yy + (row_h - 0.01) / 2, str(value), fontsize=DETAIL_SMALL, va="center", ha="right",
                transform=ax.transAxes, color="#212121")


def draw_card(ax, x: float, y: float, w: float, h: float, title: str) -> "tuple":
    """Legacy: draw rounded card for overview/table pages. Return inner content box."""
    pad = 0.02
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                   facecolor=CARD_FILL, edgecolor=CARD_BORDER, linewidth=0.5,
                                   transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x + pad, y + h - 0.03, title, fontsize=H2, fontweight="bold", color="#424242",
            transform=ax.transAxes, va="top")
    return (x + pad, y + pad, w - 2 * pad, h - 0.05)


def create_audit_report_pdf(
    ui_df: pd.DataFrame,
    output_path: str | Path = "artifacts/audit_report.pdf",
    plots_dir: str | Path = "artifacts/audit_plots",
    engineered_numeric_cols: list | None = None,
    skipped_numeric_cols: dict | None = None,
) -> Path:
    """
    Create multipage PDF with modern design.
    Page 1: Overview + Legend. Page 2: Compact table. Pages 3+: Detail per worst column.
    """
    apply_mpl_theme()
    output_path = Path(output_path)
    plots_dir = Path(plots_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    worst = ui_df.sort_values(
        ["severity", "missing_pct", "nunique"],
        ascending=[False, False, False],
    )
    detail_reason_col = "detail_reason" if "detail_reason" in ui_df.columns else "reason_pretty"
    detail_stats_col = "detail_stats" if "detail_stats" in ui_df.columns else "stats_pretty"
    key_issue_col = "key_issue" if "key_issue" in ui_df.columns else "reason_pretty"

    def _safe(s):
        out = str(s).strip() if pd.notna(s) else ""
        return "" if not out or str(out).lower() == "nan" else out

    with PdfPages(output_path) as pdf:
        # ---- Page 1: Overview + Legend ----
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
        fig.suptitle("Column Audit — Overview", fontsize=TITLE, fontweight="bold", y=0.98)

        total = len(ui_df)
        sev3 = (ui_df["severity"] == 3).sum()
        sev2 = (ui_df["severity"] == 2).sum()
        sev1 = (ui_df["severity"] == 1).sum()
        sev0 = (ui_df["severity"] == 0).sum()

        ax0 = axes[0]
        ax0.axis("off")
        ax0.text(0.05, 0.92, f"Total columns: {total}", fontsize=H2, transform=ax0.transAxes)
        sev_counts = {3: sev3, 2: sev2, 1: sev1, 0: sev0}
        for sev, label, y in [(3, "Critical (leakage, high missing)", 0.82), (2, "Warning (outliers, ID-like)", 0.72), (1, "Minor (near constant)", 0.62), (0, "Clean", 0.52)]:
            c = COLORS[sev]["main"]
            ax0.add_patch(mpatches.Circle((0.08, y + 0.02), 0.015, facecolor=c, transform=ax0.transAxes))
            ax0.text(0.12, y, f"Severity {sev}: {sev_counts[sev]}", fontsize=BODY, transform=ax0.transAxes)
            ax0.text(0.38, y, label, fontsize=SMALL, color="#616161", transform=ax0.transAxes)

        # Legend card
        legend_y = 0.38
        draw_card(ax0, 0.02, legend_y - 0.22, 0.96, 0.28, "Legend — Flag meanings")
        leg_text = (
            "potential_leakage: Name suggests target info (close, amount, arr, etc.)\n"
            "extreme_outliers: Robust z-score > 3.5\n"
            "negative_values: Numeric column has negative values\n"
            "suspicious_id_like: ID-like name or nunique ≈ nrows\n"
            "high_cardinality: >200 unique categorical values\n"
            "near_constant: nunique ≤ 2 or dominant category ≥95%"
        )
        ax0.text(0.05, legend_y - 0.02, leg_text, fontsize=SMALL, va="top", transform=ax0.transAxes,
                 fontfamily="monospace")

        ax1 = axes[1]
        flags_col = "flags_pretty" if "flags_pretty" in ui_df.columns else "flags"
        flag_series = ui_df[ui_df[flags_col].fillna("") != ""][flags_col]
        if not flag_series.empty:
            flag_counts = flag_series.str.split(r"\s*[•|]\s*").explode().str.strip().replace("", pd.NA).dropna().value_counts()
            if not flag_counts.empty:
                y_pos = range(len(flag_counts))
                ax1.barh(y_pos, flag_counts.values, alpha=0.85, color=PRIMARY)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels([_shorten(str(s), 28) for s in flag_counts.index], fontsize=BODY)
                ax1.set_xlabel("Count", fontsize=H2)
        ax1.set_title("Flag counts", fontsize=H2, fontweight="bold")
        ax1.grid(axis="x", alpha=0.2)
        ax1.tick_params(axis="both", labelsize=SMALL)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "01_overview.png", bbox_inches="tight", dpi=DPI)
        plt.close(fig)

        # ---- Page 2: Compact table ----
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.axis("off")
        tbl_data = worst.head(15).copy()
        flags_col = "flags_pretty" if "flags_pretty" in tbl_data.columns else "flags"
        cols_display = ["severity", "column", "dtype", "missing_pct", "nunique", flags_col, key_issue_col]
        cols_avail = [c for c in cols_display if c in tbl_data.columns]
        tbl_data = tbl_data[cols_avail].copy()
        tbl_data["column"] = tbl_data["column"].apply(lambda s: _shorten(_safe(s), 24))
        tbl_data[flags_col] = tbl_data[flags_col].fillna("").apply(lambda s: _shorten(_safe(s), 45))
        tbl_data[key_issue_col] = tbl_data[key_issue_col].fillna("").apply(lambda s: _shorten(_safe(s), 55))
        tbl_data["missing_pct"] = tbl_data["missing_pct"].fillna(0).round(1)
        tbl_data["nunique"] = tbl_data["nunique"].fillna(0).astype(int)
        tbl_data["severity"] = tbl_data["severity"].fillna(0).astype(int)

        cell_text = []
        for row in tbl_data.values.tolist():
            cell_text.append([_safe(c) for c in row])
        col_labels = ["sev", "column", "dtype", "miss%", "nunique", "flags", "key issue"][: len(cols_avail)]

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
            colColours=[TABLE_HEADER] * len(col_labels),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(BODY)
        table.scale(1.2, 2.0)
        for i in range(len(col_labels)):
            table[(0, i)].set_text_props(fontweight="bold")
        for r in range(1, len(cell_text) + 1):
            for c in range(len(col_labels)):
                cell = table[(r, c)]
                if r % 2 == 0:
                    cell.set_facecolor(TABLE_ALT_ROW)
                sev_idx = cols_avail.index("severity") if "severity" in cols_avail else -1
                if sev_idx >= 0 and c == sev_idx:
                    try:
                        sev = int(cell_text[r - 1][c])
                        cell.set_facecolor(COLORS.get(sev, COLORS[0])["main"])
                        cell.set_text_props(color="white", fontweight="bold")
                    except (ValueError, IndexError):
                        pass
        ax.set_title("Top 15 Worst Columns", fontsize=TITLE, fontweight="bold", pad=12)
        plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04)
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "02_table.png", bbox_inches="tight", dpi=DPI)
        plt.close(fig)

        # ---- Pages 3+: Detail per worst column (GridSpec compact layout) ----
        detail_rows = worst.head(10)
        for idx, (_, row) in enumerate(detail_rows.iterrows()):
            fig = plt.figure(figsize=FIG_SIZE)
            # height_ratios: header thin, flags compact, cards main, stats table
            gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[0.35, 0.5, 2.2, 1.5],
                                  hspace=0.08, left=0.06, right=0.94, top=0.96, bottom=0.04)

            sev = int(row.get("severity", 0))
            col_name = _safe(row.get("column", ""))
            ax_header = fig.add_subplot(gs[0])
            _draw_detail_header(ax_header, col_name, sev)

            flags_txt = _safe(row.get("flags_pretty", ""))
            pills = [p.strip() for p in re.split(r"\s*[•|]\s*", flags_txt) if p.strip()]
            ax_flags = fig.add_subplot(gs[1])
            _draw_detail_pills(ax_flags, pills)

            # Two cards side by side
            gs_cards = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.06)
            ax_left = fig.add_subplot(gs_cards[0])
            ax_right = fig.add_subplot(gs_cards[1])

            # Summary card: 2-3 key bullets
            mpct = float(row.get("missing_pct", 0) or 0)
            nu = int(row.get("nunique", 0) or 0)
            pdict = dict(_parse_stats(row.get(detail_stats_col, "")))
            summary_lines = [f"- Missing: {mpct:.1f}%", f"- Nunique: {nu}"]
            if pdict.get("Count Neg"):
                summary_lines.append(f"- Negatives: {pdict.get('Count Neg','')} ({pdict.get('Pct Neg','')}%), min={pdict.get('Min','')}")
            if pdict.get("P99") or pdict.get("Max"):
                summary_lines.append(f"- Outliers: p99={pdict.get('P99','—')}, max={pdict.get('Max','—')}")
            _draw_detail_card(ax_left, "Summary", summary_lines[:6])

            # Why flagged: short wrapped sentences, no semicolon soup
            reason_txt = _safe(row.get(detail_reason_col, ""))
            reason_clean = reason_txt.replace("; ", ". ").replace(";", ".")
            why_lines = _wrap_text_fill(reason_clean, 48)
            _draw_detail_card(ax_right, "Why flagged", why_lines[:6] if why_lines else ["—"])

            # Stats table: directly below cards
            stats_pairs = _parse_stats(row.get(detail_stats_col, ""))
            ax_stats = fig.add_subplot(gs[3])
            _draw_detail_stats_table(ax_stats, stats_pairs)

            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(plots_dir / f"03_detail_{idx + 1:02d}.png", bbox_inches="tight", dpi=DPI)
            plt.close(fig)

        # ---- Numeric engineering page ----
        if engineered_numeric_cols is not None or skipped_numeric_cols:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
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
            ax.text(0.05, 0.95, body, transform=ax.transAxes, fontsize=BODY,
                    verticalalignment="top", fontfamily="monospace")
            ax.set_title("Numeric Feature Engineering", fontsize=TITLE, fontweight="bold")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(plots_dir / "07_numeric_engineering.png", bbox_inches="tight", dpi=DPI)
            plt.close(fig)

    return output_path
