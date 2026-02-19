"""
Visualization: matplotlib PDF report for top 5 features vs target.
Uses target in original dollars. No plotly dependency.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _raw_column_for_plot(col: str, available: list[str]) -> str:
    """Prefer raw column over engineered variant for plotting. Returns col or base if base exists."""
    for suffix in ("_log1p", "_sqrt", "_winsor"):
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            if base in available:
                return base
    return col


def _shorten_label(s: str, max_len: int = 18) -> str:
    """Shorten label with ellipsis if needed."""
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _plot_numeric(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    col: str,
    y_col: str,
    importance: float,
    n_bins: int = 30,
    hexbin_threshold: int = 5000,
) -> None:
    """Scatter or hexbin (if >hexbin_threshold points) vs target with binned median. Clips y at p99 for display."""
    valid = df_plot[[col, y_col]].dropna()
    x = valid[col].values.astype(float)
    y = valid[y_col].values.astype(float)
    if len(x) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        return

    upper = np.percentile(y, 99)
    y_plot = np.clip(y, None, upper)

    if len(x) > hexbin_threshold:
        hb = ax.hexbin(x, y_plot, gridsize=40, mincnt=1, cmap="Blues", alpha=0.8)
        plt.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(x, y_plot, alpha=0.12, s=6)
    ax.set_ylim(0, upper)
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Target ($)", fontsize=9)
    ax.annotate("(clipped at p99 for display)", xy=(0.02, 0.98), xycoords="axes fraction", fontsize=7, va="top")

    # Binned median line on clipped y
    df_temp = pd.DataFrame({"x": x, "y": y_plot})
    df_temp["bin"] = pd.qcut(df_temp["x"], q=min(n_bins, len(df_temp) // 5 or 5), duplicates="drop")
    medians = df_temp.groupby("bin", observed=True).agg({"x": "median", "y": "median"}).reset_index()
    medians = medians.sort_values("x")
    ax.plot(medians["x"], medians["y"], color="red", linewidth=2, label="Binned median")

    ax.set_title(f"{col} vs Target (importance: {importance:.3f})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.2)


def _plot_text_length(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    col: str,
    y_col: str,
    importance: float,
    n_bins: int = 30,
    hexbin_threshold: int = 5000,
) -> None:
    """String length vs target: hexbin if >hexbin_threshold points, else scatter. y clipped at p99."""
    valid = df_plot[[col, y_col]].dropna()
    valid = valid.copy()
    valid["_len"] = valid[col].astype(str).str.len()
    x = valid["_len"].values.astype(float)
    y = valid[y_col].values.astype(float)
    if len(x) < 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        return

    upper = np.percentile(y, 99)
    y_plot = np.clip(y, None, upper)

    if len(x) > hexbin_threshold:
        hb = ax.hexbin(x, y_plot, gridsize=40, mincnt=1, cmap="Blues", alpha=0.8)
        plt.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(x, y_plot, alpha=0.12, s=6)
    ax.set_ylim(0, upper)
    ax.annotate("(clipped at p99 for display)", xy=(0.02, 0.98), xycoords="axes fraction", fontsize=7, va="top")

    df_temp = pd.DataFrame({"x": x, "y": y_plot})
    df_temp["bin"] = pd.qcut(df_temp["x"], q=min(n_bins, len(df_temp) // 5 or 5), duplicates="drop")
    medians = df_temp.groupby("bin", observed=True).agg({"x": "median", "y": "median"}).reset_index()
    medians = medians.sort_values("x")
    ax.plot(medians["x"], medians["y"], color="red", linewidth=2, label="Binned median")

    ax.set_xlabel(f"{col} (string length)", fontsize=9)
    ax.set_ylabel("Target ($)", fontsize=9)
    ax.set_title(f"{col} vs Target (importance: {importance:.3f})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.2)


def _plot_categorical(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    col: str,
    y_col: str,
    importance: float,
    top_n: int = 12,
    max_label_len: int = 18,
) -> None:
    """Bar chart: median target by top N categories (by frequency). Cap to top 12, rotate labels."""
    # If object dtype and (median string length >= 30 or nunique > 200), use text length plot instead
    if pd.api.types.is_object_dtype(df_plot[col]):
        ser = df_plot[col].dropna().astype(str)
        if not ser.empty:
            median_len = ser.str.len().median()
            nunique = ser.nunique()
            if median_len >= 30 or nunique > 200:
                _plot_text_length(ax, df_plot, col, y_col, importance, n_bins=30)
                return

    valid = df_plot[[col, y_col]].dropna()
    counts = valid[col].astype(str).value_counts()
    top_cats = counts.head(top_n).index.tolist()
    valid = valid.copy()
    valid["_cat"] = valid[col].astype(str).where(valid[col].astype(str).isin(top_cats), "Other")
    cat_order = [c for c in top_cats] + (["Other"] if "Other" in valid["_cat"].values else [])
    medians = valid.groupby("_cat", observed=True)[y_col].median()
    medians = medians.reindex([c for c in cat_order if c in medians.index])
    xs = range(len(medians))
    ax.bar(xs, medians.values, alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([_shorten_label(s, max_label_len) for s in medians.index], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median Target ($)", fontsize=9)
    ax.set_title(f"{col} vs Target (importance: {importance:.3f})", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.2)


def create_report_pdf(
    X_raw: pd.DataFrame,
    y_dollars: np.ndarray,
    top_features: list[str],
    col_importance: pd.DataFrame,
    output_path: str | Path = "artifacts/report.pdf",
    plots_dir: str | Path = "artifacts/plots",
) -> Path:
    """
    Create multipage PDF with one plot per top feature.
    col_importance: DataFrame with 'column' and 'importance' (e.g. permutation importance).
    - Numeric: scatter with binned median (y clipped at 99th percentile)
    - Object with median len>=30 or nunique>200: string length vs target
    - Else: categorical bar chart by frequency, top 15
    Also save individual PNGs to plots_dir.
    """
    output_path = Path(output_path)
    plots_dir = Path(plots_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_plot = X_raw.copy()
    df_plot["_target"] = y_dollars

    imp_map = dict(zip(col_importance["column"], col_importance["importance"]))

    with PdfPages(output_path) as pdf:
        for i, col in enumerate(top_features[:5]):
            if col not in df_plot.columns:
                continue
            imp = imp_map.get(col, 0.0)
            # Prefer raw column for plotting (not engineered variants)
            plot_col = _raw_column_for_plot(col, list(df_plot.columns))
            if plot_col not in df_plot.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            if pd.api.types.is_numeric_dtype(df_plot[plot_col]) and df_plot[plot_col].nunique() > 10:
                _plot_numeric(ax, df_plot, plot_col, "_target", imp)
            else:
                _plot_categorical(ax, df_plot, plot_col, "_target", imp, top_n=12)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            png_path = plots_dir / f"plot_{i + 1}_{col.replace('/', '_')[:30]}.png"
            fig.savefig(png_path, bbox_inches="tight", dpi=100)
            plt.close(fig)

    return output_path


def create_residuals_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    output_path: str | Path = "artifacts/residuals_report.pdf",
    plots_dir: str | Path = "artifacts/residuals",
) -> Path:
    """
    Create residuals_report.pdf with:
    (1) y_true vs y_pred scatter + y=x
    (2) residual histogram
    (3) residual vs y_true
    (4) top 20 abs residual rows table
    Also save individual PNGs to plots_dir.
    """
    output_path = Path(output_path)
    plots_dir = Path(plots_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    residual = y_true - y_pred

    with PdfPages(output_path) as pdf:
        # 1. y_true vs y_pred scatter + y=x
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.4, s=10)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="y=x")
        ax.set_xlabel("y_true ($)", fontsize=9)
        ax.set_ylabel("y_pred ($)", fontsize=9)
        ax.set_title("y_true vs y_pred")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "1_y_true_vs_y_pred.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # 2. residual histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residual, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual ($)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Residual Histogram")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "2_residual_histogram.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # 3. residual vs y_true
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_true, residual, alpha=0.4, s=10)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("y_true ($)", fontsize=9)
        ax.set_ylabel("Residual ($)", fontsize=9)
        ax.set_title("Residual vs y_true")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "3_residual_vs_y_true.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

        # 4. top 20 abs residual rows table (include key cols from X_test if present)
        tbl_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residual,
        })
        key_hints = ("product", "opportunity", "name", "type", "region", "customer")
        if isinstance(X_test, pd.DataFrame) and len(X_test) == len(tbl_df):
            for col in X_test.columns:
                c_lower = col.lower()
                if any(h in c_lower for h in key_hints) and X_test[col].dtype in ("object", "category", "bool"):
                    if col not in tbl_df.columns and len(tbl_df.columns) < 6:
                        tbl_df[col] = X_test[col].values
        tbl_df["abs_residual"] = np.abs(residual)
        display_cols = [c for c in tbl_df.columns if c != "abs_residual"][:6]
        top20 = tbl_df.nlargest(20, "abs_residual")[display_cols].copy()
        for c in top20.columns:
            if pd.api.types.is_numeric_dtype(top20[c]):
                top20[c] = top20[c].round(2)
        top20.index = range(1, len(top20) + 1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        table = ax.table(
            cellText=top20.values,
            rowLabels=top20.index,
            colLabels=top20.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        ax.set_title("Top 20 Absolute Residual Rows", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(plots_dir / "4_top20_residuals_table.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

    return output_path
