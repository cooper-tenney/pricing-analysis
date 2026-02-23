"""
Central theme for matplotlib PDFs (viz.py, audit_viz.py).
Blue-toned, modern, minimal borders. Single source for colors and typography.
"""

# ---- Blue palette ----
PRIMARY = "#1a56db"       # deep blue accent (header line, trend line, chips)
PRIMARY_LIGHT = "#e8f0fe" # light blue background for info states
SECONDARY = "#64748b"     # muted slate for secondary text
NEUTRAL_50 = "#f8fafc"
NEUTRAL_100 = "#f1f5f9"
NEUTRAL_200 = "#e2e8f0"
NEUTRAL_400 = "#94a3b8"
NEUTRAL_600 = "#475569"
NEUTRAL_800 = "#1e293b"

# Muted severity (consistent with blue theme)
SEVERITY = {
    3: {"main": "#b91c1c", "light": "#fef2f2"},   # muted red
    2: {"main": "#c2410c", "light": "#fff7ed"},   # muted orange
    1: {"main": "#b45309", "light": "#fffbeb"},   # muted amber
    0: {"main": "#64748b", "light": "#f8fafc"},   # gray
}

# Card/chip styling
CARD_FILL = "#f8fafc"
CARD_BORDER = "#e2e8f0"
CARD_BORDER_WIDTH = 0.5
CHIP_BG = "#e8f0fe"
CHIP_BORDER = "#bfdbfe"
TABLE_ALT_ROW = "#f1f5f9"
TABLE_HEADER = "#f1f5f9"

# Font sizes (pt)
FONT_TITLE = 16
FONT_H2 = 12
FONT_BODY = 10
FONT_SMALL = 9

# Layout
FIG_SIZE = (11, 8.5)
DPI = 150
RADIUS_PX = 10  # rounded corners ~10-14px
SECTION_GAP = 0.04  # axes fraction between sections

# Grid
GRID_ALPHA = 0.15
SPINE_COLOR = "none"  # hide top/right spines


def apply_mpl_theme():
    """Set matplotlib rcParams for consistent PDF styling."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": FONT_BODY,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_H2,
        "xtick.labelsize": FONT_SMALL,
        "ytick.labelsize": FONT_SMALL,
        "axes.edgecolor": NEUTRAL_200,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def style_axes(ax):
    """Remove top/right spines, light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=GRID_ALPHA, axis="y")
