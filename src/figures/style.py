"""Shared look for every figure: soft off-white panel, light grid on both axes,
no box, thick lines with dots, shaded 95% bands and labels at the line ends.

Usage:  import style; style.apply()
        style.line_with_band(ax, x, y, lo, hi, colour, "label")
"""
import matplotlib.pyplot as plt

SURFACE = "#fbfaf8"
GRID = "#e6e5e1"
INK = "#2b2b29"
INK2 = "#5f5e5a"
MUTED = "#8a8985"
Q3_COL = "#2a78d6"   # blue
Q4_COL = "#e2703a"   # orange
THIRD_COL = "#4fae84"  # green
INTACT_COL = "#8a8985"


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "lines.linewidth": 2.4,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def line_with_band(ax, x, y, lo, hi, colour, label=None, end_label=True, dy=0.0):
    """Thick line with dots, a shaded band for the interval, and the series
    name written at the right-hand end of the line instead of in a legend box."""
    ax.fill_between(x, lo, hi, color=colour, alpha=0.14, linewidth=0)
    ax.plot(x, y, "-o", color=colour, label=label, zorder=3)
    if label and end_label:
        ax.annotate(label, (x[-1], y[-1]), xytext=(12, dy), textcoords="offset points",
                    color=colour, va="center", ha="left", fontsize=10.5, annotation_clip=False)


def reference_line(ax, y, label, x_end):
    ax.axhline(y, color=MUTED, lw=1.1, ls=":", zorder=1)
    ax.annotate(label, (x_end, y), xytext=(12, 0), textcoords="offset points",
                color=MUTED, va="center", ha="left", fontsize=10, annotation_clip=False)
