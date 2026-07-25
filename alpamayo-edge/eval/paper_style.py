#!/usr/bin/env python3
"""Shared figure style for all Alpamayo-Edge figures (CVPR-style typography).

Every figure in docs/figures/ imports this so fonts, sizes, spines, grid and
legends are identical across the whole document set.

Conventions (conference-paper style):
  * Serif type throughout: Times New Roman (Latin) + Songti SC (CJK fallback).
  * One type scale — no per-figure ad-hoc font sizes (see the SIZES block).
  * Titles are short and neutral (quantity + measurement conditions); results
    and interpretation belong in the figure caption, not inside the image.
  * Top/right spines removed, light dotted y-grid only, frameless legend.
  * 300 dpi output (print-grade).

Import before creating any figure:
    from paper_style import apply, C4, C8, CF, ANNOT, bar_kw, WIDTH_HALF, ...
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# ---- type scale (points) --------------------------------------------------
BASE = 9.0       # body / tick text
TITLE = 9.5      # axes + figure title
LABEL = 9.0      # axis labels
TICK = 8.0       # tick labels
LEGEND = 8.0     # legend entries
ANNOT = 7.5      # in-plot data labels (bar values, baseline callouts)

# ---- palette (kept from the site's brand colors) -------------------------
C4, C8, CF = "#2563eb", "#f59e0b", "#16a34a"   # INT4 blue, INT8 amber, FP16 green
RULE = "#e74c3c"                                # memory-limit rule
INK = "#1a1a2e"                                 # text / edges

# ---- canvas widths -------------------------------------------------------
# The site renders half-slot figures at ~440 px and full-width ones at ~870 px.
# Keeping width proportional to the display slot makes the *on-screen* type
# size identical across every figure (~84 px/inch in both cases).
WIDTH_HALF = 5.2
WIDTH_FULL = 10.4


def _serif_stack():
    """Latin serif first, CJK serif after it as the fallback face.

    Must be assigned to ``font.family`` as an explicit list: matplotlib only
    walks a *family list* when a glyph is missing. Pointing ``font.family`` at
    "serif" and listing the faces in ``font.serif`` resolves to the first
    available face only, which renders CJK as tofu boxes.
    """
    have = {f.name for f in _fm.fontManager.ttflist}
    latin = [n for n in ("Times New Roman", "STIX Two Text", "DejaVu Serif") if n in have]
    cjk = [n for n in ("Songti SC", "STSong", "Arial Unicode MS") if n in have]
    return (latin + cjk) or ["DejaVu Serif"]


def apply():
    """Install the shared rcParams. Call once, before building figures."""
    plt.rcParams.update({
        # type — explicit family list so CJK falls back to a serif CJK face
        "font.family": _serif_stack(),
        "font.serif": _serif_stack(),
        "mathtext.fontset": "stix",
        "font.size": BASE,
        "axes.titlesize": TITLE,
        "axes.titleweight": "normal",   # papers don't bold figure titles
        "axes.labelsize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        "figure.titlesize": TITLE,
        "axes.unicode_minus": False,
        # frame
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # grid: light dotted, y only, behind the data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#b0b0b0",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.3,
        # output
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def bar_kw():
    """Thin dark bar outline — keeps bars legible in grayscale print."""
    return dict(edgecolor="#333333", linewidth=0.5)


def fig_legend(fig, ax, ncol=3):
    """One shared frameless legend centred below the figure.

    Panels in these figures always plot the same series, so a single figure
    legend (rather than one per axes) avoids colliding with the y-axis ticks
    and keeps every figure's legend in the same place.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), frameon=False)


def label_bars(ax, bars, fmt="{:.1f}", pad=2.0):
    """Uniform in-plot data labels above bars."""
    for r in bars:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=ANNOT,
                    xytext=(0, pad), textcoords="offset points")
