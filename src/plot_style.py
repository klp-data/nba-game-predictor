"""Publication-ready matplotlib style.

Benutzung:
    from src.plot_style import apply
    apply()

Optional: `apply(use_tex=True)` aktiviert echtes LaTeX-Rendering, sofern eine
LaTeX-Installation auf dem System vorhanden ist (sonst Fallback auf Mathtext).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    'primary':   '#2A5C8B',  # gedaempftes Blau
    'secondary': '#C45A5A',  # gedaempftes Rot
    'accent':    '#D08A2C',  # warmes Orange
    'success':   '#4A8B5C',  # gedaempftes Gruen
    'neutral':   '#6B6B6B',
    'highlight': '#8B2A5C',
}

PALETTE = ['#2A5C8B', '#C45A5A', '#D08A2C', '#4A8B5C', '#6B4E8B', '#8B6B2A']


def apply(use_tex: bool = False) -> None:
    """Applies a clean, publication-ready style."""
    plt.rcdefaults()
    mpl.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size':         11,
        'axes.titlesize':    12,
        'axes.labelsize':    11,
        'axes.titleweight':  'normal',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linestyle':    '--',
        'grid.linewidth':    0.6,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'legend.frameon':    False,
        'legend.fontsize':   10,
        'figure.dpi':        110,
        'savefig.dpi':       180,
        'savefig.bbox':      'tight',
        'figure.facecolor':  'white',
        'axes.prop_cycle':   mpl.cycler(color=PALETTE),
    })
    if use_tex:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
