"""Project-wide matplotlib style.

    from src.plot_style import apply
    apply()

``apply(use_tex=True)`` switches to real LaTeX rendering — needs a LaTeX
install on the system, otherwise matplotlib falls back to mathtext.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    'primary':   '#1f77b4',  # blue
    'secondary': '#f93414',  # red
    'accent':    '#57A048',  # green
    'neutral':   '#6B6B6B',  # gray
}

PALETTE = ['#1f77b4', '#f93414', '#57A048']


def apply(use_tex: bool = False) -> None:
    """Apply the project's matplotlib style."""
    plt.rcdefaults()
    mpl.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size':         12,
        'axes.titlesize':    12,
        'axes.labelsize':    12,
        'axes.titleweight':  'normal',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linestyle':    '--',
        'grid.linewidth':    0.6,
        'xtick.labelsize':   11,
        'ytick.labelsize':   11,
        'legend.frameon':    False,
        'legend.fontsize':   8,
        'figure.figsize':    (10, 5),
        'figure.dpi':        110,
        'savefig.dpi':       180,
        'savefig.bbox':      'tight',
        'figure.facecolor':  'white',
        'axes.prop_cycle':   mpl.cycler(color=PALETTE),
    })
    if use_tex:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
