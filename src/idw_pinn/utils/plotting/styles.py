"""
Styling constants and configuration for publication-ready plots.

Contains font sizes, axis limits, color schemes, and data type configurations
for consistent visualization across all plotting modules.

Updated: January 2026
"""
import matplotlib.pyplot as plt


# =============================================================================
# DATA TYPE SELECTION - CHANGE THIS FLAG
# =============================================================================

DATA_TYPE = 'numerical'  # OPTIONS: 'numerical' or 'experimental'


# =============================================================================
# PUBLICATION-READY DEFAULTS
# =============================================================================

FONT_CONFIG = {
    'title': 16,
    'suptitle': 18,
    'axis_label': 14,
    'tick_label': 12,
    'colorbar_label': 13,
    'legend': 12,
}

# Hardcoded axis limits for cross-run consistency (all log scale)
AXIS_LIMITS = {
    'D_evolution': (1e-5, 1),
    'D_error': (1e-5, 1e5),
    'losses': (1e-12, 1e2),
    'lambdas': (1e-3, 1e5),
}

DPI_SAVE = 300

# Data type configurations
DATA_CONFIG = {
    'numerical': {
        'diff_coeff_true': 0.2,
        'diff_coeff_display': '0.2',
        'label': 'True D',
    },
    'experimental': {
        # Target D range in m²/s: 3.15e-10 to 4.05e-10
        # Corresponding D_norm range: 0.000507 to 0.000652
        'diff_coeff_range': (0.000507, 0.000652),
        'diff_coeff_physical_range': (3.15e-10, 4.05e-10),  # m²/s
        'label': 'Target D range',
    }
}


def set_publication_style():
    """Set matplotlib defaults for publication-quality figures."""
    plt.rcParams.update({
        'font.size': FONT_CONFIG['axis_label'],
        'axes.titlesize': FONT_CONFIG['title'],
        'axes.labelsize': FONT_CONFIG['axis_label'],
        'xtick.labelsize': FONT_CONFIG['tick_label'],
        'ytick.labelsize': FONT_CONFIG['tick_label'],
        'legend.fontsize': FONT_CONFIG['legend'],
        'figure.titlesize': FONT_CONFIG['suptitle'],
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'savefig.dpi': DPI_SAVE,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })