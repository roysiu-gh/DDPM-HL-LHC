import os
import numpy as np
from cycler import cycler

CWD = os.getcwd()
np.random.seed(42)

# 1 initial paths
PILEUP_PATH = f"{CWD}/data/1-initial/pileup.csv"
TT_PATH = f"{CWD}/data/1-initial/ttbar.csv"

# 2 intermediate paths
INTERMEDIATE_PATH = f"{CWD}/data/2-intermediate/"
# Remove the following?
PILEUP_EXT_PATH = f"{CWD}/data/2-intermediate/pileup_extended.csv"
TT_EXT_PATH = f"{CWD}/data/2-intermediate/ttbar_extended.csv"
JET_PATH = f"{CWD}/data/2-intermediate/ttbar_jets.csv"

PILEUP_NUM = 1_000_000
TTBAR_NUM = 70903

MAX_DATA_ROWS = None

BMAP_SQUARE_SIDE_LENGTH = 16
label_fontsize = 16
tick_fontsize = 24
UNET_DIMS = 256


    # 'font.family' : r'Computer Modern Roman',
MPL_GLOBAL_PARAMS = {
    'text.usetex' : True, # use latex text
    'text.latex.preamble' : r'\usepackage{type1cm}\usepackage{braket}\usepackage{amssymb}\usepackage{amsmath}\usepackage{txfonts}', # latex packages
    'font.size' : 24,
    'figure.dpi' : 600,
    'figure.figsize' : (8, 6),
    'figure.autolayout' : True, # tight layout (True) or not (False)
    'axes.labelpad' : 5,
    'axes.xmargin' : 0,
    'axes.ymargin' : 0,
    'axes.grid' : False,
    # 'axes.autolimit_mode' : round_numbers, # set axis limits by rounding min/max values
    'axes.autolimit_mode' : 'data', # set axis limits as min/max values
    'xtick.major.pad' : 10,
    'ytick.major.pad' : 10,
    'xtick.labelsize': label_fontsize,
    'ytick.labelsize': label_fontsize,
    'lines.linewidth' : 1.3,
    'xtick.direction' : 'in',
    'ytick.direction' : 'in',
    'xtick.top' : True,
    'ytick.right' : True,
    'xtick.minor.visible' : True,
    'ytick.minor.visible' : True,
    'axes.prop_cycle': cycler(color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*3),
    'legend.framealpha': None
  }

