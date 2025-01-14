# Package imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.dataset_ops.generate_intermediate_event_data import process_noisy_data
from DDPMLHC.generate_plots.generate_1d_plots import foobah
from DDPMLHC.generate_plots.visualisation import plot_detections, generate_2dhist
from DDPMLHC.dataset_ops.data_loading import select_event

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

MAX_DATA_ROWS = 100_000

# ======= global matplotlib params =====
# plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
# plt.rcParams["font.size"] = 12  # Set default font size (optional)

# === Read in data
print("0 :: Loading original data")
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

#################################################################################

# # === 2D Histograms ===
# BINS = (16,16)
# # === EXAMPLE USAGE OF GENERATING IMAGES ===
# max_pileup_id = np.max(pile_up[:,0])
# # # max_event_ids = np.linspace(0, max_pileup_id, num=max_pileup_id+1)
# jet_no = 493
# BINS = [32]
# for BIN in BINS:
#     generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=200, max_event_id=max_pileup_id, energies=None)
#     # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=10000)
#     # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=50, hist_plot="count")
# #  Use new visualisaton to just distinguish pile up and jet

# # ====== END 2D HIST ====

#################################################################################

INTERMEDIATE_PATH = f"{CWD}/data/2-intermediate/try/"
OUT_PATH_1D_HIST = f"{CWD}/data/plots/1D_histograms/particles/"
MAX_DATA_ROWS = 100_000

# === Create noisy events
print("1 :: Creating noisy events")
write_combined_csv(range(100), tt, pile_up, 10, save_path=INTERMEDIATE_PATH)
write_combined_csv(range(100), tt, pile_up, 100, save_path=INTERMEDIATE_PATH)
write_combined_csv(range(100), tt, pile_up, 200, save_path=INTERMEDIATE_PATH)
print("FINISHED creating noisy events\n")

# === Make noisy events with extra data (e.g. transverse mmtm)
print("2 :: Making noisy events with extra data")
process_noisy_data(10, INTERMEDIATE_PATH)
process_noisy_data(100, INTERMEDIATE_PATH)
process_noisy_data(200, INTERMEDIATE_PATH)
print("FINISHED making noisy events with extra data\n")

# === Draw 1D histograms
print("3 :: Drawing 1D histograms")

print("Loading intermediate data...")
# Load intermediate data
tt = np.genfromtxt(
    TT_EXT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
px = tt[:, 1]
py = tt[:, 2]
pz = tt[:, 3]
eta = tt[:, 4]
p_T = tt[:, 6]
p = p_magnitude(px, py, pz)
hist_data_particles = [
    {
        "name": "Momentum $p$ [GeV]",
        "data": p,
        "plot_params": {"xlog": True},
        "save_filename": "p",
        "save_path": OUT_PATH_1D_HIST,
    },
    {
        "name": "Pseudorapidity $\eta$",
        "data": eta,
        "plot_params": {},
        "save_filename": "eta",
        "save_path": OUT_PATH_1D_HIST,
    },
    {
        "name": "Transverse Momentum $p_T$ [GeV]",
        "data": p_T,
        "plot_params": {"xlog": True},
        "save_filename": "pT",
        "save_path": OUT_PATH_1D_HIST,
    },
]

foobah(mu=0, event_stats_path=JET_PATH)
foobah(mu=10)
foobah(mu=100)
foobah(mu=200)
foobah(mu=300)
print("FINISHED drawing 1D histograms\n")

print("DONE ALL.")
