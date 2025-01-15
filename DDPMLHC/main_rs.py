# Package imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.dataset_ops.generate_intermediate_event_data import calculate_event_level_quantities
from DDPMLHC.generate_plots.generate_1d_plots import foobah
from DDPMLHC.generate_plots.visualisation import plot_detections, generate_2dhist
from DDPMLHC.dataset_ops.data_loading import select_event

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# MAX_DATA_ROWS = 100_000

# === Read in data
print("0 :: Loading original data")
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

#################################################################################


INTERMEDIATE_PATH = f"{CWD}/data/2-intermediate/"
MAX_DATA_ROWS = 100_000

# # === Create noisy events
# print("1 :: Creating noisy events")
# make_noisy_data(range(1000), tt, pile_up, 0, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(1000), tt, pile_up, 10, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(1000), tt, pile_up, 100, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(1000), tt, pile_up, 200, save_path=INTERMEDIATE_PATH)
# print("FINISHED creating noisy events\n")

# === Collapse noisy data to event-level
print("2 :: Making noisy events with extra data")
calculate_event_level_quantities(0, INTERMEDIATE_PATH)
calculate_event_level_quantities(10, INTERMEDIATE_PATH)
calculate_event_level_quantities(100, INTERMEDIATE_PATH)
calculate_event_level_quantities(200, INTERMEDIATE_PATH)
print("FINISHED making noisy events with extra data\n")

# === Draw 1D histograms
print("3 :: Drawing 1D histograms")
foobah(mu=0)
foobah(mu=10)
foobah(mu=100)
foobah(mu=200)
print("FINISHED drawing 1D histograms\n")

# print("Loading intermediate data...")
# tt = np.genfromtxt(
#     TT_EXT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# px = tt[:, 1]
# py = tt[:, 2]
# pz = tt[:, 3]
# eta = tt[:, 4]
# p_T = tt[:, 6]
# p = p_magnitude(px, py, pz)
# OUT_PATH_1D_HIST = f"{CWD}/data/plots/1D_histograms/particles/"
# hist_data_particles = [
#     {
#         "name": "Momentum $p$ [GeV]",
#         "data": p,
#         "plot_params": {"xlog": True},
#         "save_filename": "p",
#         "save_path": OUT_PATH_1D_HIST,
#     },
#     {
#         "name": "Pseudorapidity $\eta$",
#         "data": eta,
#         "plot_params": {},
#         "save_filename": "eta",
#         "save_path": OUT_PATH_1D_HIST,
#     },
#     {
#         "name": "Transverse Momentum $p_T$ [GeV]",
#         "data": p_T,
#         "plot_params": {"xlog": True},
#         "save_filename": "pT",
#         "save_path": OUT_PATH_1D_HIST,
#     },
# ]

print("DONE ALL.")
