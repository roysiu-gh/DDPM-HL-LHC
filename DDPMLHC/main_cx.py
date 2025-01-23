# Package imports
import numpy as np
import matplotlib as mpl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.generate_plots.generate_1d_plots import plot_1d_histograms

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

MAX_DATA_ROWS = 100_000

# === Read in data
print("0 :: Loading original data")
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=10*MAX_DATA_ROWS
)
print("FINISHED loading data\n")

#################################################################################

# # === 2D Histograms ===
# BINS = (16,16)
# # === EXAMPLE USAGE OF GENERATING IMAGES ===
max_pileup_id = np.max(pile_up[:,0])
# # max_event_ids = np.linspace(0, max_pileup_id, num=max_pileup_id+1)
jet_no = 493
BINS = [32]
# for BIN in BINS:
#     generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=200, max_event_id=max_pileup_id, energies=None)
#     # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=10000)
#     # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=50, hist_plot="count")
# #  Use new visualisaton to just distinguish pile up and jet
# # ====== END 2D HIST ====

#################################################################################

# === Create noisy events
print("1 :: Creating noisy events")
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 0, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 1, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 3, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 5, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 10, save_path=INTERMEDIATE_PATH)
print("FINISHED creating noisy events\n")

# === Collapse noisy data to event-level
print("2 :: Collapsing noisy data to event-level")
calculate_event_level_quantities(0, INTERMEDIATE_PATH)
calculate_event_level_quantities(1, INTERMEDIATE_PATH)
calculate_event_level_quantities(3, INTERMEDIATE_PATH)
calculate_event_level_quantities(5, INTERMEDIATE_PATH)
calculate_event_level_quantities(10, INTERMEDIATE_PATH)
print("FINISHED collapsing noisy data to event-level\n")

# === Draw 1D histograms
print("3 :: Drawing 1D histograms")
plot_1d_histograms(mu=0)
plot_1d_histograms(mu=1)
plot_1d_histograms(mu=3)
plot_1d_histograms(mu=5)
plot_1d_histograms(mu=10)
print("FINISHED drawing 1D histograms\n")

# print("4 :: Drawing overlaid histograms with varying mu")
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
# print("FINISHED drawing overlaid histograms\n")

print("DONE ALL.")
