# Package imports
import numpy as np
import matplotlib as mpl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.data_loading import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.generate_plots.generate_1d_plots import plot_1d_histograms

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

MAX_DATA_ROWS = 100_000

# === Read in data
print("0 :: Loading original data")
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = EventSelector(tt)
pile_up = EventSelector(pile_up)
print("FINISHED loading data\n")

#################################################################################

# TTBAR_NUM = 100_000

# # === Create noisy events
# print("1 :: Creating noisy events")
# make_noisy_data(range(TTBAR_NUM), tt, pile_up, 0, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(TTBAR_NUM), tt, pile_up, 5, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(TTBAR_NUM), tt, pile_up, 10, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(TTBAR_NUM), tt, pile_up, 15, save_path=INTERMEDIATE_PATH)
# # make_noisy_data(range(TTBAR_NUM), tt, pile_up, 100, save_path=INTERMEDIATE_PATH)
# # make_noisy_data(range(TTBAR_NUM), tt, pile_up, 200, save_path=INTERMEDIATE_PATH)
# print("FINISHED creating noisy events\n")

# # === Collapse noisy data to event-level
# print("2 :: Collapsing noisy data to event-level")
# calculate_event_level_quantities(0, INTERMEDIATE_PATH)
# calculate_event_level_quantities(5, INTERMEDIATE_PATH)
# calculate_event_level_quantities(10, INTERMEDIATE_PATH)
# calculate_event_level_quantities(15, INTERMEDIATE_PATH)
# # calculate_event_level_quantities(100, INTERMEDIATE_PATH)
# # calculate_event_level_quantities(200, INTERMEDIATE_PATH)
# print("FINISHED collapsing noisy data to event-level\n")

# # === Draw 1D histograms
# print("3 :: Drawing 1D histograms")
# plot_1d_histograms(mu=0)
# plot_1d_histograms(mu=5)
# plot_1d_histograms(mu=10)
# plot_1d_histograms(mu=15)
# # plot_1d_histograms(mu=100)
# # plot_1d_histograms(mu=200)
# print("FINISHED drawing 1D histograms\n")

# print("DONE ALL.")

FOO = NoisyGenerator(tt, pile_up, mu=1)
# print(FOO)