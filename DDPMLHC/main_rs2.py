# Package imports
import numpy as np
# Local imports
from DDPMLHC.config import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.generate_plots.generate_1d_plots import plot_1d_histograms

# === Read in data
print("0 :: Loading original data")
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
print("FINISHED loading data\n")

# Calculate what percent of PUs are empty
pu_ids = pile_up[:, 0]
total_pileups = pu_ids[-1]
print(f"Final number in the first column: {total_pileups}")
unique_vals = len(np.unique(pu_ids))
print(f"Number of unique values in the first column: {unique_vals}")
print( f"Percentage of non-empty pile-ups {(unique_vals/total_pileups)*100}" )

#################################################################################

# Old data generation code

# TTBAR_NUM = 100_000

# === Create noisy events
print("1 :: Creating noisy events")
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 0, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 5, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 10, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 15, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(TTBAR_NUM), tt, pile_up, 100, save_path=INTERMEDIATE_PATH)
# make_noisy_data(range(TTBAR_NUM), tt, pile_up, 200, save_path=INTERMEDIATE_PATH)
print("FINISHED creating noisy events\n")

# === Collapse noisy data to event-level
print("2 :: Collapsing noisy data to event-level")
calculate_event_level_quantities(0, INTERMEDIATE_PATH)
calculate_event_level_quantities(5, INTERMEDIATE_PATH)
calculate_event_level_quantities(10, INTERMEDIATE_PATH)
calculate_event_level_quantities(15, INTERMEDIATE_PATH)
# calculate_event_level_quantities(100, INTERMEDIATE_PATH)
# calculate_event_level_quantities(200, INTERMEDIATE_PATH)
print("FINISHED collapsing noisy data to event-level\n")

# === Draw 1D histograms
print("3 :: Drawing 1D histograms")
plot_1d_histograms(mu=0)
plot_1d_histograms(mu=5)
plot_1d_histograms(mu=10)
plot_1d_histograms(mu=15)
# plot_1d_histograms(mu=100)
# plot_1d_histograms(mu=200)
print("FINISHED drawing 1D histograms\n")

print("DONE ALL.")
