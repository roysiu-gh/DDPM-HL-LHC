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

MAX_DATA_ROWS = 1_000_000

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

mus = [0, 1, 3, 5, 10, 15, 30, 50, 75, 100, 125, 150, 175, 200]
mus = [0, 1, 3, 5, 10, 15, 30, 50]

for mu in mus:
    cur_generator = NoisyGenerator(tt, pile_up, mu=mu)
    cur_generator.save_event_level_data()
    plot_1d_histograms(mu=mu)

#################################################################################

# ## WEIRD BEHAVIOUR
# FOO = NoisyGenerator(tt, pile_up, mu=5)
# for idx, item in enumerate(FOO):
#     print(idx)
#     # if idx == 5: break
# print(FOO._max_TT_no)