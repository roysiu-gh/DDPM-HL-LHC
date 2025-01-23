# Package imports
import numpy as np
import matplotlib as mpl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.data_loading import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.generate_plots.overlaid_1d_plots import create_overlay_plots

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

# mus = [0, 1, 3, 5, 10, 15, 25, 30, 50, 75, 100, 125, 150, 175, 200]
# for mu in mus:
#     cur_generator = NoisyGenerator(tt, pile_up, mu=mu)
#     cur_generator.save_event_level_data()
#     plot_1d_histograms(mu=mu)

#################################################################################

create_overlay_plots([0, 5, 10, 15, 30])
create_overlay_plots([0, 10, 30, 50])
create_overlay_plots([0, 25, 50, 75, 100])
create_overlay_plots([0, 50, 100, 150, 200])

#################################################################################

for mu in [0, 50, 500]:
    generator = NoisyGenerator(tt, pile_up, mu=mu)
    next(generator)  # Load jet 0
    generator.bmap_current_event()
    generator.visualise_current_event()

    next(generator)  # Load jet 1
    generator.bmap_current_event()
    generator.visualise_current_event()

# print(generator.vectorise(for_bmap=True).reshape((BMAP_SQUARE_SIDE_LENGTH, BMAP_SQUARE_SIDE_LENGTH)), "\n")
# print(generator.vectorise(for_bmap=False).reshape((BMAP_SQUARE_SIDE_LENGTH, BMAP_SQUARE_SIDE_LENGTH)))
