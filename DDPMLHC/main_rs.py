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

# # Calculate what percent of PUs are empty
# pu_ids = pile_up[:, 0]
# total_pileups = pu_ids[-1]
# print(f"Final number in the first column: {total_pileups}")
# unique_vals = len(np.unique(pu_ids))
# print(f"Number of unique values in the first column: {unique_vals}")
# print( f"Percentage of non-empty pile-ups {(unique_vals/total_pileups)*100}" )

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
    save_to_bmap(generator.vectorise(), jet_no=generator.event_id, mu=generator.mu)
    generator.visualise_current_event()
    generator.visualise_current_event(show_pdgids=True)

    next(generator)  # Load jet 1
    save_to_bmap(generator.vectorise(), jet_no=generator.event_id, mu=generator.mu)
    generator.visualise_current_event()
    generator.visualise_current_event(show_pdgids=True)

    generator.select_jet(42)
    save_to_bmap(generator.vectorise(), jet_no=generator.event_id, mu=generator.mu)
    generator.visualise_current_event()
    generator.visualise_current_event(show_pdgids=True)
