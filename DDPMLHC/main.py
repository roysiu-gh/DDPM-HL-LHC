# Package imports
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import sys
# Local imports
from DDPMLHC.config import *
from DDPMLHC.generate_plots.visualisation import plot_detections, generate_2dhist
from DDPMLHC.dataset_ops.data_loading import select_event
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import write_combined_csv
# ======= global matplotlib params =====
plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 12  # Set default font size (optional)
MAX_DATA_ROWS = None
# === BEGIN Reading in Data ===
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

# === Example Usage of foo_bar ===
write_combined_csv(range(10000), tt, pile_up, 10)

#################################################################################
# jet_no = 493
# plot_data = select_event(tt, jet_no)
# jet_centre = get_axis_eta_phi(plot_data[:,3:])
# mu = 200
# # event_IDS = np.random.choice(pile_up[:,0], size = mu).astype(int)
# event_IDS = np.random.randint(0, np.max(pile_up[:,0]), size = mu).astype(int)
# selected_pile_ups = [select_event(pile_up,event_ID, filter=True) for event_ID in event_IDS]
# selected_pile_ups = np.vstack(selected_pile_ups)
# zero_p_mask = ~((selected_pile_ups[:, 3] == 0) & (selected_pile_ups[:, 4] == 0) & (selected_pile_ups[:, 5] == 0))
# number_of_false = np.size(zero_p_mask) - np.count_nonzero(zero_p_mask)
# selected_pile_ups = selected_pile_ups[zero_p_mask]
# print("Number of false pile-ups", number_of_false)
# plot_detections(
#     tt_bar=plot_data,
#     centre = jet_centre,
#     pile_ups=selected_pile_ups,
#     jet_no=jet_no,
#     filename=f"eta_phi_jet{jet_no}_Mu{mu}_select_missing_events",
#     base_radius_size=300,
#     momentum_display_proportion=1,
#     cwd=CWD,
#     pdgids=False
# )
# plot_detections(
#     tt_bar=delta_R(jet_centre, plot_data)[0],
#     centre = jet_centre,
#     pile_ups=selected_pile_ups,
#     jet_no=493,
#     filename=f"eta_phi_jet{jet_no}_deltaR_Mu = {mu}",
#     base_radius_size=100,
#     momentum_display_proportion=1,
#     cwd=CWD,
# )
# plot_detections(
#     plot_data=plot_data,
#     centre = jet_centre,
#     filename=f"eta_phi_jet{jet_no}_cropped",
#     base_radius_size=1,
#     momentum_display_proportion=0.9,
#     cwd=CWD,
# )

#################################################################################

# === 2D Histograms ===
# BINS = (16,16)
# # === EXAMPLE USAGE OF GENERATING IMAGES ===
max_pileup_id = np.max(pile_up[:,0])
# # max_event_ids = np.linspace(0, max_pileup_id, num=max_pileup_id+1)
jet_no = 493
BINS = [256]
for BIN in BINS:
    generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=1000, max_event_id=max_pileup_id, energies=None, )
    # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=10000)
    # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=50, hist_plot="count")
#  Use new visualisaton to just distinguish pile up and jet

# ====== END 2D HIST ====

