# Import constants
from config import *

# Package imports
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import sys
# Local imports
from visualisation import plot_detections, generate_2dhist
from data_loading import select_event
from calculate_quantities import *
from process_data import foo_bar
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
# foo_bar([0,1], pile_up, 2)

#################################################################################
# jet_no = 10
# plot_data = select_event(tt, jet_no)
# jet_centre = COM_eta_phi(plot_data[:,3:])
# mu = 3
# event_IDS = np.random.choice(pile_up[:,0], size = mu).astype(int)
# selected_pile_ups = [select_event(pile_up,event_ID, filter=True) for event_ID in event_IDS]
# selected_pile_ups = np.vstack(selected_pile_ups)
# zero_p_mask = ~((selected_pile_ups[:, 3] == 0) & (selected_pile_ups[:, 4] == 0) & (selected_pile_ups[:, 5] == 0))
# selected_pile_ups = selected_pile_ups[zero_p_mask]

# plot_detections(
#     tt_bar=plot_data,
#     centre = jet_centre,
#     pile_ups=selected_pile_ups,
#     jet_no=jet_no,
#     filename=f"eta_phi_jet{jet_no}_noscreened_Mu = {mu}",
#     base_radius_size=1000,
#     momentum_display_proportion=1,
#     cwd=CWD,
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
jet_no = 493
# === EXAMPLE USAGE OF GENERATING IMAGES ===

BINS = [32]
for BIN in BINS:
    generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=200)
    # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=10000)
    # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=50, hist_plot="count")
#  Use new visualisaton to just distinguish pile up and jet

# ====== END 2D HIST ====

