# Import constants
from config import *

# Package imports
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb

# Local imports
from visualisation import plot_detections, count_hist, energy_hist, generate_2dhist
from data_loading import select_event
from calculate_quantities import *
from process_data import foo_bar
# ======= global matplotlib params =====
plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 12  # Set default font size (optional)

# === BEGIN Reading in Data ===
pile_up = np.genfromtxt(
    pileup_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

# === Example Usage of foo_bar ===
foo_bar([0,1], tt, pile_up, 2)

#################################################################################

# plot_data = select_event(tt, jet_no)

# plot_detections(
#     plot_data=plot_data,
#     centre = jet_centre,
#     filename=f"eta_phi_jet{jet_no}",
#     base_radius_size=10,
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
BINS = (16,16)
jet_no = 0
# === EXAMPLE USAGE OF GENERATING IMAGES ===
BINS = [(8,8), (16,16),(32,32), (64,64)]
# for bin in BINS:
#     generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000)
#     generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000, hist_plot="count")
#  Use new visualisaton to just distinguish pile up and jet

#====== END 2D HIST ====

#################################################################################

jet_no = 10
plot_data = select_event(tt, jet_no)
jet_centre = COM_eta_phi(plot_data[:,3:])
mu = 3
# event_IDS = np.random.choice(pile_up[:,0], size = mu).astype(int)
event_IDS = np.array([1, 2, 3])
selected_pile_ups = [select_event(pile_up, event_ID, filter=True) for event_ID in event_IDS]
selected_pile_ups = np.vstack(selected_pile_ups)
zero_p_mask = ~((selected_pile_ups[:, 3] == 0) & (selected_pile_ups[:, 4] == 0) & (selected_pile_ups[:, 5] == 0))
selected_pile_ups = selected_pile_ups[zero_p_mask]

plot_detections(
    tt_bar=plot_data,
    centre = jet_centre,
    pile_ups=selected_pile_ups,
    jet_no=jet_no,
    filename=f"eta_phi_jet{jet_no}_noscreened_Mu={mu}",
    base_radius_size=500,
    momentum_display_proportion=1,
    cwd=CWD,
)
