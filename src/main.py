# Import constants
from config import *

# Package imports
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb

# Local imports
from visualisation import plot_detections, count_hist, energy_hist
from data_loading import select_event, random_rows_from_csv
from calculate_quantities import *

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


def merge_data(tt_data, pile_up_data):
    """
    Wrapper function to vertically  stack 2D NumPy array data of the same shape, each arranged so that the same ordered information is contained in each one.

    Parameters
    ----------
    tt_data: ndarray
        2D array of particle information about t-tbar decays
    pile_up_data: ndarray
        2D array of pile up event information

    Returns
    -------
    data: ndarray
        V-stacked 2D array (same number of columns) containing both input tt_data and pile_up data
    """
    return np.concatenate((tt_data, pile_up_data), axis=0)

combined_data = merge_data(tt, pile_up)
# === END reading in Data ===
def l2_norm(data):
    # Calculate the integral of the squared values using the trapezoidal rule
    dx = (np.max(data) - np.min(data)) / len(data)
    integral = np.trapezoid(data * data, dx=dx)
    # Return the L^2 norm
    return np.sqrt(integral)

def normalize_data(data, norm):
    # Calculate the L^2 norm of the data
    # norm = l2_norm(data)
    
    # Avoid division by zero
    if norm == 0:
        raise ValueError("Norm is zero.")

    # Normalize the data
    normalized_data = data / norm
    return normalized_data

# MUs to define the number of random pileup events to take.
# Larger mu => more pileups sampled => noiser histogram
MUs = [5, 10, 100, 1000, 5000]
MAX_EVENT_NUM = 999999
# chosen_pile_up = random_rows_from_csv(pile_up, MU)
jet_no = 493
# data = np.concatenate((select_event(tt, jet_no), chosen_pile_up), axis=0) 
# jet_centre = jet_axis(data)

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

# === BEGIN Calculating energy normalisation factor  ===
# Massless limit, E^2 = p^2
energies = np.sqrt(combined_data[:,3]*combined_data[:,3] + combined_data[:,4]*combined_data[:,4]+combined_data[:,5]*combined_data[:,5])
energy_min = np.min(energies)
energy_max = np.max(energies)
energy_norm_denom = (energy_max - energy_min)

# === 2D Histograms ===
bins = (16,16)
jet_no = 493
 
def generate_hist(tt_data, pile_up_data, jet_no, bins, mu, hist_plot="energy", energies = None) -> None:
    """
    This functions wraps all routines needed to generate a 2D histogram of particle counts.

    This allows looping over mu, the number of pile ups, which allows us to generate a sequence of noisier images.

    Routine:
    1. Extract random pile-up
    2. Choose a jet number
    3. Calculate the jet centre using the jet data
    4. Merge the data together
    5. Mask the data using delta_R condition
    6. Plot the histogram using `count_hist/energy_hist` and saves it as png and pdf (vectorised and smaller filesize)

    Parameters
    ----------
    tt_data: ndarray
        2D array of particle information about t-tbar decays
    pile_up_data: ndarray
        2D array of pile up event information
    jet_no: int,
        Select jet to plot
    bins: (int, int)
        Number of bins to use for the 2D histogram plot (eta, phi)
    mu: int,
        Number of pile-up events to select

    Returns: None
    """
    chosen_pile_up = random_rows_from_csv(pile_up_data, mu)
    plot_data = select_event(tt_data, jet_no, max_data_rows=MAX_DATA_ROWS)
    data = merge_data(plot_data, chosen_pile_up)
    # All columns are passed in, so make sure to select last 3 columns for the 3-momenta
    jet_centre = jet_axis(plot_data[:,3:])
    # print("centre", jet_centre)

    # Delta R is calculated relative to the jet centre, and over all particles including pile-up
    masked_data, etas, phis = delta_R(jet_centre, data)
    # print(len(etas) == len(phis))
    masked_energies = np.sqrt(masked_data[:,3]*masked_data[:,3] + masked_data[:,4]*masked_data[:,4]+masked_data[:,5]*masked_data[:,5])
    # energy_normed = normalize_data(energies, energy_norm_factor)
    energy_normed = (masked_energies - energy_min) / energy_norm_denom
    # print(energy_normed)
    # Function appends "_hist" to the end
    if hist_plot == "count":
        count_hist(etas, phis, jet_no=jet_no,bins=bins, filename=f"eta_phi_jet{jet_no}_MU{mu}")
    elif hist_plot == "energy": 
        energy_hist(etas, phis, jet_no=jet_no,bins=bins, energies=energy_normed, filename=f"eta_phi_jet{jet_no}_MU{mu}")
    else:
        raise ValueError("Error: hist_plot was not 'count' or 'energy'.\n")
    
# === EXAMPLE USAGE OF GENERATING IMAGES ===
bins = [(8,8), (16,16),(32,32), (64,64)]
for bin in bins:
    generate_hist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000)
    generate_hist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000, hist_plot="count")
#  Use new visualisaton to just distinguish pile up and jet
# tt_jet = select_event(tt, 493, max_data_rows=MAX_DATA_ROWS)
# plot_detections(
#         tt_bar=delta_R(jet_axis(tt_jet[:,3:]), tt_jet)[0],
#         pile_ups=delta_R(jet_axis(tt_jet[:,3:]), random_rows_from_csv(pile_up, 300))[0],
#         centre = jet_axis(tt_jet[:,3:]),
#         filename=f"eta_phi_jet{jet_no}_valid",
#         base_radius_size=100,
#         jet_no=493,
#         momentum_display_proportion=1,
#         cwd=CWD,
#     )
# print("lol", delta_R(jet_axis(tt_jet[:,3:]), random_rows_from_csv(pile_up, 300))[0])
# print("tt", delta_R(jet_axis(tt_jet[:,3:]), tt_jet)[0],)
# Example looping over MU, which we will probably use
# for MU in MUs:
#    generate_hist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BINS, mu=MU)
# === END GENERATION ===
