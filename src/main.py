# Package imports
import numpy as np
import os
import matplotlib.pyplot as plt

# Local imports
from visualisation import plot_detections
from data_loading import select_jet, random_rows_from_csv
from calculate_quantities import pseudorapidity, to_phi

# ======= global matplotlib params =====
plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 12  # Set default font size (optional)

CWD = os.getcwd()
file_path = f"{CWD}/data/1-initial/pileup.csv"
tt_path = f"{CWD}/data/1-initial/ttbar.csv"

# === BEGIN Reading in Data ===
MAX_DATA_ROWS = 1000
pile_up = np.genfromtxt(
    file_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
# === END reading in Data ===

def jet_axis(p):
    """
    Finds the jet axis of a given jet using the massless limit m/E \approx 0.

    Parameters
    ----------
    p : ndarray
        2D array of floats containing momenta of constituents of a jet.
    Returns
    ----------
    eta,phi: float,float
        Location of jet axis in eta-phi space.
    """
    total_p = np.sum(p, axis=0)
    # jet_mag = p_magnitude(total_p)
    jet_mag = np.linalg.norm(total_p)
    eta = pseudorapidity(jet_mag, total_p[2])
    phi = to_phi(total_p[0], total_p[1])
    return eta, phi

def delta_R(jet_centre, jet_data, boundary=1.0):
    """
    This function takes in particle information, and removes particles whose \Delta R(eta,phi) > 1.0 and returns all the others.

    Parameters
    ----------
    centre : tuple of (float,float)
        The jet beam axis. 2-tuple in the form (eta,phi) used for calculating \Delta R
    jet_no : int
        Which jet to select from data
    data: ndarray
        2D dataset containing particle information.
    boundary: float, default = 1.0
        The maximum \Delta R for which particles with a larger value will be cut off.
    Returns
    ----------
    bounded_data: ndarray
        2D dataset of particle information, with particles whose \Delta R is greater than `boundary` removed.
    """
    # Calculate eta, phi of every particle in data
    # jet_data = select_jet(data, jet_no)
    p_mag = p_magnitude(jet_data[:,3:])
    etas = pseudorapidity(p_mag, jet_data[:,5])
    phis = to_phi(jet_data[:,3], jet_data[:,4])
    # Calculate the values of Delta R for each particle
    delta_eta= (etas - jet_centre[0])
    delta_phi = (phis - jet_centre[1])
    crit_R = np.sqrt(delta_eta*delta_eta + delta_phi*delta_phi)
    print("critR: ", crit_R)
    keep = jet_data[crit_R <= boundary]
    return keep

# MU to define the number of random pileup events to take
MU = 3
MAX_EVENT_NUM = 999999
chosen_pile_up = random_rows_from_csv(pile_up, MU)
jet_no = 0
data = np.concatenate((select_jet(tt, jet_no), chosen_pile_up), axis=0) 
jet_centre = jet_axis(data)

plot_data = select_jet(tt, jet_no)

plot_detections(
    plot_data=plot_data,
    centre = jet_centre,
    filename=f"eta_phi_jet{jet_no}",
    base_radius_size=10,
    momentum_display_proportion=1,
    cwd=CWD,
)
plot_detections(
    plot_data=plot_data,
    centre = jet_centre,
    filename=f"eta_phi_jet{jet_no}_cropped",
    base_radius_size=1,
    momentum_display_proportion=0.9,
    cwd=CWD,
)