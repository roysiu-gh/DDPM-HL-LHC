# Package imports
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb

# Local imports
from visualisation import plot_detections, count_hist, energy_hist
from data_loading import select_event
from calculate_quantities import *

# ======= global matplotlib params =====
plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 12  # Set default font size (optional)

CWD = os.getcwd()
pileup_path = f"{CWD}/data/1-initial/pileup.csv"
tt_path = f"{CWD}/data/1-initial/ttbar.csv"

# === BEGIN Reading in Data ===
MAX_DATA_ROWS = None
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
    etas: ndarray
        1D dataset of particle etas, with particles whose \Delta R is greater than `boundary` removed.
    phis: ndarray
        1D dataset of particle phis, with particles whose \Delta R is greater than `boundary` removed.
    """
    # Calculate eta, phi of every particle in data
    p_mag = p_magnitude(jet_data[:,3:])
    etas = pseudorapidity(p_mag, jet_data[:,5])
    phis = to_phi(jet_data[:,3], jet_data[:,4])
    # Calculate the values of Delta R for each particle
    delta_eta= (etas - jet_centre[0])
    delta_phi = (phis - jet_centre[1])
    crit_R = np.sqrt(delta_eta*delta_eta + delta_phi*delta_phi)
    bounded_data = jet_data[crit_R <= boundary]
    return bounded_data, etas[crit_R <= boundary], phis[crit_R <= boundary]

MAX_EVENT_NUM = 999999

MU = 5

def foo_bar(jet_nos, pile_up_data, mu: int):
    """
    This function [CHANGE NAME PLEASE FFS] takes in an array of jet IDs, pile_ups and a mu-value and performs the following:
    
    1) Randomly selects pile_up event IDs, the number of randomly selected IDs corresponds to mu
    2) Selects all pile_up particles corresponding to the randomly chosen IDs. IDs which do not exist are kept for indexing LID and discarded later on
    3) Removes PDGID, charge
    4) Calculates etas, phis and appends them as columns
    5) Inserts jet IDs at the beginning
    6) Writes final combined array to data/combined.csv.gz

    Parameters
    ----------
    jet_nos: List[int] or ndarray
        1D List or 1D NumPy array of jet IDs.
    pile_up_data: ndarray
        Complete 2D NumPy of pile_up_data
    mu: int
        The number of pile_up IDs to use in noise
    
    Returns
    -------
    0 for successful completion
    """
    combined_array = []
    for jet_no in jet_nos:
        event_IDS = np.random.choice(pile_up[:,0], size = mu).astype(int)
        print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
        selected_pile_ups = [select_event(pile_up, event_ID, filter=True) for event_ID in event_IDS]
        # This for loop writes the LIDs by taking the index as LID/
        # This means invalid pile_ups are counted and can be discarded
        for ind, pile in enumerate(selected_pile_ups):
            pile[:,0] = ind + 1
        selected_jet = select_event(tt, jet_no, filter=False)
        selected_jet = np.delete(selected_jet, [1,2], axis=1)
        X_pmag = p_magnitude(selected_jet[:,1:])
        X_etas = pseudorapidity(X_pmag, selected_jet[:,-1])
        X_phis = to_phi(selected_jet[:,1], selected_jet[:,2])
        num_rows = selected_jet.shape[0]
        new_column = np.full((1,num_rows), 0)
        selected_jet = np.insert(selected_jet, 1, new_column, axis=1)
        selected_jet = np.hstack((selected_jet, X_etas.reshape(-1, 1)))
        selected_jet  = np.hstack((selected_jet, X_phis.reshape(-1, 1)))

        # Stack arrays on top of each other
        selected_pile_ups = np.vstack(selected_pile_ups)
        # Clearly, an invalid particle has completely zero momentum in all components (violates conservation of energy)
        # Therefore this masks out all the rows where the corresponding sample has zero in all 3 components of p
        # Equivalent to ensuring the L2 norm is 0 iff components are not zero since a norm is semi-positve definite
        zero_p_mask = ~((selected_pile_ups[:, 3] == 0) & (selected_pile_ups[:, 4] == 0) & (selected_pile_ups[:, 5] == 0))
        selected_pile_ups = selected_pile_ups[zero_p_mask]

        # Delete PGDID and charge columns
        X = np.delete(selected_pile_ups, [1,2], axis=1)

        # Now momenta start at 2nd column
        # Select p for calculations
        X_pmag = p_magnitude(X[:,1:])
        X_etas = pseudorapidity(X_pmag, X[:,-1])
        X_phis = to_phi(X[:,1], X[:,2])
        # Append etas and phis to end of column
        X = np.hstack((X, X_etas.reshape(-1, 1)))
        X  = np.hstack((X, X_phis.reshape(-1, 1)))
        # Label these pile_ups with their jet IDs
        num_rows = X.shape[0]
        new_column = np.full((1,num_rows), jet_no)
        X = np.insert(X, 0, new_column, axis=1)
        combined_array.append(np.vstack((selected_jet, X)))
    # Merge all subarrays
    combined_array = np.vstack(combined_array)
    np.savetxt("data/combined.csv.gz", combined_array, delimiter=",", header="NID,LID,px,py,pz,eta,phi", comments="", fmt="%10.10f")
    return 0

# === Example Usage of foo_bar ===
foo_bar([0,1], pile_up, 2)

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
BINS = (16,16)
jet_no = 0
 # TODO: work with new select_event function
# def generate_hist(tt_data, pile_up_data, jet_no, bins, mu, hist_plot="energy", energies = None) -> None:
#     """
#     This functions wraps all routines needed to generate a 2D histogram of particle counts.

#     This allows looping over mu, the number of pile ups, which allows us to generate a sequence of noisier images.

#     Routine:
#     1. Extract random pile-up
#     2. Choose a jet number
#     3. Calculate the jet centre using the jet data
#     4. Merge the data together
#     5. Mask the data using delta_R condition
#     6. Plot the histogram using `count_hist/energy_hist` and saves it as png and pdf (vectorised and smaller filesize)

#     Parameters
#     ----------
#     tt_data: ndarray
#         2D array of particle information about t-tbar decays
#     pile_up_data: ndarray
#         2D array of pile up event information
#     jet_no: int,
#         Select jet to plot
#     bins: (int, int)
#         Number of bins to use for the 2D histogram plot (eta, phi)
#     mu: int,
#         Number of pile-up events to select

#     Returns: None
#     """
#     chosen_pile_up = select_event(pile_up_data, mu)
#     plot_data = select_event(tt_data, jet_no, max_data_rows=MAX_DATA_ROWS)
#     data = merge_data(plot_data, chosen_pile_up)
#     # All columns are passed in, so make sure to select last 3 columns for the 3-momenta
#     jet_centre = jet_axis(plot_data[:,3:])
#     # print("centre", jet_centre)

#     # Delta R is calculated relative to the jet centre, and over all particles including pile-up
#     masked_data, etas, phis = delta_R(jet_centre, data)
#     # print(len(etas) == len(phis))
#     masked_energies = np.sqrt(masked_data[:,3]*masked_data[:,3] + masked_data[:,4]*masked_data[:,4]+masked_data[:,5]*masked_data[:,5])
#     # energy_normed = normalize_data(energies, energy_norm_factor)
#     energy_normed = (masked_energies - energy_min) / energy_norm_denom
#     # print(energy_normed)
#     # Function appends "_hist" to the end
#     if hist_plot == "count":
#         count_hist(etas, phis, jet_no=jet_no,bins=bins, filename=f"eta_phi_jet{jet_no}_MU{mu}")
#     elif hist_plot == "energy": 
#         energy_hist(etas, phis, jet_no=jet_no,bins=bins, energies=energy_normed, filename=f"eta_phi_jet{jet_no}_MU{mu}")
#     else:
#         raise ValueError("Error: hist_plot was not 'count' or 'energy'.\n")
    
# === EXAMPLE USAGE OF GENERATING IMAGES ===
BINS = [(8,8), (16,16),(32,32), (64,64)]
# for bin in BINS:
#     generate_hist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000)
#     generate_hist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000, hist_plot="count")
#  Use new visualisaton to just distinguish pile up and jet
# tt_jet = select_event(tt, 493, max_data_rows=MAX_DATA_ROWS)
# plot_detections(
#         tt_bar=delta_R(jet_axis(tt_jet[:,3:]), tt_jet)[0],
#         pile_ups=delta_R(jet_axis(tt_jet[:,3:]), select_event(pile_up, 300))[0],
#         centre = jet_axis(tt_jet[:,3:]),
#         filename=f"eta_phi_jet{jet_no}_valid",
#         base_radius_size=100,
#         jet_no=493,
#         momentum_display_proportion=1,
#         cwd=CWD,
#     )
# print("lol", delta_R(jet_axis(tt_jet[:,3:]), select_event(pile_up, 300))[0])
# print("tt", delta_R(jet_axis(tt_jet[:,3:]), tt_jet)[0],)
# Example looping over MU, which we will probably use
# for MU in MUs:
#    generate_hist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BINS, mu=MU)
# === END GENERATION ===

#===========
