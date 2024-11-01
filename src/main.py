# Package imports
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb

# Local imports
from visualisation import plot_detections
from data_loading import select_jet, random_rows_from_csv
from calculate_quantities import *

# ======= global matplotlib params =====
plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 12  # Set default font size (optional)

CWD = os.getcwd()
file_path = f"{CWD}/data/1-initial/pileup.csv"
tt_path = f"{CWD}/data/1-initial/ttbar.csv"

# === BEGIN Reading in Data ===
MAX_DATA_ROWS = 100_000
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

# # MU to define the number of random pileup events to take
# MU = 3
# MAX_EVENT_NUM = 999999
# chosen_pile_up = random_rows_from_csv(pile_up, MU)
# jet_no = 0
# data = np.concatenate((select_jet(tt, jet_no), chosen_pile_up), axis=0) 
# jet_centre = jet_axis(data)

#################################################################################

# plot_data = select_jet(tt, jet_no)

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

# Calculate particle momentum magnitudes and pseudorapidity
num = len(tt)
jet_ids = tt[:, 0]
p_mag = p_magnitude(tt[:, 3:])
pz, px, py = tt[:, 5], tt[:, 3], tt[:, 4]
eta = pseudorapidity(p_mag, pz)
p_T = np.sqrt(px**2 + py**2)

jet_four_momenta = calculate_four_momentum_massless(jet_ids, px, py, pz)
jet_p2 = contraction(jet_four_momenta)
jet_masses = np.sqrt(jet_p2)

# Kinda fun to print
for jet_id in range(0, len(jet_four_momenta), 132):
    four_mmtm = jet_four_momenta[jet_id]
    p2 = jet_p2[jet_id]
    print(f"Jet ID: {jet_id}, Total 4-Momenta: [{four_mmtm[0]:.3f}, {four_mmtm[1]:.3f}, {four_mmtm[2]:.3f}, {four_mmtm[3]:.3f}], Contraction p^2: {p2:.3f}")

# Define the save path and plot characteristics
save_path = f"{CWD}/data/plots/data_exploration/"
plot_params = {
    "bins": 500,
    "color": "skyblue",
    "edgecolor": "none",
    "kde": True,
    "stat": "density"  # Equivalent to `density=True` in plt.hist
}

print("Plotting histograms...")
sb.set_theme(style="whitegrid")

# Histogram of momentum magnitudes
plt.figure(figsize=(10, 6))
sb.histplot(p_mag, **plot_params)
plt.title(f"Normalised Histogram of {num} Individual Particle Momentum Magnitudes")
plt.xlabel("Momentum Magnitude")
plt.ylabel("Frequency Density")
plt.grid(axis="y", alpha=0.75)
plt.savefig(f"{save_path}/p_mag.png", dpi=600)

# Histogram of pseudorapidity
plt.figure(figsize=(10, 6))
sb.histplot(eta, **plot_params)  # Adjust bins for eta
plt.title(f"Normalised Histogram of {num} Individual Particle Pseudorapidity ($\eta$)")
plt.xlabel("Pseudorapidity (η)")
plt.ylabel("Frequency Density")
plt.grid(axis="y", alpha=0.75)
plt.savefig(f"{save_path}/eta.png", dpi=600)

# Histogram of transverse momentum
plt.figure(figsize=(10, 6))
sb.histplot(p_T, **plot_params)
plt.title(f"Normalised Histogram of {num} Individual Particle Transverse Momentum ($p_T$)")
plt.xlabel("Transverse Momentum (p_T)")
plt.ylabel("Frequency Density")
plt.grid(axis="y", alpha=0.75)
plt.savefig(f"{save_path}/p_T.png", dpi=600)

# Histogram of p^2
plt.figure(figsize=(10, 6))
sb.histplot(jet_p2, **plot_params)
plt.title(f"Normalised Histogram of {len(jet_p2)} Jet ($p^2$)")
plt.xlabel("p2")
plt.ylabel("Frequency Density")
plt.grid(axis="y", alpha=0.75)
plt.savefig(f"{save_path}/jet_p2.png", dpi=600)


# Histogram of p^2
plt.figure(figsize=(10, 6))
sb.histplot(jet_masses, **plot_params)
plt.title(f"Normalised Histogram of {len(jet_p2)} Jet Mass")
plt.xlabel("Mass")
plt.ylabel("Frequency Density")
plt.grid(axis="y", alpha=0.75)
plt.savefig(f"{save_path}/jet_mass.png", dpi=600)