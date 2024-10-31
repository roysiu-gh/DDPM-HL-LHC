import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Circle

# ======= global matplotlib params =====
plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 12  # Set default font size (optional)

CWD = os.getcwd()
pile_path = f"{CWD}/data/1-initial/pileup.csv"
tt_path = f"{CWD}/data/1-initial/ttbar.csv"

PDG_IDS = {
    -211: r"$\pi^-$ (Pion)",
    -321: r"$K^-$ (Kaon)",
    0: r"$\gamma$ (Photon)",
    130: r"$K^0_S$ (K-short)",
    211: r"$\pi^+$ (Pion)",
    22: r"$\gamma$ (Photon)",
    321: r"$K^+$ (Kaon)",
    11: r"$e^-$ (Electron)",
    -2112: r"$\bar{n}$ (Antineutron)",
    -11: r"$e^+$ (Positron)",
    2112: r"$n$ (Neutron)",
    2212: r"$p$ (Proton)",
}

# Global color scheme (tab20 for extended color range)
unique_pdgids = sorted(PDG_IDS.keys())
cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, len(unique_pdgids))))
GLOBAL_CMAP = {pid: cmap(i) for i, pid in enumerate(unique_pdgids)}
# === BEGIN Reading in Data ===
MAX_DATA_ROWS = 1000
pile_up = np.genfromtxt(
    pile_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
# === END reading in Data ===


def select_jet(data, num):
    """Select data with jets #num from data file"""
    if MAX_DATA_ROWS <= num:
        raise ValueError(
            f"Requested jet {num} is not in data. Max jet number is {MAX_DATA_ROWS}"
        )
    return data[data[:, 0] == num]


def p_magnitude(p):
    """
    p is a 2D NumPy array where each element is a particle's 3-momentum

    This function simply calculates and returns the magnitude of the momenta of each particle and returns it as a 1D NumPy array
    """
    return np.linalg.norm(p, axis=1)


def pseudorapidity(p_mag, p_z):
    """
    p_mag is a 1D NumPy array where each element is the magnitude of momentum for a particle
    p_z is the z-component of momentum of a particle (the component along the beam-axis)

    this function calculates the pseudorapidity and returns it as a 1D NumPy array
    https://en.wikipedia.org/wiki/Pseudorapidity
    """
    if np.shape(p_mag) != np.shape(p_z):
        raise ValueError("Error: p_mag shape not equal to p_z shape")
    return np.arctanh(p_z / p_mag)


def to_phi(p_x, p_y):
    """
    p_x, p_y are 1D NumPy arrays of x and y momenta respectively, where each element corresponds to a particle.

    This function finds the angle phi (radians) from the 2 components of transverse momentum p_x,  p_y using arctan(p_y/p_x)
    """
    if np.shape(p_x) != np.shape(p_y):
        raise ValueError("Error: p_x shape not equal to p_y shape")
    return np.arctan2(p_y, p_x)


def plot_detections(
    data,
    jet_axis,
    jet_no=0,
    filename="eta_phi",
    base_radius_size=1,
    momentum_display_proportion=1.0,
    verbose=True,
) -> None:
    """Plot a jet and output to a PNG. Obvious.

    Parameters
    ----------
    data: ndarray
        2D dataset containing particle information
    jet_no: int
        Specific jet number to plot from dataset
    jet_axis: (float,float)
        (eta,phi) of the jet axis.
    filename: str
        The name to save the file as (PNG)
    base_dot_size: float
        Base size for plot circles
    momentum_display_proportion: flost
        proportion of total momentum to display in plot limits - default 1.0 (display all detections)
    verbose: bool
        print detailed information

    Returns
    -------
    """
    # Retrieve data for the specified jet and calculate momenta and angles
    jet_data = select_jet(data, jet_no)
    momenta = jet_data[:, 3:]
    pmag = p_magnitude(momenta)
    if verbose:
        print("Constituent momenta magnitudes:\n", pmag)
    pz = jet_data[:, 5]
    eta = pseudorapidity(pmag, pz)
    phi = to_phi(momenta[:, 0], momenta[:, 1])

    # Variable dot sizes, prop to pmag
    radius_sizes = 0.1 * base_radius_size * (pmag / np.max(pmag))

    # Get colours from global cmap based on PDG IDs
    pdgid_values = jet_data[:, 1]
    colours = [GLOBAL_CMAP.get(pid, "black") for pid in pdgid_values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(
        f"$\phi$ vs $\eta$ of jet {jet_no}, prop={momentum_display_proportion}"
    )
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\phi$")
    # Set phi range to +/-pi and adjust tick marks
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / 4))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda val, pos: f"{(val / np.pi)}$\pi$" if val != 0 else "0"
        )
    )
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.7)

    # Show particles up to the target proportion of total momentum
    sorted_indices = np.argsort(pmag)[::-1]
    if momentum_display_proportion != 1.0:  # Do some calculating for the cutoff idx
        total_momentum = np.sum(pmag)
        cumulative_momentum = np.cumsum(pmag[sorted_indices])
        target_momentum = total_momentum * momentum_display_proportion
        cutoff_index = np.searchsorted(cumulative_momentum, target_momentum, side='right')

        # Delete unwanted particles from plotting data
        pdgid_values = pdgid_values[sorted_indices][:cutoff_index]
        eta = eta[sorted_indices][:cutoff_index]
        phi = phi[sorted_indices][:cutoff_index]
        colours = [colours[i] for i in sorted_indices[:cutoff_index]]
        radius_sizes = radius_sizes[sorted_indices[:cutoff_index]]
        radius_sizes *= 5  # Make larger radius for cropped plots, consider making this variable in future
        linewidth = 1
    else:  # Show all particles
        cutoff_index = None
        linewidth = 0.1

    # Plot centres
    # FIX THIS FOR CROPS
    dot_sizes = radius_sizes*radius_sizes  # Dots sizes based on area so scale as square
    ax.scatter(eta, phi, color=colours, marker='.', edgecolors='none', s=dot_sizes)

    # Plot circles prop to width
    for pdgid, e, p, color, radius in zip(pdgid_values, eta, phi, colours, radius_sizes):
        linestyle = "-" if pdgid >= 0 else "--"
        circle = Circle((e, p), radius=radius/100, edgecolor=color, facecolor='none', linewidth=linewidth, linestyle=linestyle)
        ax.add_patch(circle)

    # Add legend for pdgid values and particle names
    # NB this will show all particles in the collision in the legend, even if cropped out (is desired behaviour)
    handles = []
    detected_pdgids = set(pdgid_values)
    for pid in unique_pdgids:
        particle_name = PDG_IDS.get(pid, "Not in PDGID dict")
        if pid not in detected_pdgids:
            continue  # Remove particles not detected
        handles.append(
            Patch(
                color=GLOBAL_CMAP[pid], label=f"PDG ID: {int(pid)}, \n{particle_name}"
            )
        )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.savefig(f"{CWD}/data/plots/test/{filename}.png", dpi=1000)


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

def random_rows_from_csv(data, num_rows):
    num_rows = min(num_rows, data.shape[0])
    random_indices = np.random.choice(data.shape[0], num_rows, replace=False)
    random_rows = data[random_indices]
    
    return random_rows
# MU to define the number of random pileup events to take
MU: int = 3
MAX_EVENT_NUM = 999999
chosen_pile_up = random_rows_from_csv(pile_up, MU)
# for i in range(0,MU):
jet_no = 0
data = np.concatenate((select_jet(tt, jet_no), chosen_pile_up), axis=0) 
# print(jet_axis(data[:, 3:]))
jet_centre = jet_axis(data)
# print()
# plot_detections(data=delta_R(jet_centre, data), jet_no=jet_no, filename=f"eta_phi_jet{jet_no}")
plot_detections(
    data=tt,
    jet_no=jet_no,
    jet_axis = jet_centre,
    filename=f"eta_phi_jet{jet_no}",
    base_radius_size=10,
    momentum_display_proportion=1,
)
plot_detections(
    data=tt,
    jet_no=jet_no,
    jet_axis = jet_centre,
    filename=f"eta_phi_jet{jet_no}_cropped",
    base_radius_size=1,
    momentum_display_proportion=0.9,
)