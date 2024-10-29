import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ======= global matplotlib params =====
plt.rcParams['text.usetex'] = False  # Use LaTeX for rendering text
plt.rcParams['font.size'] = 12      # Set default font size (optional)

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
    2212: r"$p$ (Proton)"
}

MAX_DATA_ROWS = 1000
# jetnumber , pdgid , charge , px , py , pz
pile_up = np.genfromtxt(pile_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS)
tt = np.genfromtxt(tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS)

def select_jet(data, num):
    """
    Select data with jets #num from data file
    """
    if MAX_DATA_ROWS <= num:
        raise ValueError(f"Requested jet {num} is not in data. Max jet number is {MAX_DATA_ROWS}")
    return data[data[:,0] == num]

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
    if ( np.shape(p_mag) != np.shape(p_z)):
        raise ValueError("Error: p_mag shape not equal to p_z shape")
    return np.arctanh(p_z/p_mag)

def to_phi(p_x, p_y):
    """
    p_x, p_y are 1D NumPy arrays of x and y momenta respectively, where each element corresponds to a particle.

    This function finds the angle phi (radians) from the 2 components of transverse momentum p_x,  p_y using arctan(p_y/p_x)
    """
    if (np.shape(p_x) != np.shape(p_y)):
        raise ValueError("Error: p_x shape not equal to p_y shape")
    return np.arctan2(p_y, p_x)

def plot_detections(data, jet_no=0, filename="eta_phi", base_dot_size=100, momentum_display_proportion=1.0, verbose=True):
    """Plot a jet and output to a PNG. Pretty self-explanatory.
    
    Parameters:
    - data: dataset containing particle information
    - jet_no: specific jet number to plot from dataset
    - filename: The name to save the file as (PNG)
    - base_dot_size: base size for plot circles
    - momentum_display_proportion: proportion of total momentum to display in plot limits - default 1.0 (display all detections)
    - verbose: print detailed information
    """
    # Retrieve data for the specified jet and calculate momenta and angles
    jet_data = select_jet(data, jet_no)
    momenta = jet_data[:,3:]
    pmag = p_magnitude(momenta)
    if verbose: print("Constituent momenta magnitudes:\n", pmag)
    pz = jet_data[:,5]
    eta = pseudorapidity(pmag, pz)
    phi = to_phi(momenta[:,0], momenta[:,1])

    # Variable dot sizes, prop to pmag
    dot_sizes = base_dot_size * (pmag / np.max(pmag))

    # Prepare color mapping based on the pdgid values
    pdgid_values = jet_data[:, 1]
    unique_pdgid = np.unique(pdgid_values)
    num_colours = len(unique_pdgid)
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, num_colours)))  # Cmap tab10 (or tab20) for easily distinguishable colours
    colour_mapping = {pid: cmap(i) for i, pid in enumerate(unique_pdgid)}
    colours = [colour_mapping[pid] for pid in pdgid_values]

    # Plotting
    # Grid stuff
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"$\phi$ vs $\eta$ of jet ${jet_no}, prop={momentum_display_proportion}$")
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\phi$")
    # Set phi range to +/-pi and adjust tick marks
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / 4))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda val, pos: f"{(val / np.pi)}$\pi$" if val != 0 else "0"))
    ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7)

    if momentum_display_proportion == 1.0:  # Show all
        ax.scatter(eta, phi, color=colours, marker='o', facecolors="none", linewidths=0.1, s=dot_sizes)
    else:  # Show a certain proportion
        # Calculate cumulative momentum and sort by eta to adjust limits
        total_momentum = np.sum(pmag)
        sorted_indices = np.argsort(pmag)[::-1]  # Sort by descending momentum
        cumulative_momentum = np.cumsum(pmag[sorted_indices])
        target_momentum = total_momentum * momentum_display_proportion

        # Find cutoff index where cumulative momentum reaches target ratio
        cutoff_index = np.searchsorted(cumulative_momentum, target_momentum, side='right')
        eta_cropped = eta[sorted_indices][:cutoff_index]
        phi_cropped = phi[sorted_indices][:cutoff_index]
        pmag_cropped = pmag[sorted_indices][:cutoff_index]
        colours_cropped = [colours[i] for i in sorted_indices[:cutoff_index]]
        dot_sizes_cropped = dot_sizes[sorted_indices[:cutoff_index]]

        ax.scatter(eta_cropped, phi_cropped, color=colours_cropped, marker='o', facecolors="none", linewidths=1 ,s=dot_sizes_cropped*5)

    # Add legend for pdgid values and particle names
    handles = []
    for pid in unique_pdgid:
        particle_name = PDG_IDS.get(pid, "Not in dict")
        handles.append(Patch(color=colour_mapping[pid], label=f"PDG ID: {int(pid)}, \n{particle_name}"))
    # Shrink plot and put right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig(f"{CWD}/data/plots/test/{filename}.png", dpi=1000)


plot_detections(data=tt, jet_no=0, filename="eta_phi")
plot_detections(data=tt, jet_no=0,filename="eta_phi_cropped", momentum_display_proportion=0.9)
