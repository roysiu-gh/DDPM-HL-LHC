"""
Contains all plotting and visualisation tools. 
"""

# Import constants
from DDPMLHC.config import *

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Circle

# Local imports
from DDPMLHC.calculate_quantities import get_axis_eta_phi, delta_R, p_magnitude, pseudorapidity, to_phi,centre_on_jet
from DDPMLHC.dataset_ops.data_loading import select_event_deprecated
from DDPMLHC.dataset_ops.process_data import wrap_phi


PDG_IDS = {
    0: r"$\gamma$ (Photon)",
    11: r"$e^-$ (Electron)",
    -11: r"$e^+$ (Positron)",
    22: r"$\gamma$ (Photon)",
    130: r"$K^0_S$ (K-short)",
    211: r"$\pi^+$ (Pion)",
    -211: r"$\pi^-$ (Pion)",
    321: r"$K^+$ (Kaon)",
    -321: r"$K^-$ (Kaon)",
    2112: r"$n$ (Neutron)",
    -2112: r"$\bar{n}$ (Antineutron)",
    2212: r"$p$ (Proton)",
}

# Global color scheme (tab20 for extended color range)
unique_abs_pdgids = sorted(abs(pdgid) for pdgid in PDG_IDS.keys())
cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, len(unique_abs_pdgids))))
GLOBAL_CMAP = {pid: cmap(i) for i, pid in enumerate(unique_abs_pdgids)}

def plot_detections(
    tt_bar,
    pile_ups=None,
    # jet_data,
    centre=(0,0),
    jet_no=0,
    filename="eta_phi",
    base_radius_size=1,
    momentum_display_proportion=1.0,
    verbose=True,
    pdgids=False,
    cwd=".",
) -> None:
    """Plot a jet and output to a PNG. Obvious.

    Parameters
    ----------
    tt_bar: ndarray
        2D dataset containing jets
    pile_ups: ndarray
        2D dataset containing pile_ups
    centre: (float,float)
        (eta,phi) of the jet axis.
    filename: str
        The name to save the file as (PNG)
    base_dot_size: float
        Base size for plot circles
    momentum_display_proportion: flost
        proportion of total momentum to display in plot limits - default 1.0 (display all detections)
    verbose: bool
        print detailed information
    pgdids: bool, default = False
        Whether to highlight PDGIDs in a legend. Useful to showcase decay products, but for just telling where the jets are, useful to leave off, When off (False), 2 colours are used: 1 for al jet constituents, the other for pile-ups
    Returns
    -------
    """
    # Retrieve data for the specified jet and calculate momenta and angles
    if pile_ups is not None:
        jet_data = np.concatenate((tt_bar, pile_ups), axis=0)
    else:
        jet_data = tt_bar
    
    print("Len jet: ", len(tt_bar))
    print("Len pileup: ", len(pile_ups))
    px = jet_data[:, 3]
    py = jet_data[:, 4]
    pz = jet_data[:, 5]
    pmag = p_magnitude(px, py, pz)
    if verbose:
        print("Constituent momenta magnitudes:\n", pmag)
    eta = pseudorapidity(pmag, pz)
    phi = to_phi(px, py)
    phi = wrap_phi(centre[1], phi)
    centre,eta, phi= centre_on_jet(centre, eta, phi)
    # Variable dot sizes, prop to pmag
    radius_sizes = 0.1 * base_radius_size * (pmag / np.max(pmag))

    # Get colours from global cmap based on PDG IDs

    pdgid_values = jet_data[:, 1]
    colours = [GLOBAL_CMAP.get(abs(pid), "black") for pid in pdgid_values]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.set_xlabel("$\Delta\eta$",fontsize=16)
    ax.set_ylabel("$\Delta\phi$", fontsize=16)
    # Set phi range to +/-pi and adjust tick marks
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / 4))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda val, pos: f"${(val / np.pi)}\pi$" if val != 0 else "$0$"
        )
    )
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.7)
        # Add circle to visualise boundary and jet centre
    ax.plot(centre[0], centre[1], marker="x", color="blue")
    
    # ax.set_ylim(0.5 * np.pi,1.25 * np.pi)
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
        linewidth = 2
    else:  # Show all particles
        cutoff_index = None
        linewidth = 0.1
    # print(len(colours[0:len(eta)]))
    # print(colours[0:len(eta)])
    # Plot centres
    # FIX THIS FOR CROPS
    # dot_sizes = radius_sizes*radius_sizes  # Dots sizes based on area so scale as square
    [px, py, pz], eta, phi = delta_R(centre, px, py, pz, eta, phi)

    bound = len(tt_bar)
    boundary_circle = plt.Circle((0,0), 1.0, color="black",linewidth=1,fill=False)
    ax.add_patch(boundary_circle)
    # Get correct number of colours for plotting
    colours = colours[0:len(eta)]
    if pdgids:
        # Plot circles prop to width
        ax.scatter(eta, phi, color=colours, marker='.', edgecolors='none', s=0)
        for pdgid, e, p, color, radius in zip(pdgid_values, eta, phi, colours, radius_sizes):
            linestyle = "-" if pdgid >= 0 else ":"
            circle = Circle((e, p), radius=radius/100, edgecolor=color, facecolor='none', linewidth=1, linestyle=linestyle, fill=False)
            ax.add_patch(circle)
            

        # Add legend for pdgid values and particle names
        # NB this will show all particles in the collision in the legend, even if cropped out (is desired behaviour)
        handles = []
        pdgid_values = jet_data[:, 1]  # Reset toget all PDG IDs again, in case some lost from the crop
        unique_detected_pdgids = sorted(set(pdgid_values))
        unique_abs_detected_pdgids = sorted(set(abs(i) for i in pdgid_values))

        # Arrange legend in ascending abs PDG IDs, with antiparts below if detected
        for abs_pid in unique_abs_detected_pdgids:
            colour = GLOBAL_CMAP.get(abs_pid, "grey")
            if abs_pid in unique_detected_pdgids:
                pid = abs_pid
                particle_name = PDG_IDS.get(pid, "Not in PDGID dict")
                handles.append( Patch(
                    label=f"{particle_name}",
                    color=colour
                ) )
            if -abs_pid in unique_detected_pdgids:
                pid = -abs_pid
                particle_name = PDG_IDS.get(pid, "Not in PDGID dict")
                handles.append( Patch(
                    label=f"{particle_name}",
                    edgecolor=colour, facecolor="none"
                ) )
        
        # Resize main plot and put legend to right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.scatter(eta[0:bound], phi[0:bound], color="red", marker='.', edgecolors='none', s=0, label="Jet particles")
        for pdgid, e, p, color, radius in zip(pdgid_values, eta[0:bound], phi[0:bound], colours, radius_sizes):
            # linestyle = "-" if pdgid >= 0 else "--"
            circle = Circle((e, p), radius=radius/100, edgecolor="red", facecolor='none', linewidth=1.1, fill=False)
            ax.add_patch(circle)
        # if pile_ups is not None:
        ax.scatter(eta[bound:], phi[bound:], color="lightblue", marker='.', edgecolors='none', s=0, label="Pile-ups")
        for pdgid, e, p, color, radius in zip(pdgid_values, eta[bound:], phi[bound:], colours, radius_sizes):
        # linestyle = "-" if pdgid >= 0 else "--"
            circle = Circle((e, p), radius=radius/100, edgecolor="blue", facecolor='none', linewidth=1.1, fill=False)
            ax.add_patch(circle)

        # ax.legend()
    # plt.savefig(f"{cwd}/data/plots/test/{filename}.png", dpi=1000)
    # ax.set_title( 
        # f"$\phi$ vs $\eta$ of jet {jet_no}, tot_num_parts={len(eta)}, mmtm_crop={momentum_display_proportion}"
    # )
    plt.savefig(f"{cwd}/data/plots/test/{filename}.pdf",bbox_inches = 'tight')
    plt.close()

def scale_energy_for_visual(energies, verbose=False):
    """For the purpose of visualisation.
    Scales energies to 256 bits for printing to grayscale, up to a specified std dev.
    Assumes non-negative input.
    """
    # Calculate energy scale and scale values to 256
    SD = np.std(energies)
    scale = 256 / (3 * SD)  # Value above which we represent as full brightness (256)
    print(scale)
    print("std dev", SD)
    scaled_energies = np.floor(energies * scale)
    scaled_energies[scaled_energies > 256] = 256  # Maximise at 256
    scaled_energies = scaled_energies.astype(int)
    return scaled_energies

def generate_2dhist(tt_data, pile_up_data, jet_no,mu, max_event_id, bins=32, boundary = 1.0, hist_plot="energy", energies = None, energy_norm_factor = None, cwd = ".", filename = "eta_phi") -> None:
    """
    This functions wraps all routines needed to generate a 2D histogram of particle counts or energies.

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
    event_IDS = np.random.choice(pile_up_data[:,0], size = mu).astype(int)
    # max_event_id = np.max(pile_up_data[:,0])
    
    # for jet_no in jet_nos:
    # event_IDS = np.random.randint(low = 0, high = max_event_id, size = mu, dtype=np.int32)
    # event_IDS = np.mgrid[0:(mu-1):(mu)*1j]
    # print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
    # selected_pile_ups now contain 2D arrays
    if mu != 0:
        selected_pile_ups2 = [select_event_deprecated(pile_up_data, event_ID, filter=True) for event_ID in event_IDS] 
        selected_pile_ups2 = np.vstack(selected_pile_ups2)
        selected_pile_ups = selected_pile_ups2[selected_pile_ups2[:,0] != -1]
    else:
        selected_pile_ups = np.array([])
    # Remove invalid pile_ups
    false_ids = np.size(selected_pile_ups2, axis=0) - np.size(selected_pile_ups, axis=0)
    print("Number of false Pile-up IDs: ",false_ids)
    jet_data = select_event_deprecated(tt_data, jet_no, max_data_rows=MAX_DATA_ROWS)
    data = np.vstack((jet_data,selected_pile_ups))
    # data = selected_pile_ups
    # print(len(data))
    # data = jet_data
    px = data[:,3]
    py = data[:,4]
    pz = data[:,5]
    pmag = p_magnitude(px, py, pz)
    # All columns are passed in, so make sure to select last 3 columns for the 3-momenta
    # Find jet axis
    jet_centre = get_axis_eta_phi(jet_data[:,3:])
    # Wrap jet axis phi between -1, 1
    print("centre", jet_centre)
    phis = to_phi(px, py)
    # Wrap particles w.r.t unwrapped jet axis
    phis = wrap_phi(jet_centre[1], phis)
    # jet_centre_wrapped = wrap_phi(jet_centre[1], jet_centre[1])
    # print("wrapped centre", jet_centre_wrapped)
    etas = pseudorapidity(pmag, pz)
    #shift particles so wrapped jet_axis is centred at (0,0)
    centre,etas_c, phis_c = centre_on_jet(jet_centre, etas, phis)
    # Delta R is calculated relative to the jet centre, and over all particles including pile-up
    bounded_momenta, etas2, phis2 = delta_R(centre, px, py, pz, etas_c, phis_c)
    # bounded_momenta, etas2, phis2 = [px,py,pz], etas_c, phis_c
    masked_px = bounded_momenta[0]
    masked_py = bounded_momenta[1]
    masked_pz = bounded_momenta[2]
    # print(len(etas) == len(phis))
    # masked_energies = np.sqrt(masked_data[:,3]*masked_data[:,3] + masked_data[:,4]*masked_data[:,4]+masked_data[:,5]*masked_data[:,5])
    # # energy_normed = normalize_data(energies, energy_norm_factor)
    # Unnormalised energies
    
    # print("etas2", etas2)
    # print("phis2", phis2)
    # print("Orig num of eta: ", len(etas))
    # print("Orig num of phi: ", len(phis))
    # print("New num of eta: ", len(etas2))
    # print("New num of phi: ", len(phis2))
    # print("Masked number of particles, etas: ", len(etas) - len(etas2))
    # print("Masked number of particles, etas: ", len(phis) - len(phis2))
    # print(len(etas2) == len(phis2))
    # print(np.min(etas2), np.max(etas2))
    # print(np.min(etas), np.max(etas))
    # exit(1)
    # energy_normed = (masked_energies - energy_min) / energy_norm_denom
    # print(energy_normed)
    # Function appends "_hist" to the end
    fig, ax = plt.subplots(1,1,figsize=(8, 6))
    # plt.subplots
    plt.xlabel(r'$\Delta\eta$', fontsize=16)
    plt.ylabel(r'$\Delta\phi$', fontsize=16)
    # plt.title(
    #     f"$\Delta\phi$ vs $\Delta\eta$ of jet {jet_no}, real_parts={len(etas2)}+false_pileup={false_ids}, bins={bins}"
    # )
    created_bins = np.mgrid[-boundary:boundary:bins*1j]
    save_str = f"{cwd}/data/hist/{filename}_jet{jet_no}_MU{mu}_bins_{bins}"
    if hist_plot == "count":
        plt.hist2d(etas2, phis2, bins=(created_bins, created_bins), cmap='Greys_r',)
        plt.colorbar(label='Count') 
        plt.savefig(f"{save_str}_count.png", dpi=600)
        plt.savefig(f"{save_str}_count.pdf",)
        # count_hist(etas2, phis2, jet_no=jet_no,bins=bins, filename=f"eta_phi_jet{jet_no}_MU{mu}")
        plt.close()
    elif hist_plot == "energy" and energies is not None:
        # energies = np.sqrt(masked_px*masked_px + masked_py*masked_py+masked_pz*masked_pz)
        plt.hist2d(etas2, phis2, bins=(created_bins, created_bins), weights=energies, cmap='Greys_r',)
        plt.colorbar(label='Energies') 
        plt.savefig(f"{save_str}_energies.png", dpi=600,bbox_inches="tight")
        plt.savefig(f"{save_str}_energies.pdf",bbox_inches="tight")
        plt.close()
    elif hist_plot == "energy" and energies is None:
        energies = np.sqrt(masked_px*masked_px + masked_py*masked_py+masked_pz*masked_pz)
        print(energies)
        energies2 = scale_energy_for_visual(energies)
        print(energies2)
        ax.hist2d(etas2, phis2, bins=(created_bins, created_bins), weights=np.log(energies2), cmap='Greys',)
        plt.colorbar(label='Energies') 
        plt.savefig(f"{save_str}_energies.png", dpi=600,bbox_inches="tight")
        plt.savefig(f"{save_str}_energies.pdf",bbox_inches="tight")
        plt.close()
    else:
        plt.clf()
        plt.close()
        raise ValueError("Error: hist_plot was not 'count' or 'energy'.\n")
    
# def generate_multiple2d()