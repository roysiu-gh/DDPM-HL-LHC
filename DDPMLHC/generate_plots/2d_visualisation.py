"""
Contains all plotting and visualisation tools. 
"""

# Import constants
from DDPMLHC.config import *

# Package imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from DDPMLHC.calculate_quantities import *

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
        selected_pile_ups2 = [pile_up_data.select_event(event_ID) for event_ID in event_IDS] 
        selected_pile_ups2 = [x if x is not None for x in selected_pile_ups2]
        selected_pile_ups2 = np.vstack(selected_pile_ups2)
        # selected_pile_ups = selected_pile_ups2[selected_pile_ups2[:,0] != -1]
    else:
        selected_pile_ups2 = np.array([])
    # Remove invalid pile_ups
    false_ids = np.size(selected_pile_ups2, axis=0) - np.size(selected_pile_ups, axis=0)
    print("Number of false Pile-up IDs: ",false_ids)
    jet_data = tt_data.select_event(jet_no)
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
    px, py, pz = jet_data[:,3], jet_data[:,4], jet_data[:,5]
    jet_centre = get_axis_eta_phi(px, py, pz)
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
    # Unnormalised energies
    
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
    