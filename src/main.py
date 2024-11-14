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
MAX_DATA_ROWS = None
# === BEGIN Reading in Data ===
pile_up = np.genfromtxt(
    pileup_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
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
#     jet_no=jet_no,
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
# jet_no = 0
# === EXAMPLE USAGE OF GENERATING IMAGES ===
# BINS = [(8,8), (16,16),(32,32), (64,64)]
# for bin in BINS:
#     generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000)
#     generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=bin, mu=10000, hist_plot="count")
#  Use new visualisaton to just distinguish pile up and jet

#====== END 2D HIST ====

def quantity_diff(jet_ids, jet_px, jet_py, jet_pz, pu_px, pu_py, pu_pz):
    """
    This function calculates and returns q_\mu^jet - q_0^jet for different jets with \mu pileups versus no pile_up.

    Parameters
    ----------
    max_jet_no: int
        Maximum number of jets to take an average over. Increasing this will yield a more accurate mass distortion but also increase the runtime!
    jet_data:
    pile_up_data: ndarray
        2D NumPy array of pileups to add to the jet
    q: str
        Quantity to plot. Valid values are "pt", "energy", "eta", "phi", "mass"
    Returns
    -------
    differences: tuple[float]
        Tuple containing (mass, energy, p_T, eta, phi) differences
    """
    # Case insensitivity
    # q = q.lower()
    # Calculate q_0^jet quantities. Here, since we are doing over a jet, we sum the 4-momenta and then do calcs
    jet_four_momenta = calculate_four_momentum_massless(jet_ids, jet_px, jet_py, jet_pz)
    jet_p2 = contraction(jet_four_momenta)
    jet_mass = np.sum(np.sqrt(jet_p2))
    jet_pt = np.sum(np.sqrt(jet_px*jet_px + jet_py*jet_py))
    jet_energy = jet_four_momenta[0] # p^\nu = (E,px,py,pz) in natural units
    # Calculate quantities with pileup
    print("jet_px: ", jet_px)
    print("pu_px: ", pu_px)
    combined_px = np.concatenate((jet_px, pu_px), axis=0)
    combined_py = np.concatenate((jet_py, pu_py), axis=0)
    combined_pz = np.concatenate((jet_pz, pu_pz), axis=0)
    # print(combined_px)
    # total_four_momenta = np.array([np.zeros(4) for _ in range(len(combined_px) + 1)])
    total_energy = np.sum(np.sqrt(combined_px*combined_px + combined_py*combined_py+combined_pz*combined_pz))
    total_px = np.sum(combined_px)
    total_py = np.sum(combined_py)
    total_pz = np.sum(combined_pz) 
    # c_four_momenta = np.array([total_energy, total_px, total_py, total_pz])
    c_p2 = total_energy*total_energy - (total_px*total_px + total_py*total_py + total_pz*total_pz)
    c_m = np.sqrt(c_p2)
    c_pt = np.sqrt(total_px*total_px + total_py*total_py)
    # mass difference, energy difference, p_T difference, eta difference, phi difference
    differences = (c_m - jet_mass, total_energy - jet_energy, c_pt - jet_pt)
    return differences
 # p^\nu = (E,px,py,pz) in natural units
    # if q == "mass":
    #     s
    # elif q == "energy":
    
    # else:
    #     raise ValueError("q was not one of 'mass' or 'energy'")
    # elif q == "eta":

    # elif q == "phi"
def mean_quantity_diff(jet_data, pile_up_data, MUs, max_jet_no=1000):
    """
    For a given number of pile_up events mu in MUs

    Sample random pile_ups

    Find q_\mu^jet - q_0^jet using quantity_diff

    Repeat for all jets (or a max number of jets)

    Find the average by dividing by the number of jets (not jet particles)
    """
    # max_jet_no = np.shape(np.unique(jet_data[:,0]))[0]
    max_event_num = np.unique(pile_up_data[:,0]).astype(int)
    data_y = np.zeros((3, len(MUs)))
    for ind,mu in enumerate(MUs):
        m_total = 0
        E_total = 0
        p_T_total = 0
        for jet_no in range(0, max_jet_no + 1):
            event_IDS = np.random.choice(max_event_num, size = mu)
            event_IDS = event_IDS[np.isin(event_IDS, max_event_num)]
            # print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
            selected_pile_ups = [select_event(pile_up_data, event_ID, filter=False) for event_ID in event_IDS]
            selected_pile_ups = np.vstack(selected_pile_ups)[:,3:]
            cd = jet_data[jet_data[:,0] == jet_no]
            m, E, p_T = quantity_diff(cd[:,0], cd[:,3],cd[:,4],cd[:,5], selected_pile_ups[:,0], selected_pile_ups[:,1], selected_pile_ups[:,2])
            m_total += m
            E_total += E
            p_T_total += p_T
        # Mean over number of jets
        m_total /= max_jet_no
        E_total /= max_jet_no
        p_T_total /= max_jet_no
        data_y[0][ind] = m_total
        data_y[1][ind] = E_total
        data_y[2][ind] = p_T_total
        print(f"End mu = {mu}")
    for name,data in zip(["Mass", "Energy", "pT"], data_y):
        fig  = plt.figure((8,6))
        plt.plot(MUs, data)
        plt.savefig(f"{CWD}/data/plots/Mean_{name}_diff.pdf", format="pdf")
        plt.close()
mean_quantity_diff(tt, pile_up, [1,5,10,15,20,25])