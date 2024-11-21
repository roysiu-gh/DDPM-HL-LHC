# from visualisation import plot_detections, count_hist, energy_hist, generate_2dhist
from config import *
from data_loading import select_event
from calculate_quantities import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 1.0 # set the value globally
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.labelsize'] = '14'     # fontsize of the x and y labels
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.axisbelow'] = 'True'   # whether axis gridlines and ticks are below
                                        # the axes elements (lines, text, etc)
mpl.rcParams['legend.fontsize'] = 25
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5

pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt_int = np.genfromtxt(
    TT_INTER_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
def quantity_diff(jet_ids, jet_px, jet_py, jet_pz, jet_mass, jet_pt, pu_px, pu_py, pu_pz):
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
    # TODO
    jet_4mmtm = calculate_four_momentum_massless(jet_ids, jet_px, jet_py, jet_pz)
    summed_jet_4mmtm = np.sum(jet_4mmtm, axis = 1)
    # j_m = contraction2(summed_jet_4mmtm[0], summed_jet_4mmtm[1], summed_jet_4mmtm[2], summed_jet_4mmtm[3])
    jet_energy = summed_jet_4mmtm[0]
    # jet_pt = np.sqrt(summed_jet_4mmtm[1] +  summed_jet_4mmtm[2])
    # jet_energy = np.sum(jet_four_momenta[0]) # p^\nu = (E,px,py,pz) in natural units
    # print("jet_energy:", jet_energy)
    # Calculate quantities with pileup
    # print("jet_px: ", jet_px)
    # print("pu_px: ", pu_px)
    combined_px = np.concatenate((jet_px, pu_px), axis=0)
    combined_py = np.concatenate((jet_py, pu_py), axis=0)
    combined_pz = np.concatenate((jet_pz, pu_pz), axis=0)
    total_px = np.sum(combined_px)
    total_py = np.sum(combined_py)
    total_pz = np.sum(combined_pz) 
    # print(combined_px)
    # total_four_momenta = np.array([np.zeros(4) for _ in range(len(combined_px) + 1)])
    # print(3-mom)
    p_mag = p_magnitude(combined_px,combined_py,combined_pz)
    total_energy = np.sum(p_mag)
    # print("tot_eng", total_energy)
    
    # c_four_momenta = np.array([total_energy, total_px, total_py, total_pz])
    c_pt2 = total_px*total_px + total_py*total_py
    c_pt = np.sqrt(c_pt2)
    c_p2 = total_energy*total_energy - (c_pt2 + total_pz*total_pz)
    # print("c_p2", c_p2)
    c_m = np.sqrt(c_p2)
    # mass difference, energy difference, p_T difference, eta difference, phi difference
    differences = (c_m - jet_mass, total_energy - jet_energy, c_pt - jet_pt)
    print("differences", differences)
    return differences
 # p^\nu = (E,px,py,pz) in natural units
    # if q == "mass":
    #     s
    # elif q == "energy":
    
    # else:
    #     raise ValueError("q was not one of 'mass' or 'energy'")
    # elif q == "eta":

    # elif q == "phi"
max_event_num = np.unique(pile_up[:,0]).astype(int)
jet_masses = tt_int[:,6]
jet_pt = tt_int[:,7]
def mean_quantity_diff(jet_data, pile_up_data, MUs, max_event_num = max_event_num, max_jet_no=1000):
    """
    For a given number of pile_up events mu in MUs

    Sample random pile_ups

    Find <q_\mu^jet - q_0^jet> against \mu using quantity_diff

    Repeat for all jets (or a max number of jets)

    Find the average by dividing by the number of jets (not jet particles)
    """
    # max_jet_no = np.shape(np.unique(jet_data[:,0]))[0]
    data_y = np.zeros((3, len(MUs)))
    # Store jet COMS
    for ind,mu in enumerate(MUs):
        print("Begin mu = ", mu)
        # print("ind", ind)
        # sys.exit(1)
        m_total = 0
        E_total = 0
        p_T_total = 0
        # TODO: mask
        for jet_no in range(0, max_jet_no + 1):
            event_IDS = np.random.choice(max_event_num, size = mu)
            event_IDS = event_IDS[np.isin(event_IDS, max_event_num)]
            # print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
            p_mag = p_magnitude(jet_px, jet_py, jet_pz)
            etas = pseudorapidity(p_mag, jet_pz)
            phis = to_phi(jet_px, jet_py)
            # Calculate the values of Delta R for each particle
            delta_eta= (etas - centre[0])
            delta_phi = (phis - centre[1])
    crit_R = np.sqrt(delta_eta*delta_eta + delta_phi*delta_phi)
            selected_pile_ups = [select_event(pile_up_data, event_ID, filter=False) for event_ID in event_IDS]
            selected_pile_ups = np.vstack(selected_pile_ups)[:,3:]
            cd = jet_data[jet_data[:,0] == jet_no]
            m, E, p_T = quantity_diff(cd[:,0], cd[:,3],cd[:,4],cd[:,5], jet_masses[jet_no], jet_pt[jet_no], selected_pile_ups[:,0], selected_pile_ups[:,1], selected_pile_ups[:,2])
            m_total += m
            E_total += E
            p_T_total += p_T
        # Mean over number of jets
        m_total /= max_jet_no
        E_total /= max_jet_no
        p_T_total /= max_jet_no
        # print(m_total)
        # print(E_total)
        # print(p_T_total)
        data_y[0][ind] += m_total
        data_y[1][ind] += E_total
        data_y[2][ind] += p_T_total
        print(f"End mu = {mu}")
    for name,data in zip(["Mass", "Energy", "pT"], data_y):
        fig  = plt.figure(figsize=(8,6))
        plt.plot(MUs, data)
        plt.savefig(f"{CWD}/data/plots/Mean_{name}_diff.pdf", format="pdf")
        plt.xlabel("$\mu$")
        plt.ylabel(f"{name}")
        plt.close()
# Array of jet etas and phis
# jet_centres = tt_int[:,4:6]
# tt_masked = [delta_R(jet_centre, tt)]
mus = np.linspace(1,100,5).astype(int)
mean_quantity_diff(tt, pile_up, mus, max_jet_no=10)