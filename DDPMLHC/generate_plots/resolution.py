# from visualisation import plot_detections, count_hist, energy_hist, generate_2dhist
from DDPMLHC.config import *
from DDPMLHC.data_loading import select_event_deprecated
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.data_loading import *
from DDPMLHC.dataset_ops.process_data import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import multiprocessing
# start_time_global = time.time()

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# pile_up = np.genfromtxt(
#     PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# tt = np.genfromtxt(
#     TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# tt = EventSelector(tt)
# pile_up = EventSelector(pile_up)

# mus = np.arange(0,51,step=1)
# high_PU_no = pile_up.max_ID


# tt_int = np.genfromtxt(
#     TT_INTER_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
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
    # TODO
    jet_4mmtm = calculate_four_momentum_massless(jet_ids, jet_px, jet_py, jet_pz)
    summed_jet_4mmtm = jet_4mmtm
    # summed_jet_4mmtm = np.sum(jet_4mmtm, axis = 1)
    # j_m = contraction(summed_jet_4mmtm[0], summed_jet_4mmtm[1], summed_jet_4mmtm[2], summed_jet_4mmtm[3])
    jet_energy = np.sum(summed_jet_4mmtm[0])
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
    differences = total_energy - jet_energy
    # print("differences", differences)
    return differences
 # p^\nu = (E,px,py,pz) in natural units
    # if q == "mass":
    #     s
    # elif q == "energy":
    
    # else:
    #     raise ValueError("q was not one of 'mass' or 'energy'")
    # elif q == "eta":

    # elif q == "phi"
# pile_up_indices = pile_up[:,0]
# max_event_num = np.unique(pile_up_indices).astype(int)
# jet_masses = tt_int[:,6]
# jet_pt = tt_int[:,7]

def mean_quantity_diff(params):
    """
    For a given number of pile_up events mu in MUs

    Sample random pile_ups

    Find <q_\mu^jet - q_0^jet> against \mu using quantity_diff
    Find std(q_\mu^jet - q_0^jet) 

    Repeat for all jets (or a max number of jets)

    Find the average by dividing by the number of jets (not jet particles)
    """
    # max_jet_no = np.shape(np.unique(jet_data[:,0]))[0]
    # Store jet COMS
    jet_data, pile_up_data, mu, max_jet_no, high_PU_no= params
    # for ind,mu in enumerate(MUs):
    print("Begin mu = ", mu)
    m_total = 0
    E_total = 0
    p_T_total = 0
    # start_time = time.time()
    # print("ind", ind)
    # sys.exit(1)
    
    # pu_nos = pile_up_data.select_event()
    # event_IDS = np.random.choice(max_event_num, size = (max_jet_no, mu))
    # valid_event_IDS = event_IDS[np.isin(event_IDS, max_event_num)]        print(event_IDS)
    E_arr = []  
    for jet_no in range(0, max_jet_no):
        cd = jet_data.select_event(jet_no)
        # print(f"jet {jet_no}: ",cd)
        if mu != 0:
            event_IDS = np.random.randint(low = 0, high = high_PU_no, size = mu, dtype=np.int32)
            # print(event_IDS)
            pile_ups = [pile_up_data.select_event(event_ID) for event_ID in event_IDS]
            # print("Pile ups: ", pile_ups)
            pile_ups = np.vstack(pile_ups)
            [px, py, pz], eta, phi = delta_R(centre, px, py, pz, eta, phi)
            # pile_ups = np.delete(pile_ups, [0,1,2], axis=1)
            E= quantity_diff(cd[:,0], cd[:,3],cd[:,4],cd[:,5], pile_ups[:,0], pile_ups[:,1], pile_ups[:,2])
            # print(E)
            E_arr.append(E)
        else:
            # 0 pile  up means diff is 0 anyways
            E = 0
        # print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
        # selected_pile_ups = [select_event(pile_up_data, event_ID, filter=False) for event_ID in event_IDS]
        # selected_pile_ups = np.vstack(selected_pile_ups)
        # exit(1)
        # selected_pile_ups = pile_up_data[np.isin(pile_up_indices, selected_events)]
       
        # m_total += m
        E_total += E
        # p_T_total += p_T
    # Mean over number of jets
    # end_time = time.time()
    # print(f"Loop mu = {mu}: {end_time - start_time} seconds")
    # m_total /= max_jet_no
    E_total /= max_jet_no
    E_std = np.std(E_arr) if mu != 0 else 0
    # p_T_total /= max_jet_no
    # print(m_total)
    # print(E_total)
    # print(p_T_total)
    print(f"End mu = {mu}")
    return E_total, E_std

def mean_quantity_diff_noisy(mu):
    if event_stats_path is None:
        event_stats_path = f"{CWD}/data/2-intermediate/noisy_mu{mu}_event_level.csv"
    print(f"Doing mu = {mu}...")
    events_dat = np.genfromtxt(
        event_stats_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
    )
    save_path = f"{CWD}/data/plots/1D_histograms/mu{mu}/"
    Path(save_path).mkdir(parents=False, exist_ok=True)

    event_eta = events_dat[:, 4]
    event_mass = events_dat[:, 6]
    event_pT = events_dat[:, 7]