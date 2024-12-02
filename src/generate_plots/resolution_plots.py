# from visualisation import plot_detections, count_hist, energy_hist, generate_2dhist
from config import *
from dataset_ops.data_loading import select_event
from calculate_quantities import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import multiprocessing
start_time_global = time.time()

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

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
    # j_m = contraction(summed_jet_4mmtm[0], summed_jet_4mmtm[1], summed_jet_4mmtm[2], summed_jet_4mmtm[3])
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
pile_up_indices = pile_up[:,0]
max_event_num = np.unique(pile_up_indices).astype(int)
jet_masses = tt_int[:,6]
jet_pt = tt_int[:,7]
y_qlabel = {
    "mass": r'$\braket{m_{\mu}^{\text{jet}} - m_{0}^{\text{jet}}}$ [GeV]',
    "energy": r"$\braket{E_{\mu}^{\text{jet}} - E_{0}^{\text{jet}}}$ [GeV]",
    "pt": r"$\braket{p_{T,\mu}^{\text{jet}} - p_{T,0}^{\text{jet}}}$ [GeV]"
}
def mean_quantity_diff(params):
    """
    For a given number of pile_up events mu in MUs

    Sample random pile_ups

    Find <q_\mu^jet - q_0^jet> against \mu using quantity_diff

    Repeat for all jets (or a max number of jets)

    Find the average by dividing by the number of jets (not jet particles)
    """
    # max_jet_no = np.shape(np.unique(jet_data[:,0]))[0]
    # Store jet COMS
    jet_data, pile_up_data, mu, max_event_num, max_jet_no, jet_masses, jet_pt = params
    # for ind,mu in enumerate(MUs):
    print("Begin mu = ", mu)
    m_total = 0
    E_total = 0
    p_T_total = 0
    start_time = time.time()
    # print("ind", ind)
    # sys.exit(1)
    
    # TODO: mask
    event_IDS = np.random.choice(max_event_num, size = (max_jet_no, mu))
    # valid_event_IDS = event_IDS[np.isin(event_IDS, max_event_num)]        print(event_IDS)
    for jet_no in range(0, max_jet_no):
        selected_events = event_IDS[jet_no]
        # print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
        # selected_pile_ups = [select_event(pile_up_data, event_ID, filter=False) for event_ID in event_IDS]
        # selected_pile_ups = np.vstack(selected_pile_ups)
        # exit(1)
        selected_pile_ups = pile_up_data[np.isin(pile_up_indices, selected_events)]
        cd = select_event(jet_data, jet_no)
        m, E, p_T = quantity_diff(cd[:,0], cd[:,3],cd[:,4],cd[:,5], jet_masses[jet_no], jet_pt[jet_no], selected_pile_ups[:,3], selected_pile_ups[:,4], selected_pile_ups[:,5])
        m_total += m
        E_total += E
        p_T_total += p_T
    # Mean over number of jets
    end_time = time.time()
    print(f"Loop mu = {mu}: {end_time - start_time} seconds")
    m_total /= max_jet_no
    E_total /= max_jet_no
    p_T_total /= max_jet_no
    # print(m_total)
    # print(E_total)
    # print(p_T_total)
    print(f"End mu = {mu}")
    return m_total, E_total, p_T_total

# Array of jet etas and phis
# jet_centres = tt_int[:,4:6]
# tt_masked = [delta_R(jet_centre, tt)]
rlim = 50
MUs = np.linspace(0,rlim,rlim+1, dtype=np.int64)
max_jet_no = 1000
data_y = np.zeros((3, len(MUs)))

tasks = [
        (tt, pile_up, mu, pile_up_indices, max_jet_no, jet_masses, jet_pt) 
        for mu in MUs
    ]

with multiprocessing.Pool() as pool:
    results = pool.map(mean_quantity_diff, tasks)
pool.close()
pool.join()

for ind, (m_total, E_total, p_T_total) in enumerate(results):
    data_y[0][ind] += m_total
    data_y[1][ind] += E_total
    data_y[2][ind] += p_T_total

for name,data in zip(["Mass", "Energy", "pT"], data_y):
    fig  = plt.figure(figsize=(8,6))
    plt.tight_layout()
    plt.plot(MUs, data)
    plt.xlabel("$\mu$")
    plt.ylabel(y_qlabel[name.lower()])
    plt.xlim(0, np.max(MUs))
    plt.ylim(0 if 0 < np.min(data) else np.min(data), np.max(data))
    plt.savefig(f"{CWD}/data/plots/Mean_{name}_diff.pdf", format="pdf")
    plt.close()

end_time_global = time.time()
print(f"Globaal runtime: {end_time_global - start_time_global} seconds")

# mean_quantity_diff(tt, pile_up, mus, max_jet_no=1000)
