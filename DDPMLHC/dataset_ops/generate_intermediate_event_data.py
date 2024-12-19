import numpy as np
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *

def process_noisy_data(mu, save_path):
    """
    Process noisy data for a given mu value.

    Parameters:
    mu (int): The mu value used for the computation and file naming.
    file_path (str): The path to the noisy data file.
    save_path (str): The directory where the results will be saved.

    Returns:
    None
    """
    # Load noisy data
    pileup_path = f"{CWD}/data/2-intermediate/noisy_mu{mu}.csv"
    noisy_data = np.genfromtxt(
        pileup_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
    )

    # Extract relevant columns
    NIDs = noisy_data[:, 0].astype(int)
    LIDs = noisy_data[:, 1].astype(int)
    pxs, pys, pzs = noisy_data[:, 2], noisy_data[:, 3], noisy_data[:, 4]
    etas = noisy_data[:, 5]
    phis = noisy_data[:, 6]
    enes = noisy_data[:, 7]
    pTs = to_pT(pxs, pys)

    p2s = contraction(enes, pxs, pys, pzs)
    masses = np.sqrt(p2s)

    # ===== Create Noisy Event Data ===== #

    NIDs_unique = np.unique(NIDs)
    event_enes, event_pxs, event_pys, event_pzs = calculate_four_momentum_massless(NIDs, pxs, pys, pzs)
    event_p2s = contraction(event_enes, event_pxs, event_pys, event_pzs)
    event_masses = np.sqrt(event_p2s)
    event_etas = pseudorapidity(event_enes, event_pzs)
    event_phis = to_phi(event_pxs, event_pys)
    event_pTs = to_pT(event_pxs, event_pys)

    # Print results for each event
    for event_id in range(len(NIDs_unique)):
        print(f"NID: {event_id}, Total 4-Momenta: ["
              f"{event_enes[event_id]:.3f}, {event_pxs[event_id]:.3f}, "
              f"{event_pys[event_id]:.3f}, {event_pzs[event_id]:.3f}], "
              f"Mass: {event_masses[event_id]:.3f}")

    # Save results to CSV
    combined_array = np.array([
        NIDs_unique, event_pxs, event_pys, event_pzs, event_etas, event_phis, event_masses, event_pTs
    ]).T

    output_file = f"{save_path}/noisy_event_stats_mu{mu}.csv"
    np.savetxt(
        output_file,
        combined_array,
        delimiter=",",
        header="event_id, px, py, pz, eta, phi, mass, p_T",
        comments="",
        fmt="%i, %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f"
    )
    print(f"Saved to {output_file}.")
