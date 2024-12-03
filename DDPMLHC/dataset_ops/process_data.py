import numpy as np
import multiprocessing
from DDPMLHC.calculate_quantities import to_phi, pseudorapidity, p_magnitude,get_axis_eta_phi, centre_on_jet
from DDPMLHC.dataset_ops.data_loading import select_event
def unit_square_the_unit_circle(etas, phis):
    """Squeezes unit circle (eta^2 + phi^2 = 1) into unit square [0,1]x[0,1]."""
    etas /= 4
    phis /= 4
    etas += 0.5
    phis += 0.5
    return etas, phis

def wrap_phi(phi_centre, phis, R=1):
    # If near top edge
    shifted_phis = phis
    if abs(phi_centre - np.pi) < R:
        mask_for_bottom_half = phis < 0  # Only shift for particles below line phi=0
        shifts = 2 * np.pi * mask_for_bottom_half
        shifted_phis += shifts
    # If near bottom edge
    if abs(phi_centre + np.pi) < R:
        mask_for_top_half = phis > 0  # Only shift for particles above line phi=0
        shifts = 2 * np.pi * mask_for_top_half
        shifted_phis -= shifts
    return shifted_phis


def combined_jet(args):
    """
    This function [CHANGE NAME PLEASE FFS] takes in an array of jet IDs, pile_ups and a mu-value and performs the following:
    
    1) Randomly selects pile_up event IDs, the number of randomly selected IDs corresponds to mu
    2) Selects all pile_up particles corresponding to the randomly chosen IDs. IDs which do not exist are kept for indexing LID and discarded later on
    3) Removes PDGID, charge
    4) Calculates etas, phis and appends them as columns
    5) Inserts jet IDs at the beginning
    6) Writes final combined array to data/combined.csv.gz

    Parameters
    ----------
    jet_nos: List[int] or ndarray
        1D List or 1D NumPy array of jet IDs.
    pile_up_data: ndarray
        Complete 2D NumPy of pile_up_data
    mu: int
        The number of pile_up IDs to use in noise
    
    Returns
    -------
    0 for successful completion
    """
    # combined_array = []
    jet_no, tt_data, pile_up_data, mu = args
    max_event_id = np.max(pile_up_data[:,0])
    
    # for jet_no in jet_nos:
    event_IDS = np.random.randint(low = 0, high = max_event_id, size = mu, dtype=np.int32)
    # print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
    selected_pile_ups = [select_event(pile_up_data, event_ID, filter=True) for event_ID in event_IDS]
    # This for loop writes the LIDs by taking the index as LID/
    # This means invalid pile_ups are counted and can be discarded
    for ind, pile in enumerate(selected_pile_ups):
        if isinstance(pile[0], np.ndarray) == False:
            pile[0] = ind + 1
        else:
            pile[:,0] = ind + 1
    # exit(1)
    selected_jet = select_event(tt_data, jet_no, filter=False)
    selected_jet = np.delete(selected_jet, [1,2], axis=1)

    selected_jet_px = selected_jet[:, 1]
    selected_jet_py = selected_jet[:, 2]
    selected_jet_pz = selected_jet[:, 3]
    centre = get_axis_eta_phi([selected_jet_px, selected_jet_py, selected_jet_pz])
    # print(centre)
    X_pmag = p_magnitude(selected_jet_px, selected_jet_py, selected_jet_pz)
    X_etas = pseudorapidity(X_pmag, selected_jet_pz)
    X_phis = to_phi(selected_jet_px, selected_jet_py)
    X_phis = wrap_phi(centre[1], X_phis) 
    # print(X_etas)
    new_centre, etas, phis = centre_on_jet(centre, X_etas, X_phis)
    # print(phis)
    # print(X_phis)
    # exit(1)
    num_rows = selected_jet.shape[0]
    new_column = np.full((1,num_rows), 0)
    selected_jet = np.insert(selected_jet, 1, new_column, axis=1)
    selected_jet = np.hstack((selected_jet, etas.reshape(-1, 1)))
    selected_jet  = np.hstack((selected_jet, phis.reshape(-1, 1)))
    selected_jet  = np.hstack((selected_jet, X_pmag.reshape(-1, 1)))

    # Stack arrays on top of each other
    selected_pile_ups = np.vstack(selected_pile_ups)
    # Clearly, an invalid particle has completely zero momentum in all components (violates conservation of energy)
    # Therefore this masks out all the rows where the corresponding sample has zero in all 3 components of p
    # Equivalent to ensuring the L2 norm is 0 iff components are not zero since a norm is semi-positve definite
    zero_p_mask = ~((selected_pile_ups[:, 3] == 0) & (selected_pile_ups[:, 4] == 0) & (selected_pile_ups[:, 5] == 0))
    selected_pile_ups = selected_pile_ups[zero_p_mask]

    # Delete PGDID and charge columns
    X = np.delete(selected_pile_ups, [1,2], axis=1)

    # Now momenta start at 2nd column
    # Select p for calculations
    X_momenta = X[:,1:]
    
    X_px = X[:, 1]
    X_py = X[:, 2]
    X_pz = X[:, 3]

    X_pmag2 = p_magnitude(X_px, X_py, X_pz)
    X_etas = pseudorapidity(X_pmag2, X_pz)
    X_phis = to_phi(X_px, X_py)
    X_phis = wrap_phi(centre[1], X_phis) 
    _, etas, phis = centre_on_jet(centre, X_etas, X_phis)
    
    # Append etas and phis, pmags to end of column
    X = np.hstack((X, etas.reshape(-1, 1)))
    X  = np.hstack((X, phis.reshape(-1, 1)))
    X  = np.hstack((X, X_pmag2.reshape(-1, 1)))
    # Label these pile_ups with their event IDs
    num_rows = X.shape[0]
    new_column = np.full((1,num_rows), jet_no)
    X = np.insert(X, 0, new_column, axis=1)
    combined_array = np.vstack((selected_jet, X))
    # # Merge all subarrays

    return combined_array

def write_combined_csv(jet_nos, tt_data, pile_up_data, mu):
    tasks = [
        (jet_no, tt_data, pile_up_data, mu) 
        for jet_no in jet_nos
    ]
    with multiprocessing.Pool() as pool:
        results = pool.map(combined_jet, tasks)
    pool.close()
    pool.join()
    combined_array = np.vstack(results)
    np.savetxt(f"data/noisy_mu{mu}.csv.gz", combined_array, delimiter=",", header="NID,LID,px,py,pz,d_eta,d_phi,pmag", comments="", fmt="%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f")

