import numpy as np
from calculate_quantities import to_phi, pseudorapidity, p_magnitude
from data_loading import select_event

def foo_bar(jet_nos, pile_up_data, mu: int):
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
    combined_array = []
    for jet_no in jet_nos:
        event_IDS = np.random.choice(pile_up_data[:,0], size = mu).astype(int)
        print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
        selected_pile_ups = [select_event(pile_up_data, event_ID, filter=True) for event_ID in event_IDS]
        # This for loop writes the LIDs by taking the index as LID/
        # This means invalid pile_ups are counted and can be discarded
        for ind, pile in enumerate(selected_pile_ups):
            pile[:,0] = ind + 1
        selected_jet = select_event(tt, jet_no, filter=False)
        selected_jet = np.delete(selected_jet, [1,2], axis=1)
        X_pmag = p_magnitude(selected_jet[:,1:])
        X_etas = pseudorapidity(X_pmag, selected_jet[:,-1])
        X_phis = to_phi(selected_jet[:,1], selected_jet[:,2])
        num_rows = selected_jet.shape[0]
        new_column = np.full((1,num_rows), 0)
        selected_jet = np.insert(selected_jet, 1, new_column, axis=1)
        selected_jet = np.hstack((selected_jet, X_etas.reshape(-1, 1)))
        selected_jet  = np.hstack((selected_jet, X_phis.reshape(-1, 1)))

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
        X_pmag = p_magnitude(X[:,1:])
        X_etas = pseudorapidity(X_pmag, X[:,-1])
        X_phis = to_phi(X[:,1], X[:,2])
        # Append etas and phis to end of column
        X = np.hstack((X, X_etas.reshape(-1, 1)))
        X  = np.hstack((X, X_phis.reshape(-1, 1)))
        # Label these pile_ups with their jet IDs
        num_rows = X.shape[0]
        new_column = np.full((1,num_rows), jet_no)
        X = np.insert(X, 0, new_column, axis=1)
        combined_array.append(np.vstack((selected_jet, X)))
    # Merge all subarrays
    combined_array = np.vstack(combined_array)
    np.savetxt("data/combined.csv.gz", combined_array, delimiter=",", header="NID,LID,px,py,pz,eta,phi", comments="", fmt="%10.10f")
    return 0