import numpy as np
from DDPMLHC.calculate_quantities import to_phi, pseudorapidity, p_magnitude,get_axis_eta_phi, centre_on_jet, to_pT
from DDPMLHC.dataset_ops.data_loading import select_event
from DDPMLHC.config import *

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

############################################################################################

def make_noisy_data(jet_nos, tt_data, pile_up_data, mu, save_path="data"):

    #NID,LID,px,py,pz,d_eta,d_phi,pmag
    all = []

    for jet_no in jet_nos:
        jet_event = select_event(tt_data, jet_no, filter=False)

        num_rows = jet_event.shape[0]
        LID_column = np.full((1, num_rows), 0) # Make array of zeros
        NID_column = np.full((1, num_rows), jet_no) # Make array of zeros
        jet_event = np.insert(jet_event, 0, LID_column, axis=1) # LID
        jet_event = np.insert(jet_event, 0, NID_column, axis=1) # NID

        # all = np.vstack((all, jet_event))
        all.append(jet_event)

        for current_mu in range(mu):
            pu_no = np.random.randint(low = 0, high = PILEUP_NUM, dtype=np.int32)
            pu_event = select_event(pile_up_data, pu_no, filter=False)

            if pu_event is None: continue

            num_rows = pu_event.shape[0]
            LID_column = np.full((1, num_rows), 0) # Make array of zeros
            NID_column = np.full((1, num_rows), jet_no) # Make array of zeros
            pu_event = np.insert(pu_event, 0, LID_column, axis=1) # LID
            pu_event = np.insert(pu_event, 0, NID_column, axis=1) # NID

            # all = np.vstack((all, pu_event))
            all.append(pu_event)
    
    stacked = np.vstack(all)
    NIDs = stacked[:, 0].astype(int)
    LIDs = stacked[:, 1].astype(int)
    pxs, pys, pzs = stacked[:, 4], stacked[:, 5], stacked[:, 6]

    enes = p_magnitude(pxs, pys, pzs)
    etas = pseudorapidity(enes, pzs)

    # Combine the following 2 ops later to optimise
    phis = to_phi(pxs, pys)
    pTs = to_pT(pxs, pys)

    #############TURN INTO DELTAS

    out = np.hstack((NIDs, LIDs, pxs, pys, pzs, etas, phis, enes, pTs))

    output_file = f"{save_path}/noisy_mu{mu}.csv"
    
    # np.savetxt(output_file, combined_array, delimiter=",", header="NID,LID,px,py,pz,d_eta,d_phi,pmag", comments="", fmt="%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f")
    np.savetxt(output_file, out, delimiter=",")



