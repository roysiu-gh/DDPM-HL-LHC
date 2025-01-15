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
    combined = []
    running_length = 0
    for jet_no in jet_nos:
        LID_index = 0
        jet_event = select_event(tt_data, jet_no, filter=False)
        axis = get_axis_eta_phi([jet_event[:,3], jet_event[:,4], jet_event[:,5]]) # Get jet centre to shift all particles eta/phi by this

        num_rows = jet_event.shape[0]
        LID_column = np.full((1, num_rows), 0) # Make array of zeros, LID for jet is always 0
        NID_column = np.full((1, num_rows), jet_no) # Make array of zeros
        jet_event = np.insert(jet_event, 0, LID_column, axis=1) # LID
        jet_event = np.insert(jet_event, 0, NID_column, axis=1) # NID
        # all = np.vstack((all, jet_event))
        jetpluspu = jet_event
        # combined.append(jet_event)
        pu_nos = np.random.randint(low = 0, high = PILEUP_NUM, size = mu, dtype=np.int32)
        # print(pu_nos)
        for pu_no in pu_nos:
            LID_index += 1
            pu_event = select_event(pile_up_data, pu_no, filter=False)
            if pu_event is None: continue
            num_rows = pu_event.shape[0]
            LID_column = np.full((1, num_rows), LID_index) # Make array of LIDs
            NID_column = np.full((1, num_rows), jet_no) # Make array of NIDs
            pu_event = np.insert(pu_event, 0, LID_column, axis=1) # LID
            pu_event = np.insert(pu_event, 0, NID_column, axis=1) # NID

            # all = np.vstack((all, pu_event))
            # temp.append(pu_event)
            jetpluspu = np.vstack((jetpluspu,pu_event))
        # running_length += len(jetpluspu)
        # current_length = len(jetpluspu)
        pxs, pys, pzs = jetpluspu[:, 5], jetpluspu[:, 6], jetpluspu[:, 7]
    
        enes = p_magnitude(pxs, pys, pzs)
        pTs = to_pT(pxs, pys)
        etas = pseudorapidity(enes, pzs)

        # Combine the following 2 ops later to optimise
        phis = to_phi(pxs, pys)
        phis = wrap_phi(axis[1], phis)
        origin, eta_c, phi_c = centre_on_jet(axis, etas, phis)
        jetpluspu = np.hstack((jetpluspu, eta_c.reshape(-1, 1)))
        jetpluspu = np.hstack((jetpluspu, phi_c.reshape(-1, 1)))
        jetpluspu = np.hstack((jetpluspu, enes.reshape(-1, 1)))
        jetpluspu = np.hstack((jetpluspu, pTs.reshape(-1, 1)))
        combined.append(jetpluspu)
        # for current_mu in range(mu):
    stacked = np.vstack(combined)
    # pxs, pys, pzs = stacked[:, 5], stacked[:, 6], stacked[:, 7]
    # Delete ID,PDGID,CHARGE columns
    stacked = np.delete(stacked, [2,3,4], axis=1)
    # NIDs = stacked[:, 0].astype(int)
    # LIDs = stacked[:, 1].astype(int)
    
    

    #############TURN INTO DELTAS

    # out = np.column_stack((NIDs, LIDs, pxs, pys, pzs, eta_c, phi_c, enes, pTs))
    # print(out)
    output_file = f"{save_path}/noisy_mu{mu}.csv"
    # ,
    np.savetxt(output_file, stacked, delimiter=",",  comments="",header="NID,LID,px,py,pz,d_eta,d_phi,pmag", fmt="%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f")
    # np.savetxt(output_file, out, delimiter=",")



