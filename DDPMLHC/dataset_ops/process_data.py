import numpy as np
from DDPMLHC.calculate_quantities import *
from DDPMLHC.config import *


### Move the next two funcs into diff file?

def unit_square_the_unit_circle(etas, phis):
    """Squeezes unit circle (eta^2 + phi^2 = 1) into unit square [0,1]x[0,1]."""
    etas /= 2
    phis /= 2
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
    combined = []
    running_length = 0
    # high_PU_no = pile_up_data[-1, 0]
    high_PU_no = pile_up_data.max_ID
    for jet_no in jet_nos:
        LID_index = 0
        jet_event = tt_data.select_event(jet_no)
        if jet_event.size == 0:
            print(f"Jet number {jet_no} not found. Perhaps MAX_DATA_ROWS is set. Breaking loop.")
            break
        px, py, pz = jet_event[:,3], jet_event[:,4], jet_event[:,5]
        axis = get_axis_eta_phi(px, py, pz) # Get jet centre to shift all particles eta/phi by this

        num_rows = jet_event.shape[0]
        LID_column = np.full((1, num_rows), 0) # Make array of zeros, LID for jet is always 0
        NID_column = np.full((1, num_rows), jet_no) # Make array of zeros
        jet_event = np.insert(jet_event, 0, LID_column, axis=1) # LID
        jet_event = np.insert(jet_event, 0, NID_column, axis=1) # NID
        # all = np.vstack((all, jet_event))
        jetpluspu = jet_event
        # combined.append(jet_event)
          # Last available ID is tot num of loaded pileup
        pu_nos = np.random.randint(low = 0, high = high_PU_no, size = mu, dtype=np.int32)
        # print(pu_nos)
        for pu_no in pu_nos:
            LID_index += 1
            pu_event = pile_up_data.select_event(pu_no)
            # print(pu_event)
            # print(pu_event.shape)
            # print(np.array([]).shape)
            #if pu_event == np.array([]): continue  # Skip if empty pile-up
            if pu_event.size == 0: continue  # Skip if empty pile-up
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
    output_filename = f"noisy_mu{mu}_raw.csv"
    output_filepath = f"{save_path}/{output_filename}"
    # ,
    np.savetxt(output_filepath, stacked, delimiter=",",  comments="",header="NID,LID,px,py,pz,d_eta,d_phi,pmag,p_T", fmt="%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f")
    # np.savetxt(output_file, out, delimiter=",")

    print(f"Made {int(stacked[-1,0])} events of mu = {mu} data.\n    Saved to {output_filename}.")



def calculate_event_level_quantities(mu, save_path, verbose=False, mask=True):
    # Load noisy data
    load_path = f"{CWD}/data/2-intermediate/noisy_mu{mu}_raw.csv"
    noisy_data = np.genfromtxt(
        load_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
    )

    # Do masking
    if mask:
        LIDs = noisy_data[:, 1].astype(int)
        d_etas = noisy_data[:, 5]
        d_phis = noisy_data[:, 6]
        dR2s = d_etas*d_etas + d_phis*d_phis
        noisy_data = noisy_data[ (LIDs == 0) | (dR2s < 1) ]  # First condition so 

    # Extract relevant columns
    NIDs = noisy_data[:, 0].astype(int)
    LIDs = noisy_data[:, 1].astype(int)
    pxs, pys, pzs = noisy_data[:, 2], noisy_data[:, 3], noisy_data[:, 4]

    NIDs_unique = np.unique(NIDs)
    event_enes, event_pxs, event_pys, event_pzs = calculate_four_momentum_massless(NIDs, pxs, pys, pzs)
    event_p2s = contraction(event_enes, event_pxs, event_pys, event_pzs)
    event_masses = np.sqrt(event_p2s)
    event_etas = pseudorapidity(event_enes, event_pzs)
    event_phis = to_phi(event_pxs, event_pys)
    event_pTs = to_pT(event_pxs, event_pys)

    # Print results for each event if verbose
    if verbose:
        for event_id in range(len(NIDs_unique)):
            print(f"NID: {event_id}, Total 4-Momenta: ["
                f"{event_enes[event_id]:.3f}, {event_pxs[event_id]:.3f}, "
                f"{event_pys[event_id]:.3f}, {event_pzs[event_id]:.3f}], "
                f"Mass: {event_masses[event_id]:.3f}")

    # Save results to CSV
    combined_array = np.array([
        NIDs_unique, event_pxs, event_pys, event_pzs, event_etas, event_phis, event_masses, event_pTs
    ]).T

    output_filename = f"noisy_mu{mu}_event_level.csv"
    output_filepath = f"{save_path}/{output_filename}"
    np.savetxt(
        output_filepath,
        combined_array,
        delimiter=",",
        header="event_id,px,py,pz,eta,phi,mass,p_T",
        comments="",
        fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
    )
    print(f"Collapsed {int(combined_array[-1,0])} events of mu = {mu} data to event-level.\n    Saved to {output_filename}.")
