import numpy as np

def p_magnitude(px, py, pz):
    """
    This function simply calculates and returns the magnitude of the momenta of each particle and returns it as a 1D NumPy array
    Parameters
    ----------
    px,py,pz: ndarray
        1D arrays of x-momentum, y-momentum and z-momentum of particles.
         
    Returns
    ------
    p_mag: ndarray
        1D array of same shape containing 3-momentum magnitude 
    """
    # np.linalg.norm([px_val, py_val, pz_val]) ?
    return np.sqrt(px*px + py*py + pz*pz)


def pseudorapidity(p_mag, p_z):
    """
    p_mag is a 1D NumPy array where each element is the magnitude of momentum for a particle
    p_z is the z-component of momentum of a particle (the component along the beam-axis)

    this function calculates the pseudorapidity and returns it as a 1D NumPy array
    https://en.wikipedia.org/wiki/Pseudorapidity
    """
    if np.shape(p_mag) != np.shape(p_z):
        raise ValueError(f"Error: p_mag shape {np.shape(p_mag)} not equal to p_z shape {np.shape(p_z)}.")
    return np.arctanh(p_z / p_mag)


def to_phi(p_x, p_y):
    """
    p_x, p_y are 1D NumPy arrays of x and y momenta respectively, where each element corresponds to a particle.

    This function finds the angle phi (radians) from the 2 components of transverse momentum p_x,  p_y using arctan(p_y/p_x)
    """
    if np.shape(p_x) != np.shape(p_y):
        raise ValueError(f"Error: p_x shape {np.shape(p_x)} not equal to p_y {np.shape(p_y)} shape")
    return np.arctan2(p_y, p_x)

def to_pT(p_x, p_y):
    """Calculate transverse momentum."""
    if np.shape(p_x) != np.shape(p_y):
        raise ValueError(f"Error: p_x shape {np.shape(p_x)} not equal to p_y {np.shape(p_y)} shape")
    return np.sqrt(p_x**2 + p_y**2)


def calculate_four_momentum_massless(event_ids, px, py, pz, mask=True):
    """Calculate the total 4-momentum of jets. Massless limit. Natural units. Optional masking to remove Delta R > 1 particles."""
    if not (len(event_ids) == len(px) == len(py) == len(pz)):
        raise ValueError(f"All input arrays must have same length. Got {len(event_ids)}, {len(px)}, {len(py)}, {len(pz)}.")
    
    max_jet_id = int( event_ids[-1] )

    event_ene = np.zeros(max_jet_id + 1)
    event_px = np.zeros(max_jet_id + 1)
    event_py = np.zeros(max_jet_id + 1)
    event_pz = np.zeros(max_jet_id + 1)

    for event_id, px_val, py_val, pz_val in zip(event_ids, px, py, pz):
        ene_val = p_magnitude(px_val, py_val, pz_val)

        event_ene[int(event_id)] += ene_val
        event_px[int(event_id)] += px_val
        event_py[int(event_id)] += py_val
        event_pz[int(event_id)] += pz_val

    return event_ene, event_px, event_py, event_pz

def contraction(time_like_0, space_like_1, space_like_2, space_like_3):
    """Calculate the contractions."""
    return time_like_0**2 - (space_like_1**2 + space_like_2**2 + space_like_3**2)

def get_axis_eta_phi(p):
    """Find COM of a collection of particles.
    Uses the massless limit (m << E).

    Parameters
    ----------
    p : ndarray
        2D array of floats containing particle momenta [px, py, pz].
    
    Returns
    ----------
    eta,phi: float,float
        Location of jet axis in eta-phi space.
    """
    total_p = np.sum(p, axis=0)
    jet_mag = np.linalg.norm(total_p)
    if jet_mag == 0:
        raise ValueError("Jet magnitude is zero. Input is invalid for calculations.")
    eta = pseudorapidity(jet_mag, total_p[2])
    phi = to_phi(total_p[0], total_p[1])
    return eta, phi

def centre_on_jet(centre, eta, phi):
    """ 
    Centres the jet axis on (0,0) and shifts all particles by this displacement.
    """
    return (0,0), eta - centre[0], phi - centre[1]

def delta_R(centre, px, py, pz, etas, phis, boundary=1.0):
    """
    This function takes in particle information, and removes particles whose \Delta R(eta,phi) > 1.0 and returns all the others.

    Parameters
    ----------
    centre : tuple of (float,float)
        The jet beam axis. 2-tuple in the form (eta,phi) used for calculating \Delta R
    jet_no : int
        Which jet to select from data
    eta: ndarray
        1D dataset containing particle \eta s.
    phi: ndarray
        1D dataset containing particle \phi s.
    boundary: float, default = 1.0
        The maximum \Delta R for which particles with a larger value will be cut off.
    Returns
    ----------
    bounded_momenta: ndarray
        2D dataset of particle momenta, with particles whose \Delta R is greater than `boundary` removed.
    etas: ndarray
        1D dataset of particle etas, with particles whose \Delta R is greater than `boundary` removed.
    phis: ndarray
        1D dataset of particle phis, with particles whose \Delta R is greater than `boundary` removed.
    """
    # Calculate the values of Delta R for each particle
    # If jet axis centred on (0,0), this just evaluates to the etas, phis. See
    # centre_on_jet(centre, eta, phi)
    delta_eta= (etas - centre[0])
    delta_phi = (phis - centre[1])
    crit_R = np.sqrt(delta_eta*delta_eta + delta_phi*delta_phi)
    bounded_momenta = np.array([px[crit_R <= boundary], py[crit_R <= boundary], pz[crit_R <= boundary]])
    return bounded_momenta, etas[crit_R <= boundary], phis[crit_R <= boundary]
