import numpy as np

def p_magnitude(p):
    """
    p is a 2D NumPy array where each element is a particle's 3-momentum

    This function simply calculates and returns the magnitude of the momenta of each particle and returns it as a 1D NumPy array
    """
    return np.linalg.norm(p, axis=1)


def pseudorapidity(p_mag, p_z):
    """
    p_mag is a 1D NumPy array where each element is the magnitude of momentum for a particle
    p_z is the z-component of momentum of a particle (the component along the beam-axis)

    this function calculates the pseudorapidity and returns it as a 1D NumPy array
    https://en.wikipedia.org/wiki/Pseudorapidity
    """
    if np.shape(p_mag) != np.shape(p_z):
        raise ValueError("Error: p_mag shape not equal to p_z shape")
    return np.arctanh(p_z / p_mag)


def to_phi(p_x, p_y):
    """
    p_x, p_y are 1D NumPy arrays of x and y momenta respectively, where each element corresponds to a particle.

    This function finds the angle phi (radians) from the 2 components of transverse momentum p_x,  p_y using arctan(p_y/p_x)
    """
    if np.shape(p_x) != np.shape(p_y):
        raise ValueError("Error: p_x shape not equal to p_y shape")
    return np.arctan2(p_y, p_x)

def calculate_four_momentum_massless(jet_ids, px, py, pz):
    """Calculate the total 4-momentum of jets. Massless limit. Natural units."""
    if not (len(jet_ids) == len(px) == len(py) == len(pz)):
        raise ValueError("All input arrays must have same length.")
    
    max_jet_id = int( np.max(jet_ids) )
    total_four_momenta = np.array([np.zeros(4) for _ in range(max_jet_id + 1)])

    # Calculate total 4mmtm for each jet
    for jet_id, px_val, py_val, pz_val in zip(jet_ids, px, py, pz):
        energy = np.linalg.norm([px_val, py_val, pz_val])
        four_mmtm = np.array([energy, px_val, py_val, pz_val])
        total_four_momenta[int(jet_id)] += four_mmtm

    return total_four_momenta

def contraction(vec):
    """Calculate the contractions pf an array of 4vecs."""
    time_like_0 = vec[:, 0]
    space_like_1 = vec[:, 1]
    space_like_2 = vec[:, 2]
    space_like_3 = vec[:, 3]
    return time_like_0**2 - (space_like_1**2 + space_like_2**2 + space_like_3**2)

def COM_eta_phi(p):
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
    eta = pseudorapidity(jet_mag, total_p[2])
    phi = to_phi(total_p[0], total_p[1])
    return eta, phi

def delta_R(jet_centre, jet_data, boundary=1.0):
    """
    This function takes in particle information, and removes particles whose \Delta R(eta,phi) > 1.0 and returns all the others.

    Parameters
    ----------
    centre : tuple of (float,float)
        The jet beam axis. 2-tuple in the form (eta,phi) used for calculating \Delta R
    jet_no : int
        Which jet to select from data
    data: ndarray
        2D dataset containing particle information.
    boundary: float, default = 1.0
        The maximum \Delta R for which particles with a larger value will be cut off.
    Returns
    ----------
    bounded_data: ndarray
        2D dataset of particle information, with particles whose \Delta R is greater than `boundary` removed.
    etas: ndarray
        1D dataset of particle etas, with particles whose \Delta R is greater than `boundary` removed.
    phis: ndarray
        1D dataset of particle phis, with particles whose \Delta R is greater than `boundary` removed.
    """
    # Calculate eta, phi of every particle in data
    p_mag = p_magnitude(jet_data[:,3:])
    etas = pseudorapidity(p_mag, jet_data[:,5])
    phis = to_phi(jet_data[:,3], jet_data[:,4])
    # Calculate the values of Delta R for each particle
    delta_eta= (etas - jet_centre[0])
    delta_phi = (phis - jet_centre[1])
    crit_R = np.sqrt(delta_eta*delta_eta + delta_phi*delta_phi)
    bounded_data = jet_data[crit_R <= boundary]
    return bounded_data, etas[crit_R <= boundary], phis[crit_R <= boundary]
