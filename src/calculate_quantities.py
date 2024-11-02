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