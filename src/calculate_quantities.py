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