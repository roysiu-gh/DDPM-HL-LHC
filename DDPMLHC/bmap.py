# Import constants
from DDPMLHC.config import *

# Package imports
import numpy as np
from PIL import Image

# Local imports
from DDPMLHC.dataset_ops.data_loading import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import unit_square_the_unit_circle, wrap_phi

SAVE_PATH = f"{CWD}/data/plots/bmaps/"

def generate_random_points_in_unit_square(num_points=10):
    # Generate random x and y coordinates in the range [0, 1)
    energy = np.random.rand(num_points, 1).squeeze()
    points = np.random.rand(num_points, 2)
    return energy, points

def discretise_points(x, y, N=BMAP_SQUARE_SIDE_LENGTH):
    """Turn continuous points in the square [0,1]x[0,1] into discrete NxN grid."""
    discrete_x = np.floor(x * N).astype(int)
    discrete_y = np.floor(y * N).astype(int)
    return discrete_x, discrete_y

def scale_energy_for_visual(energies, log_scale=False):
            """Scale energies to 8-bit values (0-255)."""
            if log_scale:
                energies = np.log(energies + 1)  # +1 to handle zeros
            
            # Scale to 3 standard deviations
            sd = np.std(energies)
            mean = np.mean(energies)
            scaled = 255 * (energies - mean) / (3 * sd) + 128
            
            return scaled

def convert_to_grid(energies, x, y, N=BMAP_SQUARE_SIDE_LENGTH, verbose=False):

    grid = np.zeros((N, N), dtype=np.uint8)  # Need np.uint8 for Image.fromarray()

    for ene, x_coord, y_coord in zip(energies, x, y):
        if verbose: print(f"Adding {ene} to x {x_coord} y {y_coord}")
        try:
            grid[x_coord, y_coord] += ene
        except IndexError:
            raise ValueError("Grid coordinates are not integers.")
    
    return grid


##########################################################################################

if __name__ == "__main__":

    MAX_DATA_ROWS = 10000

    tt = np.genfromtxt(
        TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
    )
    pile_up = np.genfromtxt(
        PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
    )
    tt = EventSelector(tt)
    pile_up = EventSelector(pile_up)

    jet_no = 0
    jet = tt.select_event(jet_no)
    jet_px = jet[:, 3]
    jet_py = jet[:, 4]
    jet_pz = jet[:, 5]
    print(len(jet))
    print(len(jet_px))
    # print(jet)

    energies = p_magnitude(jet_px, jet_py, jet_pz)
    print(len(energies))
    centre = get_axis_eta_phi(jet_px, jet_py, jet_pz)
    print("centre", centre)
    p_mag = p_magnitude(jet_px, jet_py, jet_pz)
    etas = pseudorapidity(p_mag, jet_pz)
    phis = to_phi(jet_px, jet_py)
    phis = wrap_phi(centre[1], phis)
    _, etas, phis = delta_R(centre, jet_px, jet_py, jet_pz, etas, phis, boundary=1)
    _, etas, phis = centre_on_jet(centre, etas, phis)

    x, y = unit_square_the_unit_circle(etas, phis)
    # print("x", x)
    # print("y", y)

    scaled_energies = scale_energy_for_visual(energies, log_scale=False)
    x_discrete, y_discrete = discretise_points(x, y)
    grid = convert_to_grid(scaled_energies, x_discrete, y_discrete, verbose=False)

    print(grid)

    im = Image.fromarray(grid)
    im.save(f"{SAVE_PATH}/jet0.png")
