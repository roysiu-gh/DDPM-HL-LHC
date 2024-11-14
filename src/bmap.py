# Import constants
from config import *

# Package imports
import numpy as np
from PIL import Image

# Local imports
from data_loading import select_event
from calculate_quantities import COM_eta_phi, p_magnitude
from process_data import collection_crop_and_centre, unit_square_the_unit_circle

SAVE_PATH = f"{CWD}/data/plots/bmaps/"

def generate_random_points_in_unit_square(num_points=10):
    # Generate random x and y coordinates in the range [0, 1)
    energy = np.random.rand(num_points, 1).squeeze()
    points = np.random.rand(num_points, 2)
    return energy, points

def discretise_points(x, y, N=BMAP_SQUARE_SIDE_LENGTH):
    """Turn continuous points in the square [0,1]x[0,1] into discrete NxN grid."""
    discrete_x = np.floor(x * N)
    discrete_y = np.floor(y * N)
    discrete_x = discrete_x.astype(int)
    discrete_y = discrete_y.astype(int)
    return discrete_x, discrete_y

def scale_energy_for_visual(energies, N=BMAP_SQUARE_SIDE_LENGTH, verbose=False):
    """For the purpose of visualisation.
    Scales energies to 256 bits for printing to grayscale, up to a specified std dev.
    Assumes non-negative input.
    """
    # Calculate energy scale and scale values to 256
    SD = np.std(energies)
    scale = 256 / (3 * SD)  # Value above which we represent as full brightness (256)
    scaled_energies = np.floor(energies * scale)
    scaled_energies[scaled_energies > 256] = 256  # Maximise at 256
    scaled_energies = scaled_energies.astype(int)
    return scaled_energies

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

tt = np.genfromtxt(
    tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=1000
)

jet_no = 0

jet = select_event(tt, jet_no)[:, 3:6]
print(jet)
print(len(jet))

jet_px = tt[:, 3]
jet_py = tt[:, 4]
jet_pz = tt[:, 5]

energies = p_magnitude(jet_px, jet_py, jet_pz)
print("energies", energies)

centre = COM_eta_phi(jet)
print("centre", centre)

etas, phis = collection_crop_and_centre(jet, centre, R=1)
print("etas", etas)
print("phis", phis)
print(len(etas))
print(len(phis))

x, y = unit_square_the_unit_circle(etas, phis)

print("x", x)
print("y", y)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

scaled_energies = scale_energy_for_visual(energies)
scaled_energies = scale_energy_for_visual(np.log(energies))
scaled_x, scaled_y = discretise_points(x, y)
grid = convert_to_grid(scaled_energies, scaled_x, scaled_y)

print(grid)

im = Image.fromarray(grid)
im.save(f"{SAVE_PATH}/jet0.png")
