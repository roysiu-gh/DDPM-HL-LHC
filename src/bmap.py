import os
import numpy as np
from PIL import Image

BMAP_SQUARE_SIDE_LENGTH = 16

CWD = os.getcwd()
SAVE_PATH = f"{CWD}/data/plots/bmaps/"

def generate_random_points_in_unit_square(num_points=10):
    # Generate random x and y coordinates in the range [0, 1)
    energy = np.random.rand(num_points, 1).squeeze()
    points = np.random.rand(num_points, 2)
    return energy, points

def discretise_points(points, N=BMAP_SQUARE_SIDE_LENGTH):
    """Turn continuous points into discrete NxN grid."""
    discrete_points = np.floor(points * N)
    discrete_points = discrete_points.astype(int)
    return discrete_points

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

def convert_to_grid(energies, points, N=BMAP_SQUARE_SIDE_LENGTH, verbose=False):

    grid = np.zeros((N, N), dtype=np.uint8)  # Need np.uint8 for Image.fromarray()

    for ene, point in zip(energies, points):
        x_coord = point[0]
        y_coord = point[1]
        if verbose: print(f"Adding {ene} to x {x_coord} y {y_coord}")
        try:
            grid[x_coord, y_coord] += ene
        except IndexError:
            raise ValueError("Grid coordinates are not integers.")
    
    return grid

energies, points = generate_random_points_in_unit_square(100)
scaled_energies = scale_energy_for_visual(energies)
discrete_points = discretise_points(points)
grid = convert_to_grid(scaled_energies, discrete_points)

print(grid)

im = Image.fromarray(grid)
im.save(f"{SAVE_PATH}/test.png")
