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

energies, points = generate_random_points_in_unit_square(100)

def scale_and_discretise(energies, points, N=BMAP_SQUARE_SIDE_LENGTH, verbose=False):
    """Scales energies to 256 bits for printing to grayscale, up to a specifies std dev.
    Discretises grid to integers of size NxN.
    RENAME
    Assumes non-negative input
    Todo: consider doing a rectangle instead of just squares.
    """
    if verbose:
        for e, p in zip(energies, points):
            print(e, p)
    
    # Turn continuous points into discrete NxN grid
    discrete_points = np.floor(points * N)

    # Calculate energy scale and scale values to 256
    SD = np.std(energies)
    scale = 256 / (3 * SD)  # Value above which we represent as full brightness (256)
    scaled_energies = np.floor(energies * scale)
    scaled_energies[scaled_energies > 256] = 256  # Maximise at 256

    discrete_points = discrete_points.astype(int)
    scaled_energies = scaled_energies.astype(int)

    return scaled_energies, discrete_points

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

scaled_energies, discrete_points = scale_and_discretise(energies, points)

# Some testing data
# scaled_energies = np.array( [210, 50, 90, 188] )
# discrete_points = np.array( [[0, 0], [3, 7], [3, 7], [4, 4]] )

for e, p in zip(scaled_energies, discrete_points):
    print(e, p)

grid = convert_to_grid(scaled_energies, discrete_points)

print(grid)

im = Image.fromarray(grid)
im.save(f"{SAVE_PATH}/test.png")
