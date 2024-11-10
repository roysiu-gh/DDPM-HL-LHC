import os
import numpy as np
from PIL import Image

NN = 16
NX = NN
NY = NN

CWD = os.getcwd()
SAVE_PATH = f"{CWD}/data/plots/bmaps/"

def generate_random_points_in_unit_square(num_points=10):
    # Generate random x and y coordinates in the range [0, 1)
    energy = np.random.rand(num_points, 1)
    points = np.random.rand(num_points, 2)
    return energy, points

energies, points = generate_random_points_in_unit_square(100)

for e, p in zip(energies, points):
    print(e, p)

scaled_points = (points * NN) // 1
scaled_points = scaled_points.astype(int)
scaled_energies = (energies * 100) // 1
scaled_energies = scaled_energies.astype(int)

for e, p in zip(scaled_energies, scaled_points):
    print(e, p)


# scaled_energies = np.array( [3, 5, 9, 8] )
# scaled_points = np.array( [[0, 0], [3, 7], [3, 7], [4, 4]] )

for e, p in zip(scaled_energies, scaled_points):
    print(e, p)

grid = np.zeros((NX, NY), dtype=np.uint8)  # Need np.uint8 for Image.fromarray()

for e, p in zip(scaled_energies, scaled_points):
    x_coord = p[0]
    y_coord = p[1]
    print(f"Adding {e} to x {x_coord} y {y_coord}")
    grid[x_coord, y_coord] += e

print(grid)

im = Image.fromarray(grid)
im.save(f"{SAVE_PATH}/test.png")
