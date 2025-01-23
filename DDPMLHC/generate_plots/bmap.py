# Local imports
from DDPMLHC.config import *

# Package imports
import numpy as np
from PIL import Image

def save_to_bmap(vector, bins=BMAP_SQUARE_SIDE_LENGTH, jet_no="NONE", mu="NONE", save_path=None):
    if vector.size != bins*bins:
        raise RuntimeError(f"Vector not of size {bins}x{bins}={bins*bins}")
    if save_path is None:
        save_path = f"{CWD}/data/plots/bmaps"
    # print(vector)
    sd = np.std(vector)
    scaled_for_bmap = 255 * vector / (3 * sd)  # Scale to 3 std dev
    
    grid = scaled_for_bmap.reshape((bins, bins))  # Get grid, with scaling for bmap visualisation
    # print(grid)
    grid = np.clip(grid, 0, 255).astype(np.uint8)  # Remove OOB vals
    
    im = Image.fromarray(grid)
    filename = f"event_{int(jet_no)}_mu{int(mu)}"
    im.save(f"{save_path}/{filename}.png")
