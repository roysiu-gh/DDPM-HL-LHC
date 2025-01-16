# Import constants
from DDPMLHC.config import *

# Package imports
import numpy as np

def select_event(data, num, filter=False):
    """Select data with IDs num from data file. Assumes IDs are in the first (index = 0) column."""
    return data[data[:, 0] == num]
