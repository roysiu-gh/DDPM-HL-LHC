# Import constants
from DDPMLHC.config import *

# Package imports
import numpy as np

def select_event(data, num, filter=False):
    """Select data with IDs num from data file. Assumes IDs are in the first (index = 0) column."""
    ### Could make faster by storing a local list of ID changes so lookup indices instead rather than comparing whole dataset every time? 
    return data[data[:, 0] == num]
