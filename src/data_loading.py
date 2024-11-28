# Import constants
from config import *

# Package imports
import numpy as np

def select_event(data, num, filter=False, max_data_rows=MAX_DATA_ROWS):
    """Select data with IDs num from data file. Assumes IDs are in the first (index = 0) column."""
    if filter == False and (max_data_rows != None) and (max_data_rows <= num):
        raise ValueError(
            f"Requested jet {num} is not in data. Max rows is {max_data_rows} and {num} was not found."
        )
    # Find unique indices
    indices = np.unique(data[:,0])
    if (not np.isin(num, indices)):
        print(f"Warn: index was not found in data file. Returning array with ID {num} (at 0th index) of same shape as 1 sample in data, with zeroes everywhere else.")
        zero_arr = np.zeros_like(data[0])
        zero_arr[0] = -1
        return zero_arr
    return data[data[:, 0] == num]


