# Package imports
import numpy as np

def select_event(data, num, max_data_rows=10_000):
    """Select data with IDs num from data file. Assumes IDs are in the first (index = 0) column."""
    if (max_data_rows != None) and (max_data_rows <= num):
        raise ValueError(
            f"Requested jet {num} is not in data. Max jet number is {max_data_rows}"
        )
    return data[data[:, 0] == num]
# def extract_pile_ups(data, event_id):
#     """
#     Given num
#     """
#     num_rows = min(num_rows, data.shape[0])
#     random_indices = np.random.choice(data.shape[0], num_rows, replace=False)
#     event_ids = np.unique(data[random_indices][:,0])
#     #  = data[random_indices]
#     return data[data[:,0] == event_ids]
