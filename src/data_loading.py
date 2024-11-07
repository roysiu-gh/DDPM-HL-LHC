# Package imports
import numpy as np

def select_jet(data, num, max_data_rows=10_000):
    """Select data with jets #num from data file"""
    if (max_data_rows != None) and (max_data_rows <= num):
        raise ValueError(
            f"Requested jet {num} is not in data. Max jet number is {max_data_rows}"
        )
    return data[data[:, 0] == num]

def random_rows_from_csv(data, num_rows):
    num_rows = min(num_rows, data.shape[0])
    random_indices = np.random.choice(data.shape[0], num_rows, replace=False)
    random_rows = data[random_indices]
    return random_rows
