# Package imports
import numpy as np
# Local imports
from DDPMLHC.config import *

# === Read in data
print("0 :: Loading original data")
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
print("FINISHED loading data\n")

# Extract the first column from the pile_up array
first_column = pile_up[:, 0]

# Get the final number in the first column
final_number = first_column[-1]
print(f"Final number in the first column: {final_number}")

# Calculate the number of unique values in the first column
unique_vals = len(np.unique(first_column))
print(f"Number of unique values in the first column: {unique_vals}")

print( f"Percentage of non-empty pile-ups {(unique_vals/final_number)*100}" )

print("DONE ALL.")
