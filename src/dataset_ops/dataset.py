# Import constants
from src.config import *

# Package imports
import csv
import time
import numpy as np
from pprint import pprint

# Local imports
from src.calculate_quantities import p_magnitude

CSV_FILE = ""
ETA_IDX = 5
PHI_IDX = 6

class DatasetAbstract(object):
    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self

class SingleNID(DatasetAbstract):
    def __init__(self, NID=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.NID = NID
        self._LID_ptr = 0

class SingleNIDFromCSV(SingleNID):
    def __init__(self, csv_file_path=CSV_FILE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.csv_file_path = csv_file_path

        self.file = None
        self.reader = None
        self.last_row = None

    # Context manager methods

    def __enter__(self):
        self.file = open(self.csv_file_path, mode="r")
        self.reader = csv.reader(self.file)

        try:
            self.last_row = next(self.reader)  # Header
            # self.last_row =   # Read in first data row and store
            self.last_row = [float(i) for i in next(self.reader)]  # Read in first data row and store, convert strs to nums
            while (self.last_row != None) and (self.last_row[0] != self.NID):
                # print(self.last_row)
                self.last_row = next(self.reader)
                self.last_row = [float(i) for i in self.last_row]  # Convert all from str to numbers
        except FileNotFoundError:
            print(f"File {self.csv_file_path} does not exist.")
            return None
        except StopIteration:
            raise IndexError("No such Noise ID found, greater than max")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        if exc_type:
            print(f"Exception type: {exc_type}")
            print(f"Exception value: {exc_value}")
            print("Traceback:", traceback)
        return False  # Propagate exception?

    # Iterator methods

    def __next__(self):
        if self.last_row is None:  # Reached EOF
            raise StopIteration
        if self.last_row[0] != self.NID:  # Exhausted this noisy event
            raise StopIteration

        out = {
            "enes" : [],
            "etas" : [],
            "phis" : [],
        }
        while self.last_row[1] == self._LID_ptr:
            pmag = p_magnitude((self.last_row[2]), self.last_row[3], self.last_row[4])
            out["enes"].append(pmag)
            out["etas"].append(self.last_row[ETA_IDX])
            out["phis"].append(self.last_row[PHI_IDX])
            try:
                self.last_row = next(self.reader)
                self.last_row = [float(i) for i in self.last_row]  # Convert all from str to numbers
            except StopIteration:
                self.last_row = None  # Only raise our StopIteration on the next call, as we still have data to return

        self._LID_ptr += 1
        return out
    
# csv_file_pathf"{CWD}/data/plots/bmaps/"

# with SingleNIDFromCSV(csv_file_path=f"{CWD}/data/test_combined.csv") as dataset:
#     print("Opened")
#     for i in dataset:
#         pprint(i)
#         print()
#         time.sleep(1)
#     print("Closed")

with SingleNIDFromCSV(csv_file_path=f"{CWD}/data/combined.csv") as dataset:
    print("Opened")
    for i in dataset:
        pprint(i)
        print()
        time.sleep(1)
    print("Closed")
