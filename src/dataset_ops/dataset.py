# Import constants
from src.config import *

# Package imports
import csv
import time

# Local imports

CSV_FILE = ""
ETA_IDX = 5
PHI_IDX = 6

class DatasetAbstract(object):
    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self

class SingleNIDFromCSV(DatasetAbstract):
    def __init__(self, NID=0, csv_file_path=CSV_FILE) -> None:
        self.csv_file_path = csv_file_path
        self.NID = NID

        self.file = None
        self.reader = None
        self._LID_ptr = 0
        self.last_row = None

    # Context manager methods

    def __enter__(self):
        self.file = open(self.csv_file_path, mode="r")
        self.reader = csv.reader(self.file)

        self.last_row = next(self.reader)  # Header
        self.last_row = next(self.reader)  # Read in first data row and store

        try:
            while (self.last_row != None) and (self.last_row[0] != str(self.NID)):
                # print(self.last_row)
                self.last_row = next(self.reader)
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
        if int(self.last_row[0]) != self.NID:  # Exhausted this noisy event
            raise StopIteration
        
        # print(self.last_row)
        # print(self.NID)
        # print(self._LID_ptr)

        enes, etas, phis = [], [], []
        while int(self.last_row[1]) == self._LID_ptr:
            print("juan", self.last_row)
            enes.append(None) #######!!!!!
            etas.append(self.last_row[ETA_IDX])
            phis.append(self.last_row[PHI_IDX])
            try:
                self.last_row = next(self.reader)
            except StopIteration:
                self.last_row = None  # Only raise our StopIteration on the next call, as we still have data to return

        self._LID_ptr += 1
        return enes, etas, phis
    
# csv_file_pathf"{CWD}/data/plots/bmaps/"

with SingleNIDFromCSV(csv_file_path=f"{CWD}/src/test_dataset.csv") as dataset:
    print("Opened")
    for enes, etas, phis in dataset:
        print("enes", enes)
        print("etas", etas)
        print("phis", phis)
        time.sleep(1)
        print()
    print("Closed")
