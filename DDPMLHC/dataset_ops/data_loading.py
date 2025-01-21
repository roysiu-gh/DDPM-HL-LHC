# Local import
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import wrap_phi

# Package imports
import numpy as np

######################################################################################################

def select_event_deprecated(data, num, filter=False):
    """Select data with IDs num from data file. Assumes IDs are in the first (index = 0) column."""
    ### Could make faster by storing a local list of ID changes so lookup indices instead rather than comparing whole dataset every time? 
    return data[data[:, 0] == num]

######################################################################################################

class EventSelector:
    def __init__(self, data, mode="event"):
        self.data = np.asarray(data)
        self.ids = data[:, 0]
        if mode == "event":
            self.iterate_by_layer = False
        elif mode == "layer":
            self.iterate_by_layer = True
        else:
            raise ValueError(f"Argument mode must be 'event' or 'layer'.")
        
        # Precompute Indices
        self.unique_ids, self.start_indices, self.counts = np.unique(self.ids, return_index=True, return_counts=True)
        self.end_indices = self.start_indices + self.counts

        self.max_ID = self.data[-1, 0]

    def select_event(self, event_ID):
        idx = np.searchsorted(self.unique_ids, event_ID)
        if idx >= len(self.unique_ids):
            return np.empty((0, self.data.shape[1]))
        start = self.start_indices[idx]
        end = self.end_indices[idx]
        return self.data[start:end]
    


    # Treat this obj in same way as underlying data
    def __getitem__(self, key):
        return self.data[key]
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        return iter(self.data)
    def __array__(self, dtype=None):  # For numpy
        return np.array(self.data, dtype=dtype)
    
    # Access attributes from underlying np array if not def'd in this class
    def __getattr__(self, attr):
        try:
            return getattr(self.data, attr)
        except AttributeError:
            raise AttributeError(f"'EventSelector' object has no attribute '{attr}'")

class NoisyGenerator:
    def __init__(self, TTselector:EventSelector, PUselector:EventSelector, mu=0):
        self.tt = TTselector
        self.pu = PUselector
        self.mu = mu
        
        self._next_jetID = 0
        self._max_TT_no = self.tt.max_ID
        self._max_PU_no = self.pu.max_ID

        next(self)
    
    def __str__(self):
        return str(self.current_event)

    def __iter__(self):
        return self
    
    def __next__(self):
        self._build_next_noisy_event()
        self._calculate_event_level()
        self._next_jetID += 1
        if self._next_jetID == self._max_TT_no:
            raise StopIteration

    def _build_next_noisy_event(self):
        self.current_event = self.tt.select_event(self._next_jetID)  # Clean event only
