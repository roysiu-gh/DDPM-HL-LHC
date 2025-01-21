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

        self.event_level = {}  # Dict for event-level quantities
        self.column_indices = {  # Define column index mapping for getters
            "NID": 0,
            "LID": 1,
            "px": 2,
            "py": 3,
            "pz": 4,
            "d_eta": 5,
            "d_phi": 6,
            "mass": 7,
            "p_T": 8,
        }

        next(self)
    
    def __str__(self):
        out = f"""\
Current - Jet {self._next_jetID-1} with mu={self.mu}
    Event-level quantities:
        mass = {self.event_level["mass"]}
        eta  = {self.event_level["eta"]}
        phi  = {self.event_level["phi"]}
        p_T  = {self.event_level["pT"]}
        """
        return out
    
    def __repr__(self):
        return str(self.current_event)

    def __iter__(self):
        return self
    
    def __next__(self):
        self._build_next_noisy_event()
        self._mask()
        self._calculate_event_level()
        self._next_jetID += 1
        if self._next_jetID == self._max_TT_no:
            raise StopIteration

    def _build_next_noisy_event(self):
        jet_event = self.tt.select_event(self._next_jetID)
        jet_px, jet_py, jet_pz = jet_event[:,3], jet_event[:,4], jet_event[:,5]
        self.jet_axis = get_axis_eta_phi(jet_px, jet_py, jet_pz)

        LID_index = 0
        num_rows = jet_event.shape[0]
        LID_column = np.full((1, num_rows), 0) # Make array of zeros, LID for jet is always 0
        NID_column = np.full((1, num_rows), self._next_jetID) # Make array of zeros
        jet_event = np.insert(jet_event, 0, LID_column, axis=1) # LID
        jet_event = np.insert(jet_event, 0, NID_column, axis=1) # NID
        jetpluspu = jet_event
        # Last available ID is tot num of loaded pileup
        pu_nos = np.random.randint(low = 0, high = self._max_PU_no, size = self.mu, dtype=np.int32)
        for pu_no in pu_nos:
            LID_index += 1
            pu_event = self.pu.select_event(pu_no)
            if pu_event.size == 0: continue  # Skip if empty pile-up
            num_rows = pu_event.shape[0]
            LID_column = np.full((1, num_rows), LID_index) # Make array of LIDs
            NID_column = np.full((1, num_rows), self._next_jetID) # Make array of NIDs
            pu_event = np.insert(pu_event, 0, LID_column, axis=1) # LID
            pu_event = np.insert(pu_event, 0, NID_column, axis=1) # NID
            jetpluspu = np.vstack((jetpluspu, pu_event))
        pxs, pys, pzs = jetpluspu[:, 5], jetpluspu[:, 6], jetpluspu[:, 7]
    
        enes = p_magnitude(pxs, pys, pzs)
        pTs = to_pT(pxs, pys)
        etas = pseudorapidity(enes, pzs)

        # Combine the following 2 ops later to optimise
        phis = to_phi(pxs, pys)
        phis = wrap_phi(self.jet_axis[1], phis)
        _, eta_c, phi_c = centre_on_jet(self.jet_axis, etas, phis)
        jetpluspu = np.hstack((jetpluspu, eta_c.reshape(-1, 1)))
        jetpluspu = np.hstack((jetpluspu, phi_c.reshape(-1, 1)))
        jetpluspu = np.hstack((jetpluspu, enes.reshape(-1, 1)))
        jetpluspu = np.hstack((jetpluspu, pTs.reshape(-1, 1)))

        jetpluspu = np.delete(jetpluspu, [2,3,4], axis=1)

        # print("jetpluspu", jetpluspu)
        self.current_event = jetpluspu  # Noisy event
        # print(f"self.current_event.shape = {self.current_event.shape}")
    
    def _mask(self):
        LIDs = self.current_event[:, 1].astype(int)
        d_etas = self.current_event[:, 5]
        d_phis = self.current_event[:, 6]
        dR2s = d_etas*d_etas + d_phis*d_phis
        self.current_event = self.current_event[ (LIDs == 0) | (dR2s < 1) ]  # First condition so 

    def _calculate_event_level(self):
        # Extract relevant columns
        NIDs = self.current_event[:, 0].astype(int)
        pxs, pys, pzs = self.current_event[:, 2], self.current_event[:, 3], self.current_event[:, 4]

        enes, pxs, pys, pzs = calculate_four_momentum_massless(NIDs, pxs, pys, pzs)
        event_p2 = contraction(enes, pxs, pys, pzs)
        self.event_level["mass"] = np.sqrt(event_p2)[0]
        self.event_level["eta"] = pseudorapidity(enes, pzs)[0]
        self.event_level["phi"] = to_phi(pxs, pys)[0]
        self.event_level["pT"] = to_pT(pxs, pys)[0]


    # Getters for quantity arrays
    @property
    def NID(self):
        return self.current_event[:, self.column_indices["NID"]]
    @property
    def LID(self):
        return self.current_event[:, self.column_indices["LID"]]
    @property
    def px(self):
        return self.current_event[:, self.column_indices["px"]]
    @property
    def py(self):
        return self.current_event[:, self.column_indices["py"]]
    @property
    def pz(self):
        return self.current_event[:, self.column_indices["pz"]]
    @property
    def eta(self):
        return self.current_event[:, self.column_indices["d_eta"]]
    @property
    def phi(self):
        return self.current_event[:, self.column_indices["d_phi"]]
    @property
    def mass(self):
        return self.current_event[:, self.column_indices["mass"]]
    @property
    def p_T(self):
        return self.current_event[:, self.column_indices["p_T"]]