# Local import
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import wrap_phi
from DDPMLHC.bmap import *

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

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
        
        self._max_TT_no = self.tt.max_ID
        self._max_PU_no = self.pu.max_ID
        self.grid_side_bins = BMAP_SQUARE_SIDE_LENGTH

        ### TODO: need to set
        self.scaling_mean = 0
        self.scaling_sd = 1

        # Define column index mapping for getters
        self.column_indices_all = {  # For array of quantities
            "NIDs": 0,
            "LIDs": 1,
            "pxs": 2,
            "pys": 3,
            "pzs": 4,
            "d_etas": 5,
            "d_phis": 6,
            "masses": 7,
            "p_Ts": 8,
        }
        self.column_indices_event = {  # For event-level quantities
            "event_id": 0,
            "px": 1,
            "py": 2,
            "pz": 3,
            "eta": 4,
            "phi": 5,
            "mass": 6,
            "p_T": 7,
        }

        self.reset()  # Initial reset

    def reset(self):
        """Start from beginning."""
        self._next_jetID = 0
        self.current_event = np.empty((0, 9))  # Needs to be 2D
        self.event_level = np.zeros(8)  # Array for event-level quantities

    def __str__(self):
        return (
            f"Current - Jet {self._next_jetID - 1} with mu={self.mu}\n"
            f"    Event-level quantities:\n"
            f"        EID  = {self.event_id}\n"
            f"        px   = {self.event_px}\n"
            f"        py   = {self.event_py}\n"
            f"        pz   = {self.event_pz}\n"
            f"        eta  = {self.event_eta}\n"
            f"        phi  = {self.event_phi}\n"
            f"        mass = {self.event_mass}\n"
            f"        p_T  = {self.event_pT}"
        )
    
    # Iterator methods
    
    def __repr__(self):
        return str(self.current_event)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._next_jetID == self._max_TT_no:
            raise StopIteration
        
        self._build_next_noisy_event()
        self._mask()
        self._calculate_event_level()
        self._next_jetID += 1
        return self.current_event

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
        self.event_id = self._next_jetID

        self.event_px = np.sum(self.pxs)
        self.event_py = np.sum(self.pys)
        self.event_pz = np.sum(self.pzs)

        event_ene = np.sum(self.masses)
        event_p2 = contraction(event_ene, self.event_px, self.event_py, self.event_pz)
        self.event_mass = np.sqrt(event_p2)

        self.event_eta = pseudorapidity(event_ene, self.event_pz)
        self.event_phi = to_phi(self.event_px, self.event_py)
        self.event_pT = to_pT(self.event_px, self.event_py)
    
    # Event-level output methods

    def collect_event_level_data(self):
        """Collect event-level data for all events. Will start from beginning of self.TTselector."""
        self.reset()
        combined = []
        for _ in self:
            combined.append(np.copy(self.event_level))
        return np.vstack(combined)

    def save_event_level_data(self, output_path=INTERMEDIATE_PATH, data=None):
        """
        Save event-level data to CSV.
        
        Args:
            output_path (str): Base path for output file
            data (np.ndarray, optional): Data to save. If None, collects new data
            mu (int, optional): Override instance mu value for filename
        """
        if data is None:
            data = self.collect_event_level_data()
        
        output_filename = f"noisy_mu{self.mu}_event_level.csv"
        output_filepath = f"{output_path}/{output_filename}"
        
        np.savetxt(
            output_filepath,
            data,
            delimiter=",",
            header="event_id,px,py,pz,eta,phi,mass,p_T",
            comments="",
            fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
        )
        
        print(f"Collapsed {int(data[-1,0])} events of mu = {self.mu} data to event-level.\n"
              f"    Saved to {output_filename}.")
        return data

    # Visualisations

    def visualise_current_event(self, save_path=None, particle_scale_factor=3000):
        """Plot the current event in eta-phi space.
        TODO: fix red circle in legend, make legend and info text same style and equidistant from edges.
        """
        if self.current_event.size == 0:
            raise RuntimeError("No event loaded to plot")
        if save_path is None:
            save_path = f"{CWD}/data/plots/visualise"
        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("$\Delta\eta$", fontsize=16)
        ax.set_ylabel("$\Delta\phi$", fontsize=16)
        
        # Configure y-axis ticks
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / 4))
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(
                lambda val, pos: f"${val/np.pi}\pi$" if val != 0 else "0"
            )
        )
        ax.grid(axis="y", linestyle="--", color="gray", alpha=0.7)
        
        # Plot jet center and boundary
        ax.plot(0, 0, marker="x", color="blue", label="Jet axis")
        ax.add_patch(plt.Circle((0,0), 1.0, color="black", linewidth=1, 
                            fill=False, label="$\Delta R = 1$", alpha=0.3))

        # Calculate marker sizes based on pT
        sizes = particle_scale_factor * self.p_Ts / np.max(self.p_Ts)
        
        # Plot particles
        jet_mask = self.LIDs == 0
        ax.scatter(self.etas[jet_mask], self.phis[jet_mask], 
                s=sizes[jet_mask], facecolors='none', color="red", alpha=1, label="Jet particles",
                linewidth=1,
                )
        ax.scatter(self.etas[~jet_mask], self.phis[~jet_mask], 
                s=sizes[~jet_mask], facecolors='none', color="blue", alpha=1, label="Pile-up")
        
        text_box_style = dict(
            facecolor='white',
            edgecolor='black',
            alpha=0.7,
            pad=0.5,
            boxstyle='round'
        )

        # Event info top left
        ax.text(0.05, 0.95, 
                f"Jet ${int(self.event_id)}$ with $\mu = {self.mu}$\n"
                f"$m = {self.event_mass:.1f}$ GeV\n"
                f"$p_T = {self.event_pT:.1f}$ GeV\n"
                f"$\eta = {self.event_eta:.2f}$",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=12,
                bbox=text_box_style,
                )
        
        # Legend top right
        legend = ax.legend(fontsize=12, 
                          bbox_to_anchor=(1, 1),
                          loc='upper right',
                          bbox_transform=ax.transAxes)
        # Style the legend box to match the text box
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        frame.set_alpha(0.7)
    
        plt.tight_layout()
        
        # Save plot
        filename = f"event{self._next_jetID-1}_mu{self.mu}"
        plt.savefig(f"{save_path}/{filename}_vis.png", bbox_inches='tight')
        plt.close()

    def bmap_current_event(self, save_path=None):
        if self.current_event.size == 0:
            raise RuntimeError("No event loaded to plot")
        if save_path is None:
            save_path = f"{CWD}/data/plots/bmaps"
        
        bins = self.grid_side_bins

        grid = self.vectorise(for_bmap=True).reshape((bins, bins))  # Get grid, with scaling for bmap visualisation
        grid = np.clip(grid, 0, 255).astype(np.uint8)  # Remove OOB vals
        
        im = Image.fromarray(grid)
        filename = f"event_{self._next_jetID-1}_mu{self.mu}"
        im.save(f"{save_path}/{filename}.png")

    # Ops

    def vectorise(self, for_bmap=False):
        bins = self.grid_side_bins

        x, y = unit_square_the_unit_circle(self.etas, self.phis)  # Map to unit square
        x_discrete, y_discrete = discretise_points(x, y, N=bins)  # Discretise coords

        # TODO: Scale energies - for now, apply BEFORE gridding - check!
        if not for_bmap:
            scaled_energies = (self.masses - self.scaling_mean) / (self.scaling_sd)
        elif for_bmap:  # This scaling is only for visualization (to 3SD), not suitable for NN input
            mean = np.mean(self.masses)
            std = np.std(self.masses)
            scaled_energies = 255 * (self.masses - mean) / (3 * std) + 128

        grid = np.zeros((bins, bins), dtype=np.float32)
        for e, xi, yi in zip(scaled_energies, x_discrete, y_discrete):
            grid[yi, xi] += float(e)
        
        return grid.reshape(bins * bins)

    # Getters for quantity arrays
    @property
    def NIDs(self):
        return self.current_event[:, self.column_indices_all["NIDs"]]
    @property
    def LIDs(self):
        return self.current_event[:, self.column_indices_all["LIDs"]]
    @property
    def pxs(self):
        return self.current_event[:, self.column_indices_all["pxs"]]
    @property
    def pys(self):
        return self.current_event[:, self.column_indices_all["pys"]]
    @property
    def pzs(self):
        return self.current_event[:, self.column_indices_all["pzs"]]
    @property
    def etas(self):
        return self.current_event[:, self.column_indices_all["d_etas"]]
    @property
    def phis(self):
        return self.current_event[:, self.column_indices_all["d_phis"]]
    @property
    def masses(self):
        return self.current_event[:, self.column_indices_all["masses"]]
    @property
    def p_Ts(self):
        return self.current_event[:, self.column_indices_all["p_Ts"]]

    # Getters and setters for event-level quantities

    @property
    def event_id(self):
        return self.event_level[self.column_indices_event["event_id"]]
    @event_id.setter
    def event_id(self, val):
        self.event_level[self.column_indices_event["event_id"]] = val
    
    @property
    def event_px(self):
        return self.event_level[self.column_indices_event["px"]]
    @event_px.setter
    def event_px(self, val):
        self.event_level[self.column_indices_event["px"]] = val

    @property
    def event_py(self):
        return self.event_level[self.column_indices_event["py"]]
    @event_py.setter
    def event_py(self, val):
        self.event_level[self.column_indices_event["py"]] = val

    @property
    def event_pz(self):
        return self.event_level[self.column_indices_event["pz"]]
    @event_pz.setter
    def event_pz(self, val):
        self.event_level[self.column_indices_event["pz"]] = val

    @property
    def event_eta(self):
        return self.event_level[self.column_indices_event["eta"]]
    @event_eta.setter
    def event_eta(self, val):
        self.event_level[self.column_indices_event["eta"]] = val

    @property
    def event_phi(self):
        return self.event_level[self.column_indices_event["phi"]]
    @event_phi.setter
    def event_phi(self, val):
        self.event_level[self.column_indices_event["phi"]] = val

    @property

    def event_mass(self):
        return self.event_level[self.column_indices_event["mass"]]
    @event_mass.setter
    def event_mass(self, val):
        self.event_level[self.column_indices_event["mass"]] = val

    @property
    def event_pT(self):
        return self.event_level[self.column_indices_event["p_T"]]
    @event_pT.setter
    def event_pT(self, val):
        self.event_level[self.column_indices_event["p_T"]] = val
