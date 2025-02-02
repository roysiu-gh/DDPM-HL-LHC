# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

PDG_IDS = {
    0: r"$\gamma$ (Photon)",
    11: r"$e^-$ (Electron)",
    -11: r"$e^+$ (Positron)",
    22: r"$\gamma$ (Photon)",
    130: r"$K^0_S$ (K-short)",
    211: r"$\pi^+$ (Pion)",
    -211: r"$\pi^-$ (Pion)",
    321: r"$K^+$ (Kaon)",
    -321: r"$K^-$ (Kaon)",
    2112: r"$n$ (Neutron)",
    -2112: r"$\bar{n}$ (Antineutron)",
    2212: r"$p$ (Proton)",
}

# Global color scheme (tab20 for extended color range)
unique_abs_pdgids = sorted(abs(pdgid) for pdgid in PDG_IDS.keys())
cmap = mpl.colors.ListedColormap(plt.cm.tab20(np.linspace(0, 1, len(unique_abs_pdgids))))
GLOBAL_CMAP = {pid: cmap(i) for i, pid in enumerate(unique_abs_pdgids)}

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

#################################################################################

class NoisyGenerator:
    def __init__(self, TTselector:EventSelector, PUselector:EventSelector, mu=0, bins=BMAP_SQUARE_SIDE_LENGTH):
        self.tt = TTselector
        self.pu = PUselector
        self.mu = mu
        
        self._max_TT_no = self.tt.max_ID
        self._max_PU_no = self.pu.max_ID
        self.bins = bins

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

    def __len__(self):
        return int(self._max_TT_no) + 1
    
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

    def select_jet(self, jet_no):
        if jet_no >= self._max_TT_no:
            raise RuntimeError("Requested jet not in loaded set. Did nothing.")
        self._next_jetID = jet_no
        next(self)

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

        self.PDGIDs = jetpluspu[:, 3]  # For use in self.visualise_current_event(show_pdgids=True)
        jetpluspu = np.delete(jetpluspu, [2,3,4], axis=1)  # Remove charge, PDGIDs

        self.current_event = jetpluspu  # Noisy event
    
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

    def visualise_current_event(self, save_path=None, particle_scale_factor=3000, show_pdgids=False):
        """
        Plot the current event in eta-phi space.
        
        Args:
            save_path: Directory to save plot
            particle_scale_factor: Scale factor for particle marker sizes
            show_pdgids: Whether to show PDG IDs in legend
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
        ax.plot(0, 0, marker="x", color="blue")
        ax.add_patch(plt.Circle((0,0), 1.0, color="black", linewidth=1, fill=False, alpha=0.3))

        # NB particle AREAs (not radii) are proportional to masses
        sizes = particle_scale_factor * self.masses / np.max(self.masses)
        
        # Plot particles
        if show_pdgids:
            pdgid_values = self.PDGIDs.astype(int)
            colours = [GLOBAL_CMAP.get(abs(pid), "black") for pid in pdgid_values]
            
            # Plot particles with PDG colors
            ax.scatter(self.etas, self.phis, s=sizes, facecolors="none", 
                    edgecolors=colours, alpha=1, linewidth=1)
            
            # Create PDG ID legend
            handles = []
            unique_detected_pdgids = sorted(set(pdgid_values))
            unique_abs_detected_pdgids = sorted(set(abs(i) for i in pdgid_values))
            
            for abs_pid in unique_abs_detected_pdgids:
                colour = GLOBAL_CMAP.get(abs_pid, "grey")
                if abs_pid in unique_detected_pdgids:
                    handles.append(Patch(
                        label=PDG_IDS.get(abs_pid, "Unknown"),
                        color=colour
                    ))
                if -abs_pid in unique_detected_pdgids:
                    handles.append(Patch(
                        label=PDG_IDS.get(-abs_pid, "Unknown"),
                        edgecolor=colour,
                        facecolor="none"
                    ))
        else:
            # Plot with simple jet/pile-up colors
            jet_mask = self.LIDs == 0
            ax.scatter(self.etas[jet_mask], self.phis[jet_mask], 
                    s=sizes[jet_mask], facecolors="none", color="red", 
                    alpha=1, linewidth=0.5)
            ax.scatter(self.etas[~jet_mask], self.phis[~jet_mask], 
                    s=sizes[~jet_mask], facecolors="none", color="blue", 
                    alpha=0.5)
            
            # Create legend handles with fixed-size markers
            handles = [
                Line2D([0], [0], marker='x', color='blue', label='Jet axis'),
                plt.Circle((0,0), 1.0, color="black", fill=False, alpha=0.5, label="$\Delta R = 1$"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor='red', markersize=10, label='Jet particles'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor='blue', markersize=10, label='Pile-up')
            ]

        # Same style for all boxes
        text_box_style = dict(
            facecolor="white",
            edgecolor="black",
            alpha=0.7,
            pad=0.5,
            boxstyle="round"
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
        
        # Add mass scale references (bottom left)
        reference_mass_1 = 10.0  # GeV
        reference_mass_2 = 100.0  # GeV
        reference_size_1 = particle_scale_factor * reference_mass_1 / np.max(self.masses)
        reference_size_2 = particle_scale_factor * reference_mass_2 / np.max(self.masses)
        # Position circles
        ax.scatter(-0.68, -0.9, s=reference_size_1, facecolor="green", edgecolor="none", alpha=0.7)
        ax.scatter(-0.85, -0.8, s=reference_size_2, facecolor="orange", edgecolor="none", alpha=0.7)
        # Add to legend handles
        handles.append(Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='green',
                            markeredgecolor='none',
                            markersize=10, 
                            label=f"{reference_mass_1:.0f} GeV"))
        handles.append(Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='orange',
                            markeredgecolor='none',
                            markersize=10, 
                            label=f"{reference_mass_2:.0f} GeV"))
        
        # Legend top right
        if show_pdgids:
            # Adjust plot size to accommodate PDG legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax.legend(handles=handles, loc="center left", 
                            bbox_to_anchor=(1, 0.5), fontsize=12)
        else:
            legend = ax.legend(handles=handles, fontsize=12, 
                            bbox_to_anchor=(1, 1),
                            loc="upper right", 
                            bbox_transform=ax.transAxes)
        
        # Style legend box
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("black")
        frame.set_alpha(0.7)

        plt.tight_layout()
        
        # Save plot
        filename = f"event{self._next_jetID-1}_mu{self.mu}"
        suffix = "_pdgids" if show_pdgids else "_vis"
        plt.savefig(f"{save_path}/{filename}{suffix}.png", bbox_inches="tight")
        plt.close()

    # Ops

    def get_grid(self):
        bins = self.bins

        x, y = unit_square_the_unit_circle(self.etas, self.phis)  # Map to unit square
        x_discrete, y_discrete = discretise_points(x, y, N=bins)  # Discretise coords

        # TODO: Scale energies - for now, apply BEFORE gridding - check!
        scaled_energies = (self.masses - self.scaling_mean) / (self.scaling_sd)

        grid = np.zeros((bins, bins), dtype=np.float32)
        for e, xi, yi in zip(scaled_energies, x_discrete, y_discrete):
            grid[yi, xi] += float(e)
        
        return grid

    def vectorise(self):
        grid = self.get_grid()        
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
