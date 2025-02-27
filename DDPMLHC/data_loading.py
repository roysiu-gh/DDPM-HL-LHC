# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
# Package imports
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
import random
from tqdm import tqdm

from torch.utils.data import Dataset
from denoising_diffusion_pytorch import GaussianDiffusion
from torch.amp import autocast
import torch.nn.functional as F
from einops import reduce

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
        # print(start)
        # print(end)
        return self.data[start:end]
    
    def to(self, device):
        self.data = torch.from_numpy(self.data)
        self.data.to(device)
    
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

class NoisyGenerator(object):
    def __init__(self, TTselector:EventSelector, PUselector:EventSelector, mu=0, bins=BMAP_SQUARE_SIDE_LENGTH, pu_only=False):
        self.tt = TTselector
        self.pu = PUselector
        self.mu = mu
        
        self._max_TT_no = self.tt.max_ID.astype(int)
        self._max_PU_no = self.pu.max_ID.astype(int)
        # self.grid_side_bins = BMAP_SQUARE_SIDE_LENGTH
        self.bins = bins
        self.grid = None
        self.max_energy = 0
        ### TODO: need to set
        self.scaling_mean = 0
        self.scaling_sd = 1
        self.pu_only = pu_only
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
        self._max_energy()

        self.reset()  # Initial reset

    def reset(self):
        """Start from beginning."""
        self._next_jetID = 0
        self.current_event = np.empty((0, 9))  # Needs to be 2D
        self.event_level = np.zeros(8)  # Array for event-level quantities

    def to(self, device):
        self.tt.to(device)
        self.pu.to(device)

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
            raise StopIteration#
        # print(self.tt) if isinstance(self.tt, np.ndarray) else None
        self._build_next_noisy_event()
        self._mask()
        if not self.pu_only:
            self._calculate_event_level()
        self._next_jetID += 1
        return self.current_event
    def __len__(self):
        # print(len(self.current_event))
        length = len(self.current_event)
        return length
    def select_jet(self, jet_no):
        if jet_no >= self._max_TT_no:
            raise RuntimeError("Requested jet not in loaded set. Did nothing.")
        self._next_jetID = jet_no
        next(self)
    def _max_energy(self):
        for jet_no in range(self._max_TT_no):
            jet_event = self.tt.select_event(jet_no)
            pxs, pys, pzs = jet_event[:,3], jet_event[:,4], jet_event[:,5]
            # self.jet_axis = get_axis_eta_phi(jet_px, jet_py, jet_pz)
            enes = np.sum(p_magnitude(pxs, pys, pzs))
            if self.max_energy < enes:
                self.max_energy = enes
            else: 
                continue
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
        # If self.jet_only, can skip pile-up step
        if self.mu > 0:
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
        self.current_event = jetpluspu
        # print(jetpluspu)
        if self.pu_only:
            data = jetpluspu[(jet_event.shape[0]):] 
            self.current_event =  data # Noisy event
            return data
        return jetpluspu
        # self.current_event = jetpluspu  # Noisy event
        
        
    def _mask(self):
        # If using noisy generator to only return pile-up

        LIDs = self.current_event[:, 1].astype(int)
        d_etas = self.current_event[:, 5]
        d_phis = self.current_event[:, 6]
        dR2s = d_etas*d_etas + d_phis*d_phis
        self.current_event = self.current_event[ (LIDs == 0) | (dR2s < 1) ]  # First condition so 
        return 

    def _calculate_event_level(self):
        self.event_id = self._next_jetID

        event_quantities = particle_momenta_to_event_level(self.masses, self.pxs, self.pys, self.pzs)

        self.event_mass = event_quantities[0]
        self.event_px = event_quantities[1]
        self.event_py = event_quantities[2]
        self.event_pz = event_quantities[3]
        self.event_eta = event_quantities[4]
        self.event_phi = event_quantities[5]
        self.event_pT = event_quantities[6]

    def _collect_event_level_data(self):
        """Collect event-level data for all events. Will start from beginning of self.TTselector."""
        self.reset()
        combined = []
        for _ in self:
            combined.append(np.copy(self.event_level))
        return np.vstack(combined)

    def save_event_level_data(self, output_path=INTERMEDIATE_PATH):
        data = self._collect_event_level_data()
        
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

    def get_grid(self, normalise=True):
        bins = self.bins
        x, y = unit_square_the_unit_circle(self.etas, self.phis)  # Map to unit square
        x_discrete, y_discrete = discretise_points(x, y, N=bins)  # Discretise coords
        # TODO: Scale energies - for now, apply BEFORE gridding - check!
        scaled_energies = (self.masses - self.scaling_mean) / (self.scaling_sd)
        grid = np.zeros((bins, bins), dtype=np.float32)
        for e, xi, yi in zip(scaled_energies, x_discrete, y_discrete):
            grid[yi, xi] += float(e)
        
        if not normalise:
            return grid
        elif self.max_energy <= 1:
            return grid
        else:
            return grid / self.max_energy

    def vectorise(self):
        grid = self.get_grid()        
        return grid.reshape(self.bins * self.bins)

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

class NGenForDataloader(Dataset):
    def __init__(self, noisy_generator, njets=100):
        self.ng = noisy_generator
        self.jets = []
        self.njets = njets
        # next(self.ng)
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.ng._max_TT_no - 1
    
    def __getitem__(self, idx):
        self.ng.select_jet(idx)
        x = torch.from_numpy( self.ng.get_grid() ).float()
        # x = x.unsqueeze(0)
        x = x.unsqueeze(0)

        return x


###############################################################################
# Some functions from denoising_diffusion_pytorch that are required but couldn't import
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def extract(a, t, x_shape):
    """from denoising_diffusion_pytorch that are required but couldn't import"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# Custom DataLoader class to pass in entire jet dataset
class NGenForDataloader(Dataset):
    def __init__(self, noisy_generator, njets=100):
        self.ng = noisy_generator
        self.jets = []
        self.njets = njets
        # next(self.ng)
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.ng._max_TT_no - 1
    
    def __getitem__(self, idx):
        self.ng.select_jet(idx)
        x = torch.from_numpy( self.ng.get_grid() ).float()
        x = x.unsqueeze(0)
        return x

#############################################################################################

class PUDiffusion(GaussianDiffusion):
    def __init__(self, model, image_size, timesteps, puNG: NoisyGenerator, jet_ng: NoisyGenerator, mu=200, **kwargs):
        super(PUDiffusion, self).__init__(model=model, image_size=image_size, timesteps=timesteps, **kwargs)
        self.puNG = puNG
        self.jetNG = jet_ng
        self.channels = model.channels
        self.mu_counter = 1
        self.timesteps = timesteps
        self.mu = mu
    
    #############################################################################################

    def cond_noise(self, x_shape, noise):
        return self.pu_to_tensor(x_shape) if noise is None else noise
        # return torch.zeros_like(x_start) if noise is None else noise
    def generate_data(self, shape, NG: NoisyGenerator):
        """
        This function generates image data matched to the correct shape
        """
        # Start next jet
        next(NG)
        selected = NG.get_grid()
        # If empty pile-up, return array of 0s instead since model should account for this
        if selected.size == 0:
            return  "Error in PUDiffusion.generate_jet"
        # print(selected_pu.shape)
        pu_tensor = torch.from_numpy(selected).float()

        pu_tensor = torch.unsqueeze(pu_tensor,0)
        # This tensor has dimensions BxCxHxW to match x_start
        pu_tensor = torch.unsqueeze(pu_tensor,0)
        pu_tensor = pu_tensor.expand(shape[0], shape[1], -1, -1) 
        # pu_tensor = torch.zeros(shape)
        pu_tensor = pu_tensor.to(self.device)
        return pu_tensor
    # @torch.inference_mode()
    def pu_to_tensor(self, shape):
        # Select random number of pile-ups (mu) to generate, max 200 for now since HL-LHC expected to do up to this
        # We are doing it per batch
        mu = 1
        # Align jetIDs for correct centering of pile-up
        self.puNG._next_jetID = self.jetNG._next_jetID
        NG = self.puNG
        NG.mu = mu
        # NG.reset()
        # next(self.puNG)
        pu_tensor = self.generate_data(shape=shape, NG=NG)
        return pu_tensor
    # @torch.inference_mode()
    def jet_to_tensor(self, shape):
        NG = self.jetNG
        # Align jetIDs for correct centering of pile-up
        self.puNG._next_jetID = self.jetNG._next_jetID
        # next(NG)
        pu_tensor = self.generate_data(shape=shape, NG=self.jetNG)
        return pu_tensor
    
    #############################################################################################
    
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        # print("batched times", t)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        ######## MODIFY
        noise = self.pu_to_tensor(x.shape) if t > 0 else 0 # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = self.cond_noise(x_shape=x_start.shape, noise=noise)

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]
        # print("q_sample t", t)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # @torch.inference_mode()
    def generate_noise(self,shape):
        batch, device = shape[0], self.device
        jets = []
        # jet_indices = [16897, 54328, 7898, 2854]
        for i in range(batch):
            # random_jet_no = np.random.randint(low=0, high=self.jetNG._max_TT_no, size=None)
            self.jetNG._next_jetID = i
            # Prints jets to make images from
            # print(self.jetNG._next_jetID)
            # self.jetNG.select_jet(random_jet_no)  # or however you select jets
            self.jetNG.select_jet(i)  # or however you select jets
            
            jet = torch.from_numpy(self.jetNG.get_grid()).unsqueeze(0).float()
            # Now to add pile-up
            # random_pu_no = np.random.randint(low=0, high=self.jetNG._max_TT_no, size=None)
            self.puNG._next_jetID = i
            # Start from 200 pileups
            self.puNG.mu = self.mu
            # Generate them
            next(self.puNG)
            selected_pu = self.puNG.get_grid()
            pu_tensor = torch.from_numpy(selected_pu).float()
            pu_tensor = torch.unsqueeze(pu_tensor,0)
            noised_jet = jet + pu_tensor # add energies element wise for each bin
            # noised_jet = noised_jet.to(self.device)
            jets.append(noised_jet)
        # Should now  be batch x 1 x grid x grid
        jets = torch.stack(jets)
        jets = jets.to(self.device)
        return jets
   
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device
        img = self.generate_noise(shape)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)  # Returns intermediate imgs?

        ret = self.unnormalize(ret)
        print("final timestep: ", self.num_timesteps)
        return ret
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)


    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape
        noise = self.cond_noise(x_start.shape, noise=noise)
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    def forward(self, img, *args, **kwargs):
        # img = img.squeeze(0)
        # print("???", *img.shape)
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
