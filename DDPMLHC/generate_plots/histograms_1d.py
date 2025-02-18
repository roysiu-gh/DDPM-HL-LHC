"""1D Histograms"""

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import multiprocessing
from pathlib import Path
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import p_magnitude
line_thickness = 3.0
axes_thickness = 4.0
leg_size = 30

# ======= global matplotlib params =====
custom_params ={'axes.grid' : False}
sb.set_theme(style="ticks", rc=custom_params)
# plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 16  # Set default font size (optional)

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

DEFAULT_PLOT_PARAMS = {
    "bins": 100,
    "color": "skyblue",
    "edgecolor": "none",
    "kde": False,
    "stat": "density",  # Equivalent to `density=True` in plt.hist
}
class Set2DP(mpl.ticker.ScalarFormatter):
   def _set_format(self):
      self.format = "%.2f"

def plot_1D_hist(name, data, xlog=False, plot_params=DEFAULT_PLOT_PARAMS, save_path=None, save_filename="out", x_min=None, x_max=None, ax=None, plot_ylabel=True):
    """Plot a 1D histogram. Optionally plot on a provided axis."""

    ### CHANGE THIS LATER!
    xlog = False  # Otherwise breaks on the data regenerated from grid

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure if no axes provided
    else:
        fig = None  # If ax provided, do not create a new figure
    # Copy default params if not already assigned
    plot_params.update({ k:v for k, v in DEFAULT_PLOT_PARAMS.items() if k not in plot_params})
    
    sb.histplot(data, ax=ax, log_scale=(xlog, False), **plot_params)

    ax.set_xlabel(name, fontsize=16)
    ax.set_xlim(left=x_min, right=x_max)
    set_y_2p = Set2DP()
    ax.yaxis.set_major_formatter(set_y_2p)
    if plot_ylabel:  # Show y-label only for leftmost plots
        ax.set_ylabel("Frequency Density", fontsize=16)
    else:  # No ylabel for other plots
        ax.set_ylabel("")
    
    if save_path and fig:
        fig.savefig(f"{save_path}/{save_filename}.png", dpi=600)

    return fig, ax

def plot_wrapper(entry):
    # Unpack the dictionary and pass as arguments
    plot_params = entry.get("plot_params", {}).copy()  # Copy to avoid mutation
    xlog = plot_params.pop("xlog", False)
    x_min = plot_params.pop("x_min", None)
    x_max = plot_params.pop("x_max", None)

    return plot_1D_hist(
        name=entry["name"],
        data=entry["data"],
        xlog=xlog,
        x_min=x_min,
        x_max=x_max,
        plot_params=plot_params,
        save_path=entry["save_path"],
        save_filename=entry["save_filename"],
    )

# Single plot function
def plot_single_histograms(hist_data, save_path):
    print("Plotting single histograms...")
    with multiprocessing.Pool(processes=len(hist_data)) as pool:
        pool.map(plot_wrapper, hist_data)
    pool.close()
    pool.join()

# Multi-plot function
def plot_combined_histograms(hist_data, save_path):
    """Plot mass, eta, p_T, 3 cols 1 row."""
    print("Plotting combined histogram...")
    num_rows, num_cols = 1, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5))
    axes = axes.flatten()  # Flatten to iterate over axes

    for idx, (ax, entry) in enumerate(zip(axes, hist_data)):
        plot_params = entry.get("plot_params", {}).copy()  # Copy to avoid mutation
        xlog = plot_params.pop("xlog", False)
        x_min = plot_params.pop("x_min", None)
        x_max = plot_params.pop("x_max", None)
        plot_ylabel = True if idx % num_cols == 0 else False  # Show y-label for leftmost plots

        plot_1D_hist(
            name=entry["name"],
            data=entry["data"],
            save_filename=entry["save_filename"],
            xlog=xlog,
            x_min=x_min,
            x_max=x_max,
            ax=ax,
            plot_ylabel=plot_ylabel,
            plot_params=plot_params,
        )

    # Remove unused subplots
    for i in range(len(hist_data), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(f"{save_path}/multiple.png", dpi=600)

def plot_1d_histograms(mu, event_stats_path=None, output_path=None):
    if event_stats_path is None:
        event_stats_path = f"{CWD}/data/2-intermediate/noisy_mu{mu}_event_level.csv"
    print(f"Doing mu = {mu}...")
    events_dat = np.genfromtxt(
        event_stats_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
    )
    if output_path is None:
        output_path = f"{CWD}/data/plots/1D_histograms/mu{mu}/"

    event_eta = events_dat[:, 4]
    event_mass = events_dat[:, 6]
    event_pT = events_dat[:, 7]
    
    print("Final idx (#events - 1): ", events_dat[-1, 0])

    mass_max = 300
    mass_num_bins = 50
    mass_bins = np.mgrid[0:mass_max:(mass_num_bins+1)*1j]

    pT_max = 1000
    pT_num_bins = 50
    pT_bins = np.mgrid[0:pT_max:(pT_num_bins+1)*1j]

    list_of_params_foobar = [
        {
            "name": "Mass [GeV]",
            "data": event_mass,
            "plot_params": {"bins" : mass_bins},
            "save_filename": "event_mass",
            "save_path": output_path,
        },
        {
            "name": "Pseudorapidity eta",
            "data": event_eta,
            "plot_params": {"bins": 50},
            "save_filename": "event_eta",
            "save_path": output_path,
        },
        {
            "name": "Transverse Momentum p_T [GeV]",
            "data": event_pT,
            "plot_params": {"xlog": True, "bins": pT_bins},
            "save_filename": "event_pT",
            "save_path": output_path,
        }
    ]

    plot_single_histograms(list_of_params_foobar, output_path)
    plot_combined_histograms(list_of_params_foobar, output_path)
    print(f"Done mu = {mu}.\n")
