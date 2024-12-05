"""1D Histograms"""

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import multiprocessing

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

# Load intermediate data
tt = np.genfromtxt(
    TT_EXT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
jets = np.genfromtxt(
    JET_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

# Indiv parts
px = tt[:, 1]
py = tt[:, 2]
pz = tt[:, 3]
eta = tt[:, 4]
phi = tt[:, 5]
p_T = tt[:, 6]
p = p_magnitude(px, py, pz)

# Jets
jet_eta = jets[:, 4]
jet_phi = jets[:, 5]
jet_mass = jets[:, 6]
jet_pT = jets[:, 7]

# Define the save path and plot characteristics
save_path = f"{CWD}/data/plots/1D_histograms/mu0/"
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

def plot_1D_hist(name, data, xlog=False, plot_params=DEFAULT_PLOT_PARAMS, save_path=save_path, save_filename="out", x_min=None, x_max=None, ax=None, plot_ylabel=True):
    """Plot a 1D histogram. Optionally plot on a provided axis."""
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
    print("Plotting combined histogram...")
    num_rows, num_cols = 2, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
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
    plt.savefig(f"{save_path}/grid_histograms.png", dpi=600)


hist_data_cleanjets = [
    {
        "name": "Momentum $p$ [GeV]",
        "data": p,
        "plot_params": {"xlog": True},
        "save_filename": "p",
        "save_path": save_path,
    },
    {
        "name": "Pseudorapidity $\eta$",
        "data": eta,
        "plot_params": {},
        "save_filename": "eta",
        "save_path": save_path,
    },
    {
        "name": "Transverse Momentum $p_T$ [GeV]",
        "data": p_T,
        "plot_params": {"xlog": True},
        "save_filename": "pT",
        "save_path": save_path,
    },
    {
        "name": "Jet Mass [GeV]",
        "data": jet_mass,
        "plot_params": {"x_max": 250},
        "save_filename": "jet_mass",
        "save_path": save_path,
    },
    {
        "name": "Jet Pseudorapidity $\eta$",
        "data": jet_eta,
        "plot_params": {"bins": 50},
        "save_filename": "jet_eta",
        "save_path": save_path,
    },
    {
        "name": "Jet Transverse Momentum $p_T$ [GeV]",
        "data": jet_pT,
        "plot_params": {"xlog": True, "bins": 50, "x_max": 1000},
        "save_filename": "jet_pT",
        "save_path": save_path,
    }
]

for i in range(len(hist_data_cleanjets)):
    hist_data_cleanjets[i]["save_path"] = save_path

# print("-- Plot clean jets")
# plot_single_histograms(hist_data_cleanjets, save_path)
# plot_combined_histograms(hist_data_cleanjets, save_path)

##############################################################################

# mu300_event_stats_path = f"{CWD}/data/2-intermediate/noisy_event_stats_mu300.csv"

# jets_mu300 = np.genfromtxt(
#     mu300_event_stats_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )

# save_path = f"{CWD}/data/plots/1D_histograms/mu300/"

# # Jets
# jet_eta = jets_mu300[:, 4]
# jet_phi = jets_mu300[:, 5]
# jet_mass = jets_mu300[:, 6]
# jet_pT = jets_mu300[:, 7]

# print("Final idx: ", jets_mu300[-1, 0])

# hist_data_noisyjets_mu300 = [
#     {
#         "name": "Jet Mass [GeV]",
#         "data": jet_mass,
#         "plot_params": {},
#         "save_filename": "jet_mass",
#         "save_path": save_path,
#     },
#     {
#         "name": "Jet Pseudorapidity $\eta$",
#         "data": jet_eta,
#         "plot_params": {"bins": 50},
#         "save_filename": "jet_eta",
#         "save_path": save_path,
#     },
#     {
#         "name": "Jet Transverse Momentum $p_T$ [GeV]",
#         "data": jet_pT,
#         "plot_params": {"xlog": True, "bins": 50, "x_max": 1000},
#         "save_filename": "jet_pT",
#         "save_path": save_path,
#     }
# ]

# for i in range(len(hist_data_noisyjets_mu300)):
#     hist_data_noisyjets_mu300[i]["save_path"] = save_path

# print("-- Plot noisy jets, mu=300")
# plot_single_histograms(hist_data_noisyjets_mu300, save_path)
# plot_combined_histograms(hist_data_noisyjets_mu300, save_path)

##############################################################################

mu10_event_stats_path = f"{CWD}/data/2-intermediate/noisy_event_stats_mu10.csv"

jets_mu10 = np.genfromtxt(
    mu10_event_stats_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

save_path = f"{CWD}/data/plots/1D_histograms/mu10/"

# Jets
jet_eta = jets_mu10[:, 4]
jet_phi = jets_mu10[:, 5]
jet_mass = jets_mu10[:, 6]
jet_pT = jets_mu10[:, 7]

print("Final idx: ", jets_mu10[-1, 0])

hist_data_noisyjets_mu10 = [
    {
        "name": "Jet Mass [GeV]",
        "data": jet_mass,
        # "data": jet_pT,
        "plot_params": {"xlog": True, "x_min": 1, "x_max": 300},
        "save_filename": "jet_mass",
        "save_path": save_path,
    },
    {
        "name": "Jet Pseudorapidity $\eta$",
        "data": jet_eta,
        "plot_params": {"bins": 50},
        "save_filename": "jet_eta",
        "save_path": save_path,
    },
    {
        "name": "Jet Transverse Momentum $p_T$ [GeV]",
        "data": jet_pT,
        # "data": jet_mass,
        "plot_params": {"xlog": True, "bins": 50, "x_max": 1000},
        "save_filename": "jet_pT",
        "save_path": save_path,
    }
]

for i in range(len(hist_data_noisyjets_mu10)):
    hist_data_noisyjets_mu10[i]["save_path"] = save_path

print("-- Plot noisy jets, mu=10")
plot_single_histograms(hist_data_noisyjets_mu10, save_path)
plot_combined_histograms(hist_data_noisyjets_mu10, save_path)

##############################################################################

print("Done.")
