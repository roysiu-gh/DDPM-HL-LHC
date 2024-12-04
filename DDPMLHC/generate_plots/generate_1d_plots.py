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

def plot_1D_hist(name, data, xlog=False, plot_params=DEFAULT_PLOT_PARAMS, save_path=save_path, save_filename="out", x_min=None, x_max=None, ax=None, plot_ylabel=True, manual_xticklabel =  False, xtick_labels = None):
    """Plot a 1D histogram. Optionally plot on a provided axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure if no axes provided
    else:
        fig = None  # If ax provided, do not create a new figure
    # Copy default params if not already assigned
    plot_params.update({ k:v for k, v in DEFAULT_PLOT_PARAMS.items() if k not in plot_params})
    sb.histplot(data, ax=ax,log_scale=(xlog, False), **plot_params)

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
    if manual_xticklabel and xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    if name == "Jet Transverse Momentum $p_T$ [GeV]":
        # Manually set labels to fit properly for the subplots
        labels = [item.get_text() for item in ax.get_xticklabels()]
        # ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=None))
        minor_locator = mpl.ticker.LogLocator(base=10.0, subs=[3, 6, 10])
        ax.xaxis.set_minor_locator(minor_locator)
        # ax.set_xticklabels(labels)

    return fig, ax

# Plot singles

# print("Plotting single histograms...")
hist_data = [
    ("Momentum $p$ [GeV]", p, True, {}, save_path, "p"),
    ("Pseudorapidity $\eta$", eta, False, {}, save_path, "eta"),
    ("Transverse Momentum $p_T$ [GeV]", p_T, True, {}, save_path, "pT"),

    ("Jet Mass [GeV]", jet_mass, False, {}, save_path, "jet_mass", None, 250),
    ("Jet Pseudorapidity $\eta$", jet_eta, False, {"bins": 50}, save_path, "jet_eta"),
    ("Jet Transverse Momentum $p_T$ [GeV]", jet_pT, True, {"bins": 50}, save_path, "jet_pT")
]

# plot_1D_hist("Momentum $p$ [GeV]", p, save_filename="p", xlog=True)
# plot_1D_hist("Pseudorapidity $\eta$", eta, save_filename="eta", )
# plot_1D_hist("Transverse Momentum $p_T$ [GeV]", p_T, save_filename="pT", xlog=True)

# plot_1D_hist("Jet Mass [GeV]", jet_mass, save_filename="jet_mass", x_max=250)
# plot_1D_hist("Jet Pseudorapidity $\eta$", jet_eta, save_filename="jet_eta", plot_params={"bins": 50})
# plot_1D_hist("Jet Transverse Momentum $p_T$ [GeV]", jet_pT, save_filename="jet_pT", xlog=True, plot_params={"bins": 50})
print("Plotting single histograms...")

with multiprocessing.Pool(processes=len(hist_data)) as pool:
    results = pool.starmap(plot_1D_hist, hist_data)
pool.close()
pool.join()

# Plot multiplot


    
hist_data = [
    ("Momentum $p$ [GeV]", p, {"xlog": True}),
    ("Pseudorapidity $\eta$", eta, {}),
    ("Transverse Momentum $p_T$ [GeV]", p_T, {"xlog": True}),

    ("Jet Mass [GeV]", jet_mass, {"x_max": 250}),
    ("Jet Pseudorapidity $\eta$", jet_eta, {"bins": 50}),
    ("Jet Transverse Momentum $p_T$ [GeV]", jet_pT, {"xlog": True, "bins": 50},)
]
print("Plotting combined histogram...")

num_rows, num_cols = 2, 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
plt.tight_layout()
axes = axes.flatten()  # Flatten to iterate over axes

for idx, (ax, (name, data, params)) in enumerate(zip(axes, hist_data)):
    xlog = params.pop("xlog", False)  # Extract xlog parameter
    x_max = params.pop("x_max", None)  # Extract x_max parameter
    params["color"] = "skyblue"
    params["stat"] = "density"  # Equivalent to `density=True` in plt.hist
    plot_ylabel = True if idx % num_cols == 0 else False  # Show y-label only for leftmost plots
    plot_1D_hist(
        name=name,
        data=data,
        xlog=xlog,
        x_max=x_max,
        ax=ax,
        plot_ylabel=plot_ylabel,
        plot_params=params,
    )

# Remove unused subplots
for i in range(len(hist_data), len(axes)): fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig(f"{save_path}/grid_histograms.png", dpi=600)

print("Done.")