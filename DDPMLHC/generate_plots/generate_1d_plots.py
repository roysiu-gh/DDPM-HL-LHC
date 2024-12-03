"""1D Histograms"""

# Import constants
from DDPMLHC.config import *

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from cycler import cycler

# Local imports
from DDPMLHC.calculate_quantities import *
line_thickness = 3.0
axes_thickness = 4.0
leg_size = 30

# ======= global matplotlib params =====
custom_params ={'axes.grid' : False}
sb.set_theme(style="ticks", rc=custom_params)
# plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 16  # Set default font size (optional)

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# === BEGIN Reading in Data ===
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

# Calculate quantities
# Get data
num = len(tt)
jet_ids = tt[:, 0]
momenta = tt[:, 3:]
px = tt[:, 3]
py = tt[:, 4]
pz = tt[:, 5]
# px, py, pz = momenta[0], momenta[1], momenta[2]
p_mag = p_magnitude(px, py, pz)
# Basic stats
eta = pseudorapidity(p_mag, pz)
p_T = np.sqrt(px**2 + py**2)
# Stats for jets
jet_enes, jet_pxs, jet_pys, jet_pzs = calculate_four_momentum_massless(jet_ids, px, py, pz)
jet_p2s = contraction(jet_enes, jet_pxs, jet_pys, jet_pzs)
jet_masses = np.sqrt(jet_p2s)

# Kinda fun to print
for jet_id in range(0, len(jet_enes), 132):
    jet_ene, jet_px, jet_py, jet_pz = jet_enes[jet_id], jet_pxs[jet_id], jet_pys[jet_id], jet_pzs[jet_id]
    jet_mass = jet_masses[jet_id]
    print(f"Jet ID: {jet_id}, Total 4-Momenta: [{jet_ene:.3f}, {jet_px:.3f}, {jet_py:.3f}, {jet_pz:.3f}], Mass: {jet_mass:.3f}")

# Define the save path and plot characteristics
save_path = f"{CWD}/data/plots/1D_histograms/"
DEFAULT_PLOT_PARAMS = {
    "bins": 100,
    "color": "skyblue",
    "edgecolor": "none",
    "kde": False,
    "stat": "density",  # Equivalent to `density=True` in plt.hist
}

def plot_1D_hist(name, data, xlog=False, plot_params=DEFAULT_PLOT_PARAMS, save_path=save_path, save_filename="out", x_min=None, x_max=None, ax=None):
    """Plot a 1D histogram. Optionally plot on a provided axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure if no axes provided
    else:
        fig = None  # If ax provided, do not create a new figure
    
    # Copy default params if not already assigned
    plot_params.update({ k:v for k, v in DEFAULT_PLOT_PARAMS.items() if k not in plot_params})

    sb.histplot(data, ax=ax,log_scale=(xlog, False), **plot_params)

    ax.set_xlabel(name, fontsize=16)
    ax.set_ylabel("Frequency Density", fontsize=16)
    ax.set_xlim(left=x_min, right=x_max)

    if save_path and fig:
        fig.savefig(f"{save_path}/{save_filename}.png", dpi=600)

    return fig, ax

# Plot singles

print("Plotting histograms...")
plot_1D_hist("Momentum Magnitudes [GeV]",
             p_mag,         save_filename="p_mag",      xlog=True)
plot_1D_hist("Pseudorapidity $\eta$",
             eta,           save_filename="eta",)
plot_1D_hist("Transverse Momentum $p_T$ [GeV]",
             p_T,           save_filename="p_T",        xlog=True)

plot_1D_hist("JetMass [GeV]",
             jet_masses,    save_filename="jet_mass",   x_max=250)


# Plot multiplot

print("Plotting multiplot...")
fig, ax = plot_1D_hist(
    name="Momentum Magnitudes [GeV]",
    data=p_mag,
    xlog=True,
    save_filename="p_mag",
)
plt.show()

hist_data = [
    ("Momentum Magnitudes [GeV]", p_mag, {"xlog": True}),
    ("Pseudorapidity $\eta$", eta, {}),
    ("Transverse Momentum $p_T$ [GeV]", p_T, {"xlog": True}),

    ("Jet Mass [GeV]", jet_masses, {"x_max": 250}),
]

num_rows, num_cols = 2, 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
axes = axes.flatten()  # Flatten to iterate over axes

for ax, (name, data, params) in zip(axes, hist_data):
    xlog = params.pop("xlog", False)  # Extract xlog parameter
    x_max = params.pop("x_max", None)  # Extract x_max parameter
    params["color"] = "skyblue"
    params["stat"] = "density"  # Equivalent to `density=True` in plt.hist
    plot_1D_hist(
        name=name,
        data=data,
        xlog=xlog,
        x_max=x_max,
        ax=ax,
        plot_params=params,
    )

# Remove unused subplots
for i in range(len(hist_data), len(axes)): fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig(f"{save_path}/grid_histograms.png", dpi=600)

print("Done.")