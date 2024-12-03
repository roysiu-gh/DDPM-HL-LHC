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
plot_params = {
    "bins": 100,
    "color": "skyblue",
    "edgecolor": "none",
    "kde": False,
    "stat": "density",  # Equivalent to `density=True` in plt.hist
}
def plot_1D_hist(name, data, xlog=False, is_jet=False, plot_params=plot_params, save_path=save_path, save_filename="out", x_min=0, x_max=None):
    parjet = "Jets'" if is_jet else "Particles'"
    num = len(data)
    plt.figure(figsize=(10, 6))
    if xlog:
        plt.xscale("log")
        plot_params = plot_params.copy()
        # plot_params["bins"] = np.logspace(np.log10(0.1),np.log10(3.0), 50)
    sb.histplot(data, **plot_params)
    # plt.title(f"Normalised Histogram of {num} {parjet} {name}")
    plt.xlabel(name, fontsize=label_fontsize)
    plt.xlim(left=x_min, right=x_max)
    plt.ylabel("Frequency Density", fontsize=label_fontsize)
    # plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{save_filename}.png", dpi=600)

print("Plotting histograms...")
plot_1D_hist("Momentum Magnitudes [GeV]",           p_mag,      save_filename="p_mag",                      xlog=True)
plot_1D_hist("Pseudorapidity $\eta$",                                   eta,        save_filename="eta",                      x_min=None)
plot_1D_hist("Transverse Momentum $p_T$ [GeV]",     p_T,        save_filename="p_T",                        xlog=True)

plot_1D_hist("($p^2$) [GeV$^2$]",                     jet_p2s,     save_filename="jet_p2",     is_jet=True,    xlog=True)
plot_1D_hist("Mass [GeV]",                          jet_masses,   save_filename="jet_mass",   is_jet=True,    x_max=250)
print("Done.")
