"""1D Histograms"""

# Import constants
from config import *

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb

# Local imports
from calculate_quantities import *

# ======= global matplotlib params =====
sb.set_theme(style="whitegrid")
plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
plt.rcParams["font.size"] = 12  # Set default font size (optional)

# === BEGIN Reading in Data ===
pile_up = np.genfromtxt(
    pileup_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

# Calculate quantities
# Get data
num = len(tt)
jet_ids = tt[:, 0]
p_mag = p_magnitude(tt[:, 3:])
px, py, pz = tt[:, 3], tt[:, 4], tt[:, 5]
# Basic stats
eta = pseudorapidity(p_mag, pz)
p_T = np.sqrt(px**2 + py**2)
# Stats for jets
jet_four_momenta = calculate_four_momentum_massless(jet_ids, px, py, pz)
jet_p2 = contraction(jet_four_momenta)
jet_mass = np.sqrt(jet_p2)

# Kinda fun to print
for jet_id in range(0, len(jet_four_momenta), 132):
    four_mmtm = jet_four_momenta[jet_id]
    p2 = jet_p2[jet_id]
    print(f"Jet ID: {jet_id}, Total 4-Momenta: [{four_mmtm[0]:.3f}, {four_mmtm[1]:.3f}, {four_mmtm[2]:.3f}, {four_mmtm[3]:.3f}], Contraction p^2: {p2:.3f}")

# Define the save path and plot characteristics
save_path = f"{CWD}/data/plots/1D_histograms/"
plot_params = {
    "bins": 100,
    "color": "skyblue",
    "edgecolor": "none",
    "kde": False,
    "stat": "density"  # Equivalent to `density=True` in plt.hist
}

def plot_1D_hist(name, data, xlog=False, is_jet=False, plot_params=plot_params, save_path=save_path, save_filename="out"):
    parjet = "Jets'" if is_jet else "Particles'"
    num = len(data)
    plt.figure(figsize=(10, 6))
    if xlog:
        plt.xscale("log")
        plot_params = plot_params.copy()
        plot_params["bins"] = np.logspace(np.log10(0.1),np.log10(3.0), 50)
    sb.histplot(data, **plot_params)
    plt.title(f"Normalised Histogram of {num} {parjet} {name}")
    plt.xlabel(name)
    plt.ylabel("Frequency Density")
    plt.grid(axis="y", alpha=0.75)
    plt.savefig(f"{save_path}/{save_filename}.png", dpi=600)

print("Plotting histograms...")
plot_1D_hist("Momentum Magnitudes (\si{\giga\electronvolt})",           p_mag,      save_filename="p_mag",                      xlog=True)
plot_1D_hist("Pseudorapidity $\eta$",                                   eta,        save_filename="eta",                      )
plot_1D_hist("Transverse Momentum $p_T$ (\si{\giga\electronvolt})",     p_T,        save_filename="p_T",                        xlog=True)
plot_1D_hist("($p^2$) (\si{\giga\electronvolt^2})",                     jet_p2,     save_filename="jet_p2",     is_jet=True,    xlog=True)
plot_1D_hist("Mass (\si{\giga\electronvolt})",                          jet_mass,   save_filename="jet_mass",   is_jet=True,    )
print("Done.")


