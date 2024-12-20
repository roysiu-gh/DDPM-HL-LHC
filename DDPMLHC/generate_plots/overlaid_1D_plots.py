"""This file was """

# Package imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Local imports
from DDPMLHC.config import *

def plot_combined_histograms_with_overlay(hist_data_mu0, hist_data_mu100, hist_data_mu200, save_path):
    """Plot mass, eta, p_T for mu=0, mu=100, and mu=200 on the same plot for 3 variables."""
    print("Plotting combined histogram with overlay...")
    num_rows, num_cols = 1, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5))
    axes = axes.flatten()  # Flatten to iterate over axes

    for idx, (ax, entry_mu0, entry_mu100, entry_mu200) in enumerate(zip(axes, hist_data_mu0, hist_data_mu100, hist_data_mu200)):
        # Extract common parameters
        plot_params_mu0 = entry_mu0.get("plot_params", {}).copy()
        plot_params_mu100 = entry_mu100.get("plot_params", {}).copy()
        plot_params_mu200 = entry_mu200.get("plot_params", {}).copy()

        # Axis limits and log scale
        xlog = plot_params_mu0.pop("xlog", False)
        plot_params_mu100.pop("xlog", False)
        plot_params_mu200.pop("xlog", False)
        x_min = plot_params_mu0.pop("x_min", None)
        plot_params_mu100.pop("x_min", None)
        plot_params_mu200.pop("x_min", None)
        x_max = plot_params_mu0.pop("x_max", None)
        plot_params_mu100.pop("x_max", None)
        plot_params_mu200.pop("x_max", None)

        # Plot mu=0 data (normalized)
        sb.histplot(
            entry_mu0["data"],
            ax=ax,
            stat="density",  # Normalize to density
            **plot_params_mu0,
            color="blue",
            label="mu=0",
            alpha=0.7,
        )

        # Plot mu=100 data (normalized)
        sb.histplot(
            entry_mu100["data"],
            ax=ax,
            stat="density",  # Normalize to density
            **plot_params_mu100,
            color="orange",
            label="mu=100",
            alpha=0.5,
        )

        # Plot mu=200 data (normalized)
        sb.histplot(
            entry_mu200["data"],
            ax=ax,
            stat="density",  # Normalize to density
            **plot_params_mu200,
            color="green",
            label="mu=200",
            alpha=0.3,
        )

        # Configure axis labels and scaling
        if xlog:
            ax.set_xscale("log")
        ax.set_xlim(left=x_min, right=x_max)

        # Set titles and legends
        ax.set_title(entry_mu0["name"], fontsize=14)
        ax.legend(fontsize=12)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{save_path}/combined_histograms", dpi=600)
    plt.show()



# MAX_DATA_ROWS = 10_000



# Define paths for the datasets
mu0_event_stats_path = f"{CWD}/data/2-intermediate/ttbar_jets.csv"
mu100_event_stats_path = f"{CWD}/data/2-intermediate/noisy_event_stats_mu100.csv"
mu200_event_stats_path = f"{CWD}/data/2-intermediate/noisy_event_stats_mu200.csv"
save_path = f"{CWD}/data/plots/1D_histograms/"

# Load datasets
events_dat_mu0 = np.genfromtxt(
    mu0_event_stats_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
events_dat_mu100 = np.genfromtxt(
    mu100_event_stats_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
events_dat_mu200 = np.genfromtxt(
    mu200_event_stats_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

# Extract relevant data columns
event_eta_mu0 = events_dat_mu0[:, 4]
event_mass_mu0 = events_dat_mu0[:, 6]
event_pT_mu0 = events_dat_mu0[:, 7]

event_eta_mu100 = events_dat_mu100[:, 4]
event_mass_mu100 = events_dat_mu100[:, 6]
event_pT_mu100 = events_dat_mu100[:, 7]

event_eta_mu200 = events_dat_mu200[:, 4]
event_mass_mu200 = events_dat_mu200[:, 6]
event_pT_mu200 = events_dat_mu200[:, 7]

# Prepare histogram data for mu=0
list_of_params_mu0 = [
    {
        "name": "Mass [GeV]",
        "data": event_mass_mu0,
        "plot_params": {"x_max": 250},
        "save_filename": "event_mass_mu0",
    },
    {
        "name": "Pseudorapidity $\\eta$",
        "data": event_eta_mu0,
        "plot_params": {"bins": 50, "x_min": -1, "x_max": 1},
        "save_filename": "event_eta_mu0",
    },
    {
        "name": "Transverse Momentum $p_T$ [GeV]",
        "data": event_pT_mu0,
        "plot_params": {"xlog": True, "bins": 50, "x_max": 1000},
        "save_filename": "event_pT_mu0",
    },
]

# Prepare histogram data for mu=100
list_of_params_mu100 = [
    {
        "name": "Mass [GeV]",
        "data": event_mass_mu100,
        "plot_params": {"x_max": 250},
        "save_filename": "event_mass_mu100",
    },
    {
        "name": "Pseudorapidity $\\eta$",
        "data": event_eta_mu100,
        "plot_params": {"bins": 50},
        "save_filename": "event_eta_mu100",
    },
    {
        "name": "Transverse Momentum $p_T$ [GeV]",
        "data": event_pT_mu100,
        "plot_params": {"xlog": True, "bins": 50, "x_max": 1000},
        "save_filename": "event_pT_mu100",
    },
]

# Prepare histogram data for mu=200
list_of_params_mu200 = [
    {
        "name": "Mass [GeV]",
        "data": event_mass_mu200,
        "plot_params": {"x_max": 250},
        "save_filename": "event_mass_mu200",
    },
    {
        "name": "Pseudorapidity $\\eta$",
        "data": event_eta_mu200,
        "plot_params": {"bins": 50},
        "save_filename": "event_eta_mu200",
    },
    {
        "name": "Transverse Momentum $p_T$ [GeV]",
        "data": event_pT_mu200,
        "plot_params": {"xlog": True, "bins": 50, "x_max": 1000},
        "save_filename": "event_pT_mu200",
    },
]

plot_combined_histograms_with_overlay(list_of_params_mu0, list_of_params_mu100, list_of_params_mu200, save_path)

