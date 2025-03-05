# Package imports
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
# Local imports
from DDPMLHC.config import *

def relative_resolution(ground_truth, comparison):
    if len(ground_truth) != len(comparison):
        raise IndexError(f"Lengths mismatched: ground_truth ({len(ground_truth)}) != comparison ({len(comparison)})")
    if np.any(ground_truth == 0):
        raise ZeroDivisionError(f"Zero(s) in ground truth")
    return np.abs(comparison - ground_truth) / ground_truth

def load_variable_data(data_path, variable, truth_path=None):
    if truth_path is None:
        truth_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level.csv"
    truth = pl.read_csv(truth_path)[variable].to_numpy()
    data = pl.read_csv(data_path)[variable].to_numpy()
    
    if len(data) != len(truth):
        print(f"WARNING - Length mismatch in data: truth {len(truth)} != data {len(data)}. Using shortest length.")
        min_len = min(len(data), len(truth))
        data = data[:min_len]
        truth = truth[:min_len]
    
    # Delete known bad event
    bad_idx = 24716
    truth = np.delete(truth, bad_idx)
    data = np.delete(data, bad_idx)

    return relative_resolution(truth, data)

def plot_resolutions(mass_resolutions, pt_resolutions, colors=None, 
                    grid_size=BMAP_SQUARE_SIDE_LENGTH, save_path=None,
                    mass_cutoff=4, pt_cutoff=2, use_log=False, legend_title=None):
    
    # Default legend title if none provided
    if legend_title is None:
        legend_title = rf"${grid_size}\times {grid_size}$ grid"
    
    # Make plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mass plot
    for label, res in mass_resolutions.items():
        plot_data = res[res < mass_cutoff]
        ax1.hist(plot_data, 
                bins=np.linspace(0, mass_cutoff, 50),  # Explicit bin range
                label=label, 
                histtype="step",
                density=True)  # Normalize to compare shapes
    
    ax1.set_xlabel("(a) mass relative resolution")
    ax1.set_ylabel("Density")
    if use_log: ax1.set_yscale("log")
    if legend_title == "":
        ax1.legend(loc="upper right")
    else:
        ax1.legend(title=legend_title, loc="upper right")
    
    # pT plot
    for label, res in pt_resolutions.items():
        plot_data = res[res < pt_cutoff]
        ax2.hist(plot_data, 
                bins=np.linspace(0, pt_cutoff, 50),  # Explicit bin range
                label=label, 
                histtype="step",
                density=True)  # Normalize to compare shapes
    
    ax2.set_xlabel(r"(b) $p_T$ relative resolution")
    if use_log: ax2.set_yscale("log")
    if legend_title == "":
        ax2.legend(loc="upper right")
    else:
        ax2.legend(title=legend_title, loc="upper right")
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{CWD}/data/plots/relative_resolutions/resolutions_test.pdf"
    plt.savefig(save_path)
    plt.close()
