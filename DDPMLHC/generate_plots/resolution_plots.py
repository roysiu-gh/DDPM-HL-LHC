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
    return (comparison - ground_truth) / ground_truth

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
                    bins=BMAP_SQUARE_SIDE_LENGTH, save_path=None,
                    mass_cutoff=(-1, 4), pt_cutoff=(-1, 2), use_log=False, legend_title=None):
    
    if legend_title is None:
        legend_title = rf"${bins}\times {bins}$ grid"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mass plot
    for label, res in mass_resolutions.items():
        plot_data = res[(res >= mass_cutoff[0]) & (res < mass_cutoff[1])]
        # Calc stats
        bias = np.mean(plot_data)
        resolution = np.std(plot_data)
        stat_label = f"{label}\n$\zeta={bias:.3f}$\n$\\rho={resolution:.3f}$"
        
        ax1.hist(plot_data, 
                bins=np.linspace(mass_cutoff[0], mass_cutoff[1], 50),
                label=stat_label, 
                histtype="step",
                density=True,
                color=colors.get(label) if colors else None)
    
    ax1.axvline(x=0, color="black", linestyle="--")  # vertical line
    ax1.set_xlabel("(a) mass response")
    ax1.set_ylabel("Density")
    if use_log: ax1.set_yscale("log")
    if legend_title == "":
        ax1.legend(loc="upper right", labelspacing=1.0)
    else:
        ax1.legend(loc="upper right", labelspacing=1.0, title=legend_title)
    
    # pT plot
    for label, res in pt_resolutions.items():
        plot_data = res[(res >= pt_cutoff[0]) & (res < pt_cutoff[1])]
        # Calc stats
        bias = np.mean(plot_data)
        resolution = np.std(plot_data)
        stat_label = f"{label}\n$\zeta={bias:.3f}$\n$\\rho={resolution:.3f}$"
        
        ax2.hist(plot_data, 
                bins=np.linspace(pt_cutoff[0], pt_cutoff[1], 50),
                label=stat_label, 
                histtype="step",
                density=True,
                color=colors.get(label) if colors else None)
    
    ax2.axvline(x=0, color="black", linestyle="--")  # vertical line
    ax2.set_xlabel(r"(b) $p_T$ response")
    if use_log: ax2.set_yscale("log")
    if legend_title == "":
        ax2.legend(loc="upper right", labelspacing=1.0)
    else:
        ax2.legend(loc="upper right", labelspacing=1.0, title=legend_title)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{CWD}/data/plots/relative_resolutions/resolutions_test.pdf"
    plt.savefig(save_path)
    plt.close()
