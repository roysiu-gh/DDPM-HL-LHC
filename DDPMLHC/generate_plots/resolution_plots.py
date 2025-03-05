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

def load_mass_data(data_path, truth_path=None):
    if truth_path is None:
        truth_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level.csv"
    truth = pl.read_csv(truth_path)["mass"].to_numpy()
    data = pl.read_csv(data_path)["mass"].to_numpy()
    
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

def plot_resolutions(resolutions, colors=None, grid_size=BMAP_SQUARE_SIDE_LENGTH, save_path=None):
    """Plot mass resolutions"""
    if colors is None:
        colors = {"Best case": "black", "Noisy ($\mu = 200$)": "red", "Reconstructed": "blue"}
    
    # Make plot
    plt.figure(figsize=(10, 6))
    
    for label, res in resolutions.items():
        plot_data = res[res < 5]
        plt.hist(plot_data, bins=50, 
                label=label, 
                edgecolor=colors[label], 
                alpha=0.7,
                histtype="step")
    
    # plt.xlabel(r"$\frac{\left|m_{\mu}^{j} - m_{0}^{j}\right|}{m_{0}^{j}}$")
    plt.xlabel(r"Relative resolution")
    plt.ylabel("Counts")
    plt.yscale("log")
    
    plt.legend(title=rf"${grid_size}\times {grid_size}$ grid", loc="upper right")
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{CWD}/data/plots/relative_resolutions/mass_resolution_test.pdf"
    plt.savefig(save_path)
    plt.close()