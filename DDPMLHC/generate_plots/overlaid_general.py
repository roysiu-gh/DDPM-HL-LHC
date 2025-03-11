"""This base code for this file was written by Roy Siu, with augmentations by Claude 3.5 Sonnet."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from DDPMLHC.config import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

def plot_combined_histograms_with_overlay(hist_data_list, labels, save_path, stat="density", FOOBAR=False):
    """Plot mass and p_T for multiple datasets with custom labels."""
    colors = ['blue', 'orange', 'green', 'red', 'purple'][:len(labels)]
    hatch_patterns = ['O', 'o', '.', '/', '\\'][:len(labels)]  # Different hatching patterns
    hatch_patterns = hatch_patterns[::-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    first_dataset = hist_data_list[0]
    
    for idx, ax in enumerate(axes):
        entry_ref = first_dataset[idx]
        plot_params_ref = entry_ref.get("plot_params", {}).copy()
        
        # Extract parameters needed for bin calculation
        xlog = plot_params_ref.pop("xlog", False)
        x_min = plot_params_ref.pop("x_min", None)
        x_max = plot_params_ref.pop("x_max", None)
        bins = plot_params_ref.pop("bins", 50)
        
        # Calculate common bin edges for this axis
        if xlog:
            bin_edges = np.logspace(np.log10(x_min or 1), np.log10(x_max), bins)
        else:
            bin_edges = np.linspace(x_min, x_max, bins)
        
        for hist_data, label, color, hatch in zip(hist_data_list, labels, colors, hatch_patterns):
            entry = hist_data[idx]
            ax.hist(entry["data"], bins=bin_edges, 
                    label=label, edgecolor=color, 
                    facecolor='none', hatch=hatch, 
                    linewidth=1, density=(stat=="density"),
                    histtype="stepfilled")  # Use stepfilled for outer shape
        
        if xlog:
            ax.set_xscale("log")
            if idx == 1:  # For p_T plot
                ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
                ax.set_xticks([250, 300, 400, 500, 600, 700])
                
        ax.set_xlim(left=x_min, right=x_max)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
        ax.yaxis.offsetText.set_visible(False)
        ax.set_xlabel(entry_ref["name"], fontsize=14)
        
        if idx == 0:
            if stat == "count":
                ax.set_ylabel("Frequency", fontsize=12)
            elif stat == "density":
                ax.set_ylabel("Frequency Density", fontsize=12)
            else:
                raise ValueError
        else:
            ax.set_ylabel("")
            if FOOBAR:
                ax.legend(fontsize=14, frameon=False, bbox_to_anchor=(0.2, 0.6))
            else:
                ax.legend(fontsize=14, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}", dpi=600)
    plt.close(fig)

def create_overlay_plots_general(file_paths, labels, mass_max=250, save_path=None):
    """
    Create overlay plots for specified files with custom labels.
    
    Parameters:
    -----------
    file_paths : list
        List of paths to CSV files containing event data
    labels : list
        List of labels for each dataset
    mass_max : float
        Maximum mass value for plotting
    save_path : str, optional
        Path to save the output plots
    """
    if len(file_paths) > 5:
        raise ValueError("Maximum 5 datasets supported")
    
    save_path = save_path or f"{CWD}/data/plots/1D_histograms/overlaid_from_model/overlaid_comparison.png"
    
    # Load all datasets
    events_data = {path: np.genfromtxt(path,
                                     delimiter=",", encoding="utf-8", skip_header=1,
                                     max_rows=MAX_DATA_ROWS) for path in file_paths}
    
    hist_params = [
        {"name": "(a) Mass [GeV]", "col": 6, 
         "params": {"bins": 50, "x_min": 0, "x_max": mass_max}},
        {"name": "(b) Transverse Momentum $p_T$ [GeV]", "col": 7, 
         "params": {"xlog": True, "bins": 50, "x_min": 250, "x_max": 700}},
    ]
    
    list_of_params_all = [[{
        "name": param["name"],
        "data": events_data[path][:, param["col"]],
        "plot_params": param["params"],
    } for param in hist_params] for path in file_paths]
    
    plot_combined_histograms_with_overlay(list_of_params_all, labels, save_path, FOOBAR=True, stat="count")
