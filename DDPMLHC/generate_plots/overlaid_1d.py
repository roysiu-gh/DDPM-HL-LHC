"""This base code for this file was written by Roy Siu, with augmentations by Claude 3.5 Sonnet."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from DDPMLHC.config import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

def plot_combined_histograms_with_overlay(hist_data_list, mu_values, save_path):
    """Plot mass and p_T for multiple mu values."""
    colors = ['blue', 'orange', 'green', 'red', 'purple'][:len(mu_values)]
    alphas = np.linspace(0.7, 0.3, len(mu_values))
    
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
        
        for hist_data, mu, color, alpha in zip(hist_data_list, mu_values, colors, alphas):
            entry = hist_data[idx]
            sb.histplot(entry["data"], ax=ax, stat="density",
                       bins=bin_edges, color=color, 
                       label=f"$\\mu={mu}$", alpha=alpha,
                       edgecolor='black', linewidth=0.2)
        
        if xlog:
            ax.set_xscale("log")
            if idx == 1:  # For p_T plot
                ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
                ax.set_xticks([250, 300, 400, 500, 600, 700])
                
        ax.set_xlim(left=x_min, right=x_max)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
        ax.yaxis.offsetText.set_visible(False)  # Hide the offset text at the top
        ax.set_xlabel(entry_ref["name"], fontsize=14)
        
        # Set y-label with scale factor included
        if idx == 0:
            ax.set_ylabel("Frequency Density ($\\times 10^{-2}$)", fontsize=12)
        else:
            ax.set_ylabel("")
            ax.legend(fontsize=14, frameon=False)  # Only show legend for p_T plot
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/overlaid_mu_{'_'.join(map(str, mu_values))}", dpi=600)
    plt.close(fig)

def create_overlay_plots(mu_values, mass_max=250, save_path=None):
    """Create overlay plots for specified mu values."""
    if len(mu_values) > 5:
        raise ValueError("Maximum 5 mu values supported")
    
    save_path = save_path or f"{CWD}/data/plots/1D_histograms/overlaid"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Load all datasets and prepare parameters
    events_data = {mu: np.genfromtxt(f"{CWD}/data/2-intermediate/noisy_mu{mu}_event_level.csv",
                                    delimiter=",", encoding="utf-8", skip_header=1,
                                    max_rows=MAX_DATA_ROWS) for mu in mu_values}
    
    hist_params = [
        {"name": "Mass [GeV]", "col": 6, 
         "params": {"bins": 50, "x_min": 0, "x_max": mass_max}},
        {"name": "Transverse Momentum $p_T$ [GeV]", "col": 7, 
         "params": {"xlog": True, "bins": 50, "x_min": 250, "x_max": 700}},
    ]
    
    list_of_params_all = [[{
        "name": param["name"],
        "data": events_data[mu][:, param["col"]],
        "plot_params": param["params"],
        "save_filename": f"event_{param['name'].lower()}_mu{mu}"
    } for param in hist_params] for mu in mu_values]
    
    plot_combined_histograms_with_overlay(list_of_params_all, mu_values, save_path)