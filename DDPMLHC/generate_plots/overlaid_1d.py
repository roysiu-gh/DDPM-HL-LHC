"""This base code for this file was written by Roy Siu, with augmentations by Claude 3.5 Sonnet."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from DDPMLHC.config import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

def plot_combined_histograms_with_overlay(hist_data_list, mu_values, save_path):
    """Plot mass, eta, p_T for multiple mu values."""
    colors = ['blue', 'orange', 'green', 'red', 'purple'][:len(mu_values)]
    alphas = np.linspace(0.7, 0.3, len(mu_values))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    first_dataset = hist_data_list[0]
    
    for idx, ax in enumerate(axes):
        entry_ref = first_dataset[idx]
        plot_params_ref = entry_ref.get("plot_params", {}).copy()
        xlog, x_min, x_max = [plot_params_ref.pop(k, None) for k in ["xlog", "x_min", "x_max"]]
        
        for hist_data, mu, color, alpha in zip(hist_data_list, mu_values, colors, alphas):
            entry = hist_data[idx]
            plot_params = {k:v for k,v in entry.get("plot_params", {}).items() 
                         if k not in ["xlog", "x_min", "x_max"]}
            
            sb.histplot(entry["data"], ax=ax, stat="density", **plot_params,
                       color=color, label=f"$\\mu={mu}$", alpha=alpha)
        
        if xlog: ax.set_xscale("log")
        ax.set_xlim(left=x_min, right=x_max)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax.set_xlabel(entry_ref["name"], fontsize=14)
        ax.set(ylabel="Frequency Density" if idx == 0 else "")
        if idx == 0: ax.yaxis.label.set_size(12)
        ax.legend(fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/overlaid_mu_{'_'.join(map(str, mu_values))}", dpi=600)
    plt.close(fig)

def create_overlay_plots(mu_values, save_path=None):
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
         "params": {"x_min": 0, "x_max": 250}},
        {"name": "Pseudorapidity $\\eta$", "col": 4, 
         "params": {"bins": 50, "x_min": -1, "x_max": 1}},
        {"name": "Transverse Momentum $p_T$ [GeV]", "col": 7, 
         "params": {"xlog": True, "bins": 50, "x_max": 1000}},
    ]
    
    list_of_params_all = [[{
        "name": param["name"],
        "data": events_data[mu][:, param["col"]],
        "plot_params": param["params"],
        "save_filename": f"event_{param['name'].lower()}_mu{mu}"
    } for param in hist_params] for mu in mu_values]
    
    plot_combined_histograms_with_overlay(list_of_params_all, mu_values, save_path)
