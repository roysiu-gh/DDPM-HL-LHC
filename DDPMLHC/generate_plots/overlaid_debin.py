"""This base code for this file was written by Roy Siu, with augmentations by Claude 3.5 Sonnet."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from DDPMLHC.config import *
from DDPMLHC.generate_plots.overlaid_general import plot_combined_histograms_with_overlay

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# def plot_combined_histograms_with_overlay_debin(hist_data_list, bin_values, save_path, pure=False):
#     colors = ['blue', 'orange', 'green', 'red', 'purple'][:len(bin_values) + (1 if pure else 0)]
#     alphas = np.linspace(0.7, 0.3, len(bin_values) + (1 if pure else 0))
    
#     fig, axes = plt.subplots(1, 2, figsize=(11, 5))
#     first_dataset = hist_data_list[0]
    
#     for idx, ax in enumerate(axes):
#         entry_ref = first_dataset[idx]
#         plot_params_ref = entry_ref.get("plot_params", {}).copy()
        
#         xlog = plot_params_ref.pop("xlog", False)
#         x_min = plot_params_ref.pop("x_min", None)
#         x_max = plot_params_ref.pop("x_max", None)
#         bins = plot_params_ref.pop("bins", 50)
        
#         if xlog:
#             bin_edges = np.logspace(np.log10(x_min or 1), np.log10(x_max), bins)
#         else:
#             bin_edges = np.linspace(x_min, x_max, bins)
        
#         for i, (hist_data, color, alpha) in enumerate(zip(hist_data_list, colors, alphas)):
#             entry = hist_data[idx]
            
#             if i == 0 and pure:
#                 label = "Pre-Grid"
#             else:
#                 bin_idx = i - 1 if pure else i
#                 label = f"${bin_values[bin_idx]} \\times {bin_values[bin_idx]}$"
            
#             sb.histplot(entry["data"], ax=ax, stat="density",
#                        bins=bin_edges, color=color, 
#                        label=label, alpha=alpha,
#                        edgecolor='black', linewidth=0.2)
        
#         if xlog:
#             ax.set_xscale("log")
#             if idx == 1:  # For p_T plot
#                 ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#                 ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
#                 ax.set_xticks([200, 300, 400, 500, 600, 700])
                
#         ax.set_xlim(left=x_min, right=x_max)
#         ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
#         ax.yaxis.offsetText.set_visible(False)
#         ax.set_xlabel(entry_ref["name"], fontsize=14)
        
#         if idx == 0:
#             ax.set_ylabel("Frequency Density ($\\times 10^{-2}$)", fontsize=12)
#         else:
#             ax.set_ylabel("")
#             ax.legend(fontsize=14, frameon=False)
    
#     plt.tight_layout()
    
#     plt.savefig(f"{save_path}/{filename}", dpi=600)
#     plt.close(fig)
#     print(f"Done {filename}.")

def create_overlay_plots_debin(bin_values, mass_max=250, save_path=None, mu=0, pure=False, pure_path=None):
    """
    Create overlay plots for specified bin values with optional pure data comparison.
    """
    pure_data_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level.csv"

    if len(bin_values) > 5:
        raise ValueError("Maximum 5 bin values supported")
    
    save_path = save_path or f"{CWD}/data/3-grid/mu0/overlaid"
    filename = f"overlaid_pure_bins{'_'.join(map(str, bin_values))}{'_incNoGrid' if pure else ''}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Define hist_params at the beginning
    hist_params = [
        {"name": "(a) Mass [GeV]", "col": 6, 
         "params": {"bins": 50, "x_min": 0, "x_max": mass_max}},
        {"name": "(b) Transverse momentum $p_T$ [GeV]", "col": 7, 
         "params": {"bins": 50, "x_min": 200, "x_max": 500}},
    ]
    
    # Initialize list to store all datasets
    list_of_params_all = []
    
    # First load and add pure data if requested
    if pure:
        pure_data = np.genfromtxt(pure_data_path, delimiter=",", encoding="utf-8", 
                                 skip_header=1, max_rows=MAX_DATA_ROWS)
        
        pure_dataset = [{
            "name": param["name"],
            "data": pure_data[:, param["col"]],
            "plot_params": param["params"],
        } for param in hist_params]
        
        list_of_params_all.append(pure_dataset)
    
    # Then load and add grid datasets
    events_data = {bin: np.genfromtxt(f"{CWD}/data/3-grid/mu0/noisy_mu{mu}_event_level_from_grid{bin}.csv",
                                    delimiter=",", encoding="utf-8", skip_header=1,
                                    max_rows=MAX_DATA_ROWS) for bin in bin_values}
    
    # Add grid datasets
    for bins in bin_values:
        dataset = [{
            "name": param["name"],
            "data": events_data[bins][:, param["col"]],
            "plot_params": param["params"],
        } for param in hist_params]
        list_of_params_all.append(dataset)
    
    labels = ["Unpixelised"] + [ f"${bin}\\times {bin}$" for bin in bin_values ]
    
    plot_combined_histograms_with_overlay(
        list_of_params_all, 
        labels, 
        save_path = f"{save_path}/{filename}",
    )
