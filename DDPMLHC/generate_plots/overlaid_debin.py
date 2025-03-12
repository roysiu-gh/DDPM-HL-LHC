"""This base code for this file was written by Roy Siu, with augmentations by Claude 3.5 Sonnet."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from DDPMLHC.config import *
from DDPMLHC.generate_plots.overlaid_general import plot_combined_histograms_with_overlay

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

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
