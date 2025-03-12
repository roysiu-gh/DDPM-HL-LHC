"""This base code for this file was written by Roy Siu, with augmentations by Claude 3.5 Sonnet."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from DDPMLHC.config import *
from DDPMLHC.generate_plots.overlaid_general import plot_combined_histograms_with_overlay

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

def create_overlay_plots(mu_values, mass_max=250, save_path=None):
    """Create overlay plots for specified mu values."""
    if len(mu_values) > 5:
        raise ValueError("Maximum 5 mu values supported")
    
    save_path = save_path or f"{CWD}/data/plots/1D_histograms/overlaid/overlaid_mu_{'_'.join(map(str, mu_values))}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Load all datasets and prepare parameters
    events_data = {mu: np.genfromtxt(f"{CWD}/data/2-intermediate/noisy_mu{mu}_event_level.csv",
                                    delimiter=",", encoding="utf-8", skip_header=1,
                                    max_rows=MAX_DATA_ROWS) for mu in mu_values}
    
    hist_params = [
        {"name": "(a) Mass [GeV]", "col": 6, 
         "params": {"bins": 50, "x_min": 0, "x_max": mass_max}},
        {"name": "(b) Transverse momentum $p_T$ [GeV]", "col": 7, 
         "params": {"xlog": True, "bins": 50, "x_min": 250, "x_max": 700}},
    ]
    
    list_of_params_all = [[{
        "name": param["name"],
        "data": events_data[mu][:, param["col"]],
        "plot_params": param["params"],
        "save_filename": f"event_{param['name'].lower()}_mu{mu}"
    } for param in hist_params] for mu in mu_values]

    mu_values = [f"$\mu={i}$" for i in mu_values]
    
    plot_combined_histograms_with_overlay(list_of_params_all, mu_values, save_path)