# Package imports
import numpy as np
import matplotlib as mpl
import polars as pl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *
from DDPMLHC.generate_plots.histograms_1d import plot_1d_histograms
from DDPMLHC.generate_plots.overlaid_1d import create_overlay_plots
from DDPMLHC.generate_plots.overlaid_debin import create_overlay_plots_debin
from DDPMLHC.generate_plots.bmap import save_to_bmap
from DDPMLHC.generate_plots.resolution_plots import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

#################################################################################

# Comparison against pure cts
noisy_path = f"{CWD}/data/2-intermediate/noisy_mu200_event_level.csv"
for bins in [16,32,64]:
    best_case_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level_grid{bins}.csv"
  
    reconstructed_path_beta05 = f"{CWD}/data/4-reconstruction/beta0.5/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"
    reconstructed_path_beta1 = f"{CWD}/data/4-reconstruction/beta1/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"

    # First plot - comparison against pure cts
    mass_resolutions_orig = {
        "Best case": load_variable_data(best_case_path, "mass"),
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "mass"),
        "Reconstructed": load_variable_data(reconstructed_path_beta05, "mass")
    }

    pt_resolutions_orig = {
        "Best case": load_variable_data(best_case_path, "p_T"),
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "p_T"),
        "Reconstructed": load_variable_data(reconstructed_path_beta05, "p_T")
    }

    plot_resolutions(
        mass_resolutions_orig, pt_resolutions_orig,
        colors={
            "Best case": "black",
            "Noisy ($\mu = 200$)": "red",
            "Reconstructed": "blue"
        },
        use_log = True,
        save_path = f"{CWD}/data/plots/relative_resolutions/beta0.5/resolution_grid{bins}_Unet{UNET_DIMS}_mass_gtORIG.pdf",

        title=rf"${bins}\times {bins}$ grid"
    )

    # Second plot - comparison against best case
    mass_resolutions_best = {
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "mass", truth_path=best_case_path),
        "Reconstructed": load_variable_data(reconstructed_path_beta05, "mass", truth_path=best_case_path)
    }

    pt_resolutions_best = {
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "p_T", truth_path=best_case_path),
        "Reconstructed": load_variable_data(reconstructed_path_beta05, "p_T", truth_path=best_case_path)
    }

    plot_resolutions(
        mass_resolutions_best, pt_resolutions_best,
        colors={
            "Noisy ($\mu = 200$)": "red",
            "Reconstructed": "blue"
        },
        save_path = f"{CWD}/data/plots/relative_resolutions/beta0.5/resolution_grid{bins}_Unet{UNET_DIMS}_mass_gtBEST.pdf",
        title=rf"${bins}\times {bins}$ grid"
    )
