# Package imports
import numpy as np
import matplotlib as mpl
import polars as pl
from itertools import product
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *
from DDPMLHC.generate_plots.histograms_1d import plot_1d_histograms
from DDPMLHC.generate_plots.overlaid_1d import create_overlay_plots
from DDPMLHC.generate_plots.overlaid_debin import create_overlay_plots_debin
# from DDPMLHC.generate_plots.bmap import save_to_bmap
from DDPMLHC.generate_plots.resolution_plots import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# #################################################################################
# print("0 :: Loading original data")
# tt = np.genfromtxt(
#     TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# pu = np.genfromtxt(
#     PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# tt = EventSelector(tt)
# pu = EventSelector(pu)
# print("FINISHED loading data\n")

# # Ground truth ttbar jets
# NG_jet = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=0)
# NG_jet.save_event_level_data()
# # Second one to randomly generate and return pile-up events ONLY
# # NG_pu = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=0, pu_only=True)
# # Default NG just for convenience
# NG_default = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=200)
# NG_default.save_event_level_data()


def generate_event_level_gridded_jets(NG: NoisyGenerator,bins, mu=0, save_dir=INTERMEDIATE_PATH):
    NG.reset()
    NG.bins = bins
    NG.mu = mu
    gt_file = f"{save_dir}/noisy_mu{mu}_event_level_grid{bins}.csv"
    combined = []
    for idx, _ in enumerate(NG):
        # next(NG)
        grid = NG.get_grid(normalise=False)
        # NG.select_jet(idx)
        axis = NG.jet_axis
        enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=NG.bins)
        detas, dphis = decentre(axis, detas, dphis)
        pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
        # print("???")
        event_quantities = particle_momenta_to_event_level(enes, pxs, pys, pzs)
        event_mass, event_px, event_py, event_pz, event_eta, event_phi, event_pT = event_quantities
        
        event_level = np.array([
            idx,
            event_px,
            event_py,
            event_pz,
            event_eta,
            event_phi,
            event_mass,
            event_pT,
        ])
        
        combined.append(np.copy(event_level))
            
    all_data = np.vstack(combined)
    np.savetxt(
            gt_file,
            all_data,
            delimiter=",",
            header="event_id,px,py,pz,eta,phi,mass,p_T",
            comments="",
            fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
    )
    return all_data, gt_file

# for x in product([64], [200]):
#     generate_event_level_gridded_jets(NG=NG_jet, bins=x[0], mu=x[1])
# # Comparison against pure cts
noisy_path = f"{CWD}/data/2-intermediate/noisy_mu200_event_level.csv"

for x in product([16,32,64], [0.5,1]):
    bins = x[0]
    beta = x[1]
    best_case_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level_grid{bins}.csv"
  
    # beta = "0.5"
    reconstructed_path_beta05 = f"{CWD}/data/4-reconstruction/beta{beta}/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"
    # reconstructed_path_beta1 = f"{CWD}/data/4-reconstruction/beta1/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"
    reconstructed_path = reconstructed_path_beta05
    # First plot - comparison against pure cts
    mass_resolutions_orig = {
        "Ground truth": load_variable_data(best_case_path, "mass"),
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "mass"),
        "Denoised": load_variable_data(reconstructed_path, "mass")
    }

    pt_resolutions_orig = {
        "Ground truth": load_variable_data(best_case_path, "p_T"),
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "p_T"),
        "Denoised": load_variable_data(reconstructed_path, "p_T")
    }

    plot_resolutions(
        mass_resolutions_orig, pt_resolutions_orig,
        colors={
            "Ground truth": "black",
            "Noisy ($\mu = 200$)": "red",
            "Denoised": "blue"
        },
        use_log = True,
        save_path = f"{CWD}/data/plots/relative_resolutions/beta{beta}/resolution_grid{bins}_Unet{UNET_DIMS}_mass_gtORIG.pdf",
        legend_title=rf"${bins}\times {bins}$ grid"
    )

    # Second plot - comparison against best case
    mass_resolutions_best = {
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "mass", truth_path=best_case_path),
        "Denoised": load_variable_data(reconstructed_path, "mass", truth_path=best_case_path)
    }

    pt_resolutions_best = {
        "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "p_T", truth_path=best_case_path),
        "Denoised": load_variable_data(reconstructed_path, "p_T", truth_path=best_case_path)
    }

    plot_resolutions(
        mass_resolutions_best, pt_resolutions_best,
        colors={
            "Noisy ($\mu = 200$)": "red",
            "Denoised": "blue"
        },
        save_path = f"{CWD}/data/plots/relative_resolutions/beta{beta}/resolution_grid{bins}_Unet{UNET_DIMS}_mass_gtBEST.pdf",
        legend_title=rf"${bins}\times {bins}$ grid"
    )
