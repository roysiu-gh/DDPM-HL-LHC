# Package imports
import numpy as np
import matplotlib as mpl
import polars as pl
import warnings
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *
from DDPMLHC.generate_plots.histograms_1d import plot_1d_histograms
from DDPMLHC.generate_plots.overlaid_1d import create_overlay_plots
from DDPMLHC.generate_plots.overlaid_debin import create_overlay_plots_debin
from DDPMLHC.generate_plots.bmap import save_to_bmap

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# MAX_DATA_ROWS = 100_000

# # === Read in data
# print("0 :: Loading original data")
# tt = np.genfromtxt(
#     TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# pile_up = np.genfromtxt(
#     PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# tt = EventSelector(tt)
# pile_up = EventSelector(pile_up)
# print("FINISHED loading data\n")

#################################################################################

# # Calculate what percent of PUs are empty
# pu_ids = pile_up[:, 0]
# total_pileups = pu_ids[-1]
# print(f"Final number in the first column: {total_pileups}")
# unique_vals = len(np.unique(pu_ids))
# print(f"Number of unique values in the first column: {unique_vals}")
# print( f"Percentage of non-empty pile-ups {(unique_vals/total_pileups)*100}" )

#################################################################################

# mus = [0, 1, 3, 5, 10, 15, 25, 30, 50, 75, 100, 125, 150, 175, 200]
# for mu in mus:
#     cur_generator = NoisyGenerator(tt, pile_up, mu=mu)
#     cur_generator.save_event_level_data()
#     plot_1d_histograms(mu=mu)

#################################################################################

# create_overlay_plots([0, 5, 10, 15, 30])
# create_overlay_plots([0, 10, 30, 50])
# create_overlay_plots([0, 25, 50, 75, 100], mass_max=300)
# create_overlay_plots([0, 50, 100, 150, 200], mass_max=400)

#################################################################################

# mus = [0, 50, 200, 500]
# # mus = [200]

# for mu in mus:
#     generator = NoisyGenerator(tt, pile_up, mu=mu)
#     # next(generator)  # Load jet 0
#     generator.select_jet(0)
#     save_to_bmap(generator.vectorise(), jet_no=generator.event_id, mu=generator.mu)
#     generator.visualise_current_event()
#     generator.visualise_current_event(show_pdgids=True)

#     generator.select_jet(1)
#     save_to_bmap(generator.vectorise(), jet_no=generator.event_id, mu=generator.mu)
#     generator.visualise_current_event(particle_scale_factor=1200, )
#     generator.visualise_current_event(particle_scale_factor=1200, show_pdgids=True)

#     generator.select_jet(42)
#     save_to_bmap(generator.vectorise(), jet_no=generator.event_id, mu=generator.mu)
#     generator.visualise_current_event()
#     generator.visualise_current_event(show_pdgids=True)

#     generator.select_jet(493)
#     save_to_bmap(generator.vectorise(), jet_no=generator.event_id, mu=generator.mu)
#     generator.visualise_current_event(particle_scale_factor=2000, )
#     generator.visualise_current_event(particle_scale_factor=2000, show_pdgids=True)

#################################################################################

# mu = 0
# output_path = f"{CWD}/data/3-grid/mu{mu}/"
# output_filename = f"noisy_mu{mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"
# output_filepath = f"{output_path}/{output_filename}"

# ###

# generator = NoisyGenerator(tt, pile_up, mu=mu)
# combined = []

# for idx, _ in enumerate(generator):
#     grid = generator.get_grid(normalise=False)
#     axis = generator.jet_axis
#     if idx == 0:
#         print(f"grid.shape {grid.shape}")
    
#     enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=generator.bins)
#     detas, dphis = decentre(axis, detas, dphis)
#     pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
#     event_quantities = particle_momenta_to_event_level(enes, pxs, pys, pzs)
#     event_mass, event_px, event_py, event_pz, event_eta, event_phi, event_pT = event_quantities

#     event_level = np.array([
#         idx,
#         event_px,
#         event_py,
#         event_pz,
#         event_eta,
#         event_phi,
#         event_mass,
#         event_pT,
#     ])

#     combined.append(np.copy(event_level))

# all_data = np.vstack(combined)

# np.savetxt(
#     output_filepath,
#     all_data,
#     delimiter=",",
#     header="event_id,px,py,pz,eta,phi,mass,p_T",
#     comments="",
#     fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
# )

# ###

# plot_1d_histograms(mu, event_stats_path=output_filepath, output_path=f"{output_path}/grid{BMAP_SQUARE_SIDE_LENGTH}")

#################################################################################

# mu = 0
# output_path = f"{CWD}/data/3-grid/mu{mu}/"
# for bins in [1,2,4,8,16,32,64,128,256]:
#     output_filename = f"noisy_mu{mu}_event_level_from_grid{bins}.csv"
#     output_filepath = f"{output_path}/{output_filename}"

#     plot_1d_histograms(mu, event_stats_path=output_filepath, output_path=f"{output_path}/grid{bins}")

#################################################################################

# create_overlay_plots_debin([4,8,16,32])
# create_overlay_plots_debin([4,16,64,256])
# create_overlay_plots_debin([4,16,64,256], pure=True)
# create_overlay_plots_debin([4,8,16,256], pure=True)
# create_overlay_plots_debin([8,16,256], pure=True)
# create_overlay_plots_debin([2,4,8])
# create_overlay_plots_debin([2,4,8,256], pure=True)

#################################################################################

"""Base plotting code is original. Functionality augmented by Calude 3.5."""

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
    
    # Remove known bad event
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
        if len(plot_data) > 0:
            plt.hist(plot_data, bins=50, 
                    label=label, 
                    edgecolor=colors[label], 
                    alpha=0.7,
                    histtype="step")
        else:
            warnings.warn(f"No data to plot for {label}")
    
    plt.xlabel(r'$\frac{\left|m_{\mu}^{j} - m_{0}^{j}\right|}{m_{0}^{j}}$')
    plt.ylabel("Counts")
    plt.yscale("log")
    
    plt.legend(title=rf"${grid_size}\times {grid_size}$ grid", loc="upper right")
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{CWD}/data/plots/relative_resolutions/mass_resolution_test.pdf"
    plt.savefig(save_path)
    plt.close()

#################################################################################

# Define file paths
best_case_path = f"{CWD}/data/3-grid/mu0/noisy_mu0_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"
noisy_path = f"{CWD}/data/2-intermediate/noisy_mu200_event_level.csv"
reconstructed_path = f"{CWD}/data/4-reconstruction/reconstructed_mu{200}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}_Unet{UNET_DIMS}.csv"
# print(reconstructed_path)

best_case_res = load_mass_data(best_case_path)
noisy_res = load_mass_data(noisy_path)
reconstructed_res = load_mass_data(reconstructed_path)

resolutions = {
    "Best case": best_case_res,
    "Noisy ($\mu = 200$)": noisy_res,
    "Reconstructed": reconstructed_res
}

plot_resolutions(
    resolutions=resolutions,
    colors={
        "Best case": "black",
        "Noisy ($\mu = 200$)": "red",
        "Reconstructed": "blue"
    },
    save_path = f"{CWD}/data/plots/relative_resolutions/mass_resolution_test.pdf"
)
