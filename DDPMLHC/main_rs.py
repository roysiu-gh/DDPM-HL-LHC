# Package imports
import numpy as np
import matplotlib as mpl
import polars as pl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *
from DDPMLHC.generate_plots.histograms_1d import plot_1d_histograms, plot_event_level_quantities_comparison, plot_particle_level_quantities_comparison, plot_particle_level_quantities_ttbar_only
from DDPMLHC.generate_plots.overlaid_1d import create_overlay_plots
from DDPMLHC.generate_plots.overlaid_debin import create_overlay_plots_debin
from DDPMLHC.generate_plots.bmap import plot_mu_comparison, save_to_bmap
from DDPMLHC.generate_plots.overlaid_general import create_overlay_plots_general
from DDPMLHC.generate_plots.resolution_plots import *

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

# # Find avg_PU_per_nonzero_event
# num_non_empty = len(np.unique(pile_up[:, 0]))
# num_parts = len(pile_up)
# avg_PU_per_nonzero_event = num_parts / num_non_empty
# print(f"avg_PU_per_nonzero_event: {avg_PU_per_nonzero_event}")
# # avg_PU_per_nonzero_event: 7.765886374696351

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

# Comparison against pure cts
best_case_path = f"{CWD}/data/3-grid/mu0/noisy_mu0_event_level_from_grid{16}.csv"
noisy_path = f"{CWD}/data/2-intermediate/noisy_mu200_event_level.csv"
reconstructed_path = f"{CWD}/data/4-reconstruction/reconstructed_mu{200}_event_level_from_grid{16}_Unet{UNET_DIMS}.csv"

# First plot - comparison against pure cts
mass_resolutions_orig = {
    "Best case": load_variable_data(best_case_path, "mass"),
    "Noisy, $\mu = 200$": load_variable_data(noisy_path, "mass"),
    "Denoised": load_variable_data(reconstructed_path, "mass")
}

pt_resolutions_orig = {
    "Best case": load_variable_data(best_case_path, "p_T"),
    "Noisy, $\mu = 200$": load_variable_data(noisy_path, "p_T"),
    "Denoised": load_variable_data(reconstructed_path, "p_T")
}

plot_resolutions(
    mass_resolutions_orig, pt_resolutions_orig,
    colors={
        "Best case": "black",
        "Noisy, $\mu = 200$": "red",
        "Denoised": "blue"
    },
    use_log = True,
    save_path = f"{CWD}/data/plots/relative_resolutions/resolution_grid{16}_Unet{UNET_DIMS}_mass_gtORIG.pdf"
)

# Second plot - comparison against best case
mass_resolutions_best = {
    "Noisy, $\mu = 200$": load_variable_data(noisy_path, "mass", truth_path=best_case_path),
    "Denoised": load_variable_data(reconstructed_path, "mass", truth_path=best_case_path)
}

pt_resolutions_best = {
    "Noisy, $\mu = 200$": load_variable_data(noisy_path, "p_T", truth_path=best_case_path),
    "Denoised": load_variable_data(reconstructed_path, "p_T", truth_path=best_case_path)
}

plot_resolutions(
    mass_resolutions_best, pt_resolutions_best,
    colors={
        "Noisy, $\mu = 200$": "red",
        "Denoised": "blue"
    },
    save_path = f"{CWD}/data/plots/relative_resolutions/resolution_grid{16}_Unet{UNET_DIMS}_mass_gtBEST.pdf"
)

#################################################################################

# Resplots for impact of gridding, compare against cts pure
bins = [4, 8, 16, 256]
save_path = f"{CWD}/data/plots/relative_resolutions/resolution_compare_grids_{'_'.join(map(str, bins))}.pdf"

paths = [f"{CWD}/data/3-grid/mu0/noisy_mu0_event_level_from_grid{bin}.csv" for bin in bins]
mass_resolutions_grids = { rf"$b={bin}$" : load_variable_data(paths[i], "mass")
                          for i, bin in enumerate(bins) }
pt_resolutions_grids = { rf"$b={bin}$" : load_variable_data(paths[i], "p_T")
                          for i, bin in enumerate(bins) }

plot_resolutions(
    mass_resolutions_grids, pt_resolutions_grids,
    save_path = save_path,
    mass_cutoff=(-1, 4),
    pt_cutoff=(-0.1, 0.1),
    use_log=True,
    legend_title="",
    fig_vinch=4.5,
)

#################################################################################

# output_path = f"{CWD}/data/plots/bmap_comparison/"

# plot_mu_comparison(tt, pile_up, 
#                   use_log=False,
#                   save_path=f"{output_path}/mu_comparison_b{BMAP_SQUARE_SIDE_LENGTH}_linear.png")

# plot_mu_comparison(tt, pile_up, 
#                   use_log=True,
#                   save_path=f"{output_path}/mu_comparison_b{BMAP_SQUARE_SIDE_LENGTH}_log.png")

#################################################################################

# files = [
#     f"{CWD}/data/2-intermediate/noisy_mu0_event_level.csv",
#     f"{CWD}/data/3-grid/mu0/noisy_mu{0}_event_level_from_grid{64}.csv",
#     f"{CWD}/data/2-intermediate/noisy_mu200_event_level.csv",
#     f"{CWD}/data/4-reconstruction/beta001/reconstructed_mu200_event_level_from_grid64_Unet64.csv",
# ]
# labels = ["Original", "Best case", "Noisy", "Denoised"]
# save_path = f"{CWD}/data/plots/1D_histograms/overlaid_from_model/overlaid_comparison_b64_beta001_unet64.png"
# create_overlay_plots_general(files, labels, mass_max=350, save_path=save_path)

#################################################################################

# plot_event_level_quantities_comparison(tt, pile_up, f"{CWD}/data/plots/event_level_quantities_comparison.png")
# plot_particle_level_quantities_comparison(tt, pile_up, f"{CWD}/data/plots/particle_level_quantities_comparison.png")
# plot_particle_level_quantities_ttbar_only(f"{CWD}/data/plots/particle_level_ttbar.png")

#################################################################################

# Plot model output resplots
best_case_path = f"{CWD}/data/3-grid/mu0/noisy_mu0_event_level_from_grid{16}.csv"

binsbeta = [
    (32, "001", False, "(a)", "(b)"),
    (32, "0.5", False, "(c)", "(d)"),
    (64, "001", False, "(e)", "(f)"),
    (64, "0.5", True, "(g)", "(h)"),
]

for x in binsbeta:
    bins, beta, show_subtit = x[0], x[1], x[2]
    mass_text, pT_text = x[3], x[4]
    
    reconstructed_path = f"{CWD}/data/4-reconstruction/beta{beta}/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"
    
    # Second plot - comparison against best case
    noisy_binned = f"{CWD}/data/2-intermediate/noisy_mu200_event_level_grid{bins}.csv"
    mass_resolutions_best = {
        "Noisy ($\mu=200$)": load_variable_data(noisy_binned, "mass", truth_path=best_case_path),
        "Denoised": load_variable_data(reconstructed_path, "mass", truth_path=best_case_path)
    }

    pt_resolutions_best = {
        "Noisy ($\mu=200$)": load_variable_data(noisy_binned, "p_T", truth_path=best_case_path),
        "Denoised": load_variable_data(reconstructed_path, "p_T", truth_path=best_case_path)
    }

    rt=f"{CWD}/data/plots/relative_resolutions/rs_ver/"
    os.makedirs(rt, exist_ok=True)
    plot_resolutions(
        mass_resolutions_best, pt_resolutions_best,
        colors={
            "Noisy ($\mu=200$)": "red",
            "Denoised": "blue"
        },
        save_path = f"{rt}/resolution_grid{bins}_Unet{UNET_DIMS}_mass_beta{beta}.pdf",
        legend_title=f"Parameters\n$b={bins}$, $\\beta={beta}$",
        show_subtit=show_subtit,
        mass_text=mass_text,
        pT_text=pT_text,
    )

#################################################################################
