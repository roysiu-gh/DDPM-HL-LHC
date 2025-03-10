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
from DDPMLHC.generate_plots.bmap import *
from DDPMLHC.generate_plots.resolution_plots import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# #################################################################################
print("0 :: Loading original data")
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
pu = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = EventSelector(tt)
pu = EventSelector(pu)
print("FINISHED loading data\n")

# # Ground truth ttbar jets
# NG_jet = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=0)
# NG_jet.save_event_level_data()
# # Second one to randomly generate and return pile-up events ONLY
# # NG_pu = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=0, pu_only=True)
# # Default NG just for convenience
# NG_default = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=200)
# NG_default.save_event_level_data()


# def generate_event_level_gridded_jets(NG: NoisyGenerator,bins, mu=0, save_dir=INTERMEDIATE_PATH):
#     NG.reset()
#     NG.bins = bins
#     NG.mu = mu
#     gt_file = f"{save_dir}/noisy_mu{mu}_event_level_grid{bins}.csv"
#     combined = []
#     for idx, _ in enumerate(NG):
#         # next(NG)
#         grid = NG.get_grid(normalise=False)
#         # NG.select_jet(idx)
#         axis = NG.jet_axis
#         enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=NG.bins)
#         detas, dphis = decentre(axis, detas, dphis)
#         pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
#         # print("???")
#         event_quantities = particle_momenta_to_event_level(enes, pxs, pys, pzs)
#         event_mass, event_px, event_py, event_pz, event_eta, event_phi, event_pT = event_quantities
        
#         event_level = np.array([
#             idx,
#             event_px,
#             event_py,
#             event_pz,
#             event_eta,
#             event_phi,
#             event_mass,
#             event_pT,
#         ])
        
#         combined.append(np.copy(event_level))
            
#     all_data = np.vstack(combined)
#     np.savetxt(
#             gt_file,
#             all_data,
#             delimiter=",",
#             header="event_id,px,py,pz,eta,phi,mass,p_T",
#             comments="",
#             fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
#     )
#     return all_data, gt_file

# # for x in product([64], [200]):
# #     generate_event_level_gridded_jets(NG=NG_jet, bins=x[0], mu=x[1])
# # # Comparison against pure cts
# noisy_path = f"{CWD}/data/2-intermediate/noisy_mu200_event_level.csv"

# for x in product([16,32,64], ["1", "0.5", "001"]):
#     bins = x[0]
#     beta = x[1]
#     best_case_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level_grid{bins}.csv"
    
#     # beta = "0.5"
#     os.makedirs(f"{CWD}/data/plots/relative_resolutions/beta{beta}", exist_ok=True)
#     reconstructed_path_beta05 = f"{CWD}/data/4-reconstruction/beta{beta}/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"
#     # reconstructed_path_beta1 = f"{CWD}/data/4-reconstruction/beta1/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"
#     reconstructed_path = reconstructed_path_beta05
#     # First plot - comparison against pure cts
#     mass_resolutions_orig = {
#         "Ground truth": load_variable_data(best_case_path, "mass"),
#         "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "mass", ),
#         "Denoised": load_variable_data(reconstructed_path, "mass")
#     }

#     pt_resolutions_orig = {
#         "Ground truth": load_variable_data(best_case_path, "p_T"),
#         "Noisy ($\mu = 200$)": load_variable_data(noisy_path, "p_T"),
#         "Denoised": load_variable_data(reconstructed_path, "p_T")
#     }

#     plot_resolutions(
#         mass_resolutions_orig, pt_resolutions_orig,
#         colors={
#             "Ground truth": "black",
#             "Noisy ($\mu = 200$)": "red",
#             "Denoised": "blue"
#         },
#         use_log = True,
#         save_path = f"{CWD}/data/plots/relative_resolutions/beta{beta}/resolution_grid{bins}_Unet{UNET_DIMS}_mass_gtORIG.pdf",
#         legend_title=rf"${bins}\times {bins}$ grid"
#     )

#     # Second plot - comparison against best case
#     noisy_binned = f"{CWD}/data/2-intermediate/noisy_mu200_event_level_grid{bins}.csv"
#     mass_resolutions_best = {
#         "Noisy ($\mu = 200$)": load_variable_data(noisy_binned, "mass", truth_path=best_case_path),
#         "Denoised": load_variable_data(reconstructed_path, "mass", truth_path=best_case_path)
#     }

#     pt_resolutions_best = {
#         "Noisy ($\mu = 200$)": load_variable_data(noisy_binned, "p_T", truth_path=best_case_path),
#         "Denoised": load_variable_data(reconstructed_path, "p_T", truth_path=best_case_path)
#     }

#     plot_resolutions(
#         mass_resolutions_best, pt_resolutions_best,
#         colors={
#             "Noisy ($\mu = 200$)": "red",
#             "Denoised": "blue"
#         },
#         save_path = f"{CWD}/data/plots/relative_resolutions/beta{beta}/resolution_grid{bins}_Unet{UNET_DIMS}_mass_gtBEST.pdf",
#         legend_title=rf"${bins}\times {bins}$ grid"
#     )



output_path = f"{CWD}/data/plots/bmap_comparison/"
############### PURE JET NOISY JET AND DENOISED SIDE-BY-SIDE
def compare_denoised(denoised_array, use_log=True):
    save_path=f"{output_path}/denoised_comparison_linear.png" if use_log == False else f"{output_path}/denoised_comparison_log1p.png"
    fig = plt.figure(figsize=(16, 6))  # Increased height slightly for labels
    
    # Create gridspec to have better control over spacing
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    main_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Get mu values
    # mu_values = [0, 100, 200]
    vmin, vmax = float('inf'), -float('inf')

    # Store random state
    rng_state = np.random.get_state()
    
    # First pass to get global scaling across all jets and mu values
    all_grids = []
    # Get noisy jets
    mu = 200
    # for mu in mu_values:
        # Reset random state to get same jets
    np.random.set_state(rng_state)
    
    mu_grids = []
    NG = NoisyGenerator(tt, pu, mu=mu)
    NG.reset()
    NG.mu = 0
    for _ in range(4):
        next(NG)
        grid = NG.get_grid(normalise=False)
        if use_log:
            grid = np.log1p(grid)
        vmin = min(vmin, grid.min())
        vmax = max(vmax, grid.max())
        mu_grids.append(grid)
    all_grids.append(mu_grids)
    # gets mu = 200 grids
    NG.reset()
    NG.mu = mu
    for _ in range(4):
        next(NG)
        grid = NG.get_grid(normalise=False)
        if use_log:
            grid = np.log1p(grid)
        vmin = min(vmin, grid.min())
        vmax = max(vmax, grid.max())
        mu_grids.append(grid)
    all_grids.append(mu_grids)
    # denoised_array comes from sampling of first 4 jets from model
    #  Assume it has been converted to numpy array already
    # Plot each mu subplot with its 4 jets
    all_grids.append(denoised_array)
    letters = ['(a)', '(b)', '(c)']
    for idx, grids in enumerate(all_grids):
        # Create 2x2 grid for this mu
        grid_size = grids[0].shape[0]
        combined_grid = np.zeros((grid_size * 2, grid_size * 2))
        
        # Fill the 2x2 grid
        combined_grid[:grid_size, :grid_size] = grids[0]
        combined_grid[:grid_size, grid_size:] = grids[1]
        combined_grid[grid_size:, :grid_size] = grids[2]
        combined_grid[grid_size:, grid_size:] = grids[3]

        im = main_axes[idx].imshow(combined_grid, 
                                    cmap='viridis',
                                    vmin=vmin,
                                    vmax=vmax,
                                    norm=None,  # Add this to prevent automatic normalization
                                    interpolation='nearest')
        
        # Add grid lines to separate events
        main_axes[idx].axhline(y=grid_size-0.5, color='white', linewidth=1)
        main_axes[idx].axvline(x=grid_size-0.5, color='white', linewidth=1)
        
        # # Move mu label to bottom
        # main_axes[idx].text(0.5, -0.1, f'{letter} $\mu = {mu}$',
        #                   transform=main_axes[idx].transAxes,
        #                   fontsize=18, ha='center')
        
        main_axes[idx].axis('off')
    
    print(f"Maximum energy in any pixel: {max([grid.max() for grids in all_grids for grid in grids]):.4f}")

    # Add colorbar with proper spacing
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('$\ln(1+E)$' if use_log else 'Energy', fontsize=18)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, 
                   bbox_inches='tight', 
                   dpi=300,
                   pad_inches=0.2)
        print(f"Saved figure to {save_path}")
        plt.close()
    # else:
    #     plt.show()
# plot_mu_comparison(tt, pu, 
#                   use_log=False,
#                   save_path=f"{output_path}/mu_comparison_linear.png")

# plot_mu_comparison(tt, pu, 
#                   use_log=True,
#                   save_path=f"{output_path}/mu_comparison_log.png")
