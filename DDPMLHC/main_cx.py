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
from DDPMLHC.generate_plots.bmap import save_to_bmap

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# === Read in data
print("0 :: Loading original data")
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = EventSelector(tt)
pile_up = EventSelector(pile_up)
print("FINISHED loading data\n")


# mu = 0
# output_path = f"{CWD}/data/3-grid"
# BMAP_SQUARE_SIDE_LENGTH = 16
# output_filename = f"noisy_mu{mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"
# output_filepath = f"{output_path}/{output_filename}"

# generator = NoisyGenerator(tt, pile_up, mu=0, bins=BMAP_SQUARE_SIDE_LENGTH)
# combined = []

# for idx, _ in enumerate(generator):
#     grid = generator.get_grid(normalise=False)
    
#     enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=generator.bins)
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

# # Final check before saving
# if np.any(np.isnan(all_data)):
#     print("\nWarning: NaN values in final data:")
#     print(f"Total NaN count: {np.sum(np.isnan(all_data))}")
#     print("NaN locations (row, column):")
#     nan_rows, nan_cols = np.where(np.isnan(all_data))
#     column_names = ['event_id', 'px', 'py', 'pz', 'eta', 'phi', 'mass', 'p_T']
#     for row, col in zip(nan_rows, nan_cols):
#         print(f"Row {row}, Column {column_names[col]}")

# np.savetxt(
#     output_filepath,
#     all_data,
#     delimiter=",",
#     header="event_id,px,py,pz,eta,phi,mass,p_T",
#     comments="",
#     fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
# )

# plot_1d_histograms(mu, event_stats_path=output_filepath, output_path=f"{output_path}/grid{BMAP_SQUARE_SIDE_LENGTH}")


# print("DONE ALL.")


# ENERGY COUNT RESOLUTION PLOTS FOR FIXED MU
# 

mus = [200]
csv_file_paths = [f"{INTERMEDIATE_PATH}/noisy_mu{mu}_event_level.csv" for mu in mus]
fig,axs = plt.subplots(nrows=len(csv_file_paths),ncols=2, figsize=(10,15))
#  = axs
energy_counts = []
mass_counts = []
mean_mass_diffs = []
std_mass_diffs = []
data_array = [pl.read_csv(csv_file_path) for csv_file_path in csv_file_paths]
for idx,data in enumerate(data_array):
    df = data
    max_id = df['event_id'].max()
    px = df['px']
    py = df['py']
    pz = df['pz']
    # print(px)
    mass = df['mass']
    mass = mass.to_numpy()
    mass = np.concatenate((mass[0:24716], mass[24717:]))
    energy = (px ** 2) + (py ** 2) + (pz ** 2)
    energy = energy.to_numpy()
    energy = np.concatenate((energy[0:24716], energy[24717:]))

    energy = np.sqrt(energy)
    # print(energy)
    # Find energy difference between jet+pile-up and jet for feach jet_id
    energy_diffs = energy - jet_energy
    energy_diffs = energy_diffs / jet_energy
    mass_diffs = mass - jets_mass
    mass_diffs2 = mass_diffs/ jets_mass
    # print(mass_diffs2)
    # mean_energy_diff = np.sum(energy_diffs) / (max_id - 1)
    # en_bins = np.mgrid[np.min(energy_diffs):np.max(energy_diffs):(len(energy_diffs)+1)*1j]
    # mass_bins = np.mgrid[np.min(mass_diffs2):np.max(mass_diffs2):(len(mass_diffs2)+1)*1j]
    # mass_bins = np.mgrid[0:mass_max:(mass_num_bins+1)*1j]
    axs[idx][0].hist(energy_diffs, bins = 50, label=f"$\\mu = {mus[idx]}$", edgecolor="black")
    axs[idx][1].hist(mass_diffs2[mass_diffs2<5], bins = 50,label=f"$\\mu = {mus[idx]}$", edgecolor="black")
    axs[idx][0].set_ylabel(r"Counts")
    # axs[idx][0].set_ylabel(r"Counts")

    axs[idx][0].legend(prop={'size': 14})
    axs[idx][1].legend(prop={'size': 14})
    # std_energy_diff = np.std(energy_diffs)
    # print(std_energy_diff)
    # print(mean_energy_diff)

    # mean_mass_diff = np.sum(mass_diffs2) / (max_id - 1)
    # std_mass_diff = np.std(mass_diffs2)

    # mean_energy_diffs.append(mean_energy_diff)
    # std_energy_diffs.append(std_energy_diff)
    # mean_mass_diffs.append(mean_mass_diff)
    # std_mass_diffs.append(std_mass_diff)
axs[-1][0].set_xlabel(r"$\frac{E_{\mu}^{j} - E_{0}^{j}}{E_{0}^{j}}$")
axs[-1][1].set_xlabel(r"$\frac{m_{\mu}^{j} - m_{0}^{j}}{m_{0}^{j}}$")
plt.savefig(f"{CWD}/data/plots/hist_energymasscounts.pdf", format="pdf")
plt.savefig(f"{CWD}/data/plots/hist_energymasscounts.png", format="png", dpi=600)
plt.close()
