# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
import multiprocessing
mpl.rcParams.update(MPL_GLOBAL_PARAMS)
MAX_DATA_ROWS = None
import polars as pl

# tt = np.genfromtxt(
#     TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# print("FINISHED loading data\n")
mus = [0, 1, 3, 5, 10, 15, 30, 50, 75, 100, 125, 150, 175, 200]
# mus = [0, 1, 3, 5, 10, 15, 30, 50]
# tt = EventSelector(tt)
# pile_up = EventSelector(pile_up)
# for mu in mus:
#     cur_generator = NoisyGenerator(tt, pile_up, mu=mu)
#     cur_generator.save_event_level_data()
    # plot_1d_histograms(mu=mu)
csv_file_paths = [f"{INTERMEDIATE_PATH}/noisy_mu{mu}_event_level.csv" for mu in mus]

# Get jet data to compute differences
jet_quantities = pl.read_csv(f"{INTERMEDIATE_PATH}/noisy_mu0_event_level.csv")
jets_px = jet_quantities['px']
jets_py = jet_quantities['py']
jets_pz = jet_quantities['pz']
jets_mass = jet_quantities['mass'].to_numpy()
# print(your[24717:])
jets_mass = np.concatenate((jets_mass[0:24716], jets_mass[24717:]))


# print("jet_mass zeros", jets_mass[jets_mass ==0])
# print("jet mass 24716", jets_mass[24716])
# print("jet_mass zeros loc", np.where(jets_mass == 0))
# # print(jets_mass)
jet_energy = (jets_px ** 2) + (jets_py ** 2) + (jets_pz ** 2)
jet_energy = jet_energy.to_numpy()
jet_energy = np.concatenate((jet_energy[0:24716], jet_energy[24717:]))

jet_energy = np.sqrt(jet_energy)
print(len(jet_energy))
# print("max jet", max_jet_no)
# mean_energy_diffs = []
# std_energy_diffs = []
# mean_mass_diffs = []
# std_mass_diffs = []
# for csv_file_path in csv_file_paths:
#     df = pl.read_csv(csv_file_path)
#     max_id = df['event_id'].max()
#     px = df['px']
#     py = df['py']
#     pz = df['pz']
#     # print(px)
#     mass = df['mass']
#     mass = mass.to_numpy()
#     mass = np.concatenate((mass[0:24716], mass[24717:]))
#     energy = (px ** 2) + (py ** 2) + (pz ** 2)
#     energy = energy.to_numpy()
#     energy = np.concatenate((energy[0:24716], energy[24717:]))

#     energy = np.sqrt(energy)
#     # print(energy)
#     # Find energy difference between jet+pile-up and jet for feach jet_id
#     energy_diffs = energy - jet_energy
#     energy_diffs = energy_diffs / jet_energy
#     mass_diffs = mass - jets_mass
#     mass_diffs2 = mass_diffs/ jets_mass
#     # print(mass_diffs2)
#     mean_energy_diff = np.sum(energy_diffs) / (max_id - 1)
#     std_energy_diff = np.std(energy_diffs)
#     # print(std_energy_diff)
#     # print(mean_energy_diff)

#     mean_mass_diff = np.sum(mass_diffs2) / (max_id - 1)
#     std_mass_diff = np.std(mass_diffs2)

#     mean_energy_diffs.append(mean_energy_diff)
#     std_energy_diffs.append(std_energy_diff)
#     mean_mass_diffs.append(mean_mass_diff)
#     std_mass_diffs.append(std_mass_diff)


# y_qlabel = {
#     "mass": r'$\braket{m_{\mu}^{\text{jet}} - m_{0}^{\text{jet}}}$ [GeV]',
#     "energy": r"$\braket{E_{\mu}^{\text{jet}} - E_{0}^{\text{jet}}}$ [GeV]",
#     # "pt": r"$\braket{p_{T,\mu}^{\text{jet}} - p_{T,0}^{\text{jet}}}$ [GeV]"
# }
# # energy_data = np.loadtxt(f"{CWD}/data/plots/energy_resolution_data.txt", delimiter=",")
# # energy_mean = energy_data[0]
# # energy_std = energy_data[1]
# # print(energy_data)
# fig,axs = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
# ax11, ax12, ax21, ax22 = axs.flatten()
# # plt.tight_layout()
# # energy graphs
# ax11.plot(mus, mean_energy_diffs)
# ax11.set_xlabel("$\mu$")
# ax11.set_ylabel(r"$\braket{E_{\mu}^{\text{jet}} - E_{0}^{\text{jet}}} / E_{0}^{\text{jet}}$")
# ax12.plot(mus, std_energy_diffs)
# ax12.set_xlabel("$\mu$")
# ax12.set_ylabel(r"$\sigma(E)$ [GeV]")
# ax11.set_xlim(0, np.max(mus))
# ax12.set_xlim(0, np.max(mus))
# ax11.set_ylim(0 if 0 < np.min(mean_energy_diffs) else np.min(mean_energy_diffs), np.max(mean_energy_diffs))

# # mass graphs
# ax21.plot(mus, mean_mass_diffs)
# ax21.set_xlabel("$\mu$")
# ax21.set_ylabel(r"$\braket{m_{\mu}^{\text{jet}} - m_{0}^{\text{jet}}} / m_{0}^{\text{jet}}$")
# ax22.plot(mus, std_mass_diffs)
# ax22.set_xlabel("$\mu$")
# ax22.set_ylabel(r"$\sigma(m)$ [GeV]")
# ax21.set_xlim(0, np.max(mus))
# ax22.set_xlim(0, np.max(mus))
# ax21.set_ylim(0 if 0 < np.min(mean_mass_diffs) else np.min(mean_mass_diffs), np.max(mean_mass_diffs))
# plt.savefig(f"{CWD}/data/plots/Mean_EnergyMassDiff_graphs.pdf", format="pdf")
# plt.close()

# Plot the count vs energy response for fixed mu
print("Doing count vs energy response for fixed mu")

mus = [1, 50, 200]
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
    axs[idx][0].hist(energy_diffs, bins = 50, label=f"$\mu = {mus[idx]}$")
    axs[idx][1].hist(mass_diffs2, bins = 50,label=f"$\mu = {mus[idx]}$")
    # std_energy_diff = np.std(energy_diffs)
    # print(std_energy_diff)
    # print(mean_energy_diff)

    # mean_mass_diff = np.sum(mass_diffs2) / (max_id - 1)
    # std_mass_diff = np.std(mass_diffs2)

    # mean_energy_diffs.append(mean_energy_diff)
    # std_energy_diffs.append(std_energy_diff)
    # mean_mass_diffs.append(mean_mass_diff)
    # std_mass_diffs.append(std_mass_diff)

plt.savefig(f"{CWD}/data/plots/hist_energymasscounts.pdf", format="pdf")
plt.close()

print("DONE ALL.")
