# Package imports
import numpy as np
import matplotlib as mpl
import sys
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.generate_plots.generate_1d_plots import plot_1d_histograms
from DDPMLHC.generate_plots.resolution_plots import *
import multiprocessing
mpl.rcParams.update(MPL_GLOBAL_PARAMS)
MAX_DATA_ROWS = None
import polars as pl

# === Read in data
<<<<<<< HEAD
# print("0 :: Loading original data")
# pile_up = np.genfromtxt(
#     PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
=======
print("0 :: Loading original data")
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
print("FINISHED loading data\n")

#################################################################################

# # === 2D Histograms ===
# BINS = (16,16)
# # === EXAMPLE USAGE OF GENERATING IMAGES ===
max_pileup_id = np.max(pile_up[:,0])
# # max_event_ids = np.linspace(0, max_pileup_id, num=max_pileup_id+1)
jet_no = 493
BINS = [32]
# for BIN in BINS:
#     generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=200, max_event_id=max_pileup_id, energies=None)
#     # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=10000)
#     # generate_2dhist(tt, pile_up_data=pile_up, jet_no=jet_no, bins=BIN, mu=50, hist_plot="count")
# #  Use new visualisaton to just distinguish pile up and jet
# # ====== END 2D HIST ====

#################################################################################

# === Create noisy events
print("1 :: Creating noisy events")
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 0, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 1, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 3, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 5, save_path=INTERMEDIATE_PATH)
make_noisy_data(range(TTBAR_NUM), tt, pile_up, 10, save_path=INTERMEDIATE_PATH)
print("FINISHED creating noisy events\n")

# === Collapse noisy data to event-level
print("2 :: Collapsing noisy data to event-level")
calculate_event_level_quantities(0, INTERMEDIATE_PATH)
calculate_event_level_quantities(1, INTERMEDIATE_PATH)
calculate_event_level_quantities(3, INTERMEDIATE_PATH)
calculate_event_level_quantities(5, INTERMEDIATE_PATH)
calculate_event_level_quantities(10, INTERMEDIATE_PATH)
print("FINISHED collapsing noisy data to event-level\n")

# === Draw 1D histograms
print("3 :: Drawing 1D histograms")
plot_1d_histograms(mu=0)
plot_1d_histograms(mu=1)
plot_1d_histograms(mu=3)
plot_1d_histograms(mu=5)
plot_1d_histograms(mu=10)
print("FINISHED drawing 1D histograms\n")

# print("4 :: Drawing overlaid histograms with varying mu")
# print("Loading intermediate data...")
# tt = np.genfromtxt(
#     TT_EXT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
>>>>>>> fcdb204 (Delete bmap.py (functionality moved inside NoisyGenerator). Delete main.py.)
# )
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
mean_energy_diffs = []
std_energy_diffs = []
mean_mass_diffs = []
std_mass_diffs = []
for csv_file_path in csv_file_paths:
    df = pl.read_csv(csv_file_path)
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
    mean_energy_diff = np.sum(energy_diffs) / (max_id - 1)
    std_energy_diff = np.std(energy_diffs)
    # print(std_energy_diff)
    # print(mean_energy_diff)

    mean_mass_diff = np.sum(mass_diffs2) / (max_id - 1)
    std_mass_diff = np.std(mass_diffs2)

    mean_energy_diffs.append(mean_energy_diff)
    std_energy_diffs.append(std_energy_diff)
    mean_mass_diffs.append(mean_mass_diff)
    std_mass_diffs.append(std_mass_diff)


y_qlabel = {
    "mass": r'$\braket{m_{\mu}^{\text{jet}} - m_{0}^{\text{jet}}}$ [GeV]',
    "energy": r"$\braket{E_{\mu}^{\text{jet}} - E_{0}^{\text{jet}}}$ [GeV]",
    # "pt": r"$\braket{p_{T,\mu}^{\text{jet}} - p_{T,0}^{\text{jet}}}$ [GeV]"
}
# energy_data = np.loadtxt(f"{CWD}/data/plots/energy_resolution_data.txt", delimiter=",")
# energy_mean = energy_data[0]
# energy_std = energy_data[1]
# print(energy_data)
fig,axs = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
ax11, ax12, ax21, ax22 = axs.flatten()
# plt.tight_layout()
# energy graphs
ax11.plot(mus, mean_energy_diffs)
ax11.set_xlabel("$\mu$")
ax11.set_ylabel(r"$\braket{E_{\mu}^{\text{jet}} - E_{0}^{\text{jet}}} / E_{0}^{\text{jet}}$")
ax12.plot(mus, std_energy_diffs)
ax12.set_xlabel("$\mu$")
ax12.set_ylabel(r"$\sigma(E)$ [GeV]")
ax11.set_xlim(0, np.max(mus))
ax12.set_xlim(0, np.max(mus))
ax11.set_ylim(0 if 0 < np.min(mean_energy_diffs) else np.min(mean_energy_diffs), np.max(mean_energy_diffs))

# mass graphs
ax21.plot(mus, mean_mass_diffs)
ax21.set_xlabel("$\mu$")
ax21.set_ylabel(r"$\braket{m_{\mu}^{\text{jet}} - m_{0}^{\text{jet}}} / m_{0}^{\text{jet}}$")
ax22.plot(mus, std_mass_diffs)
ax22.set_xlabel("$\mu$")
ax22.set_ylabel(r"$\sigma(m)$ [GeV]")
ax21.set_xlim(0, np.max(mus))
ax22.set_xlim(0, np.max(mus))
ax21.set_ylim(0 if 0 < np.min(mean_mass_diffs) else np.min(mean_mass_diffs), np.max(mean_mass_diffs))
plt.savefig(f"{CWD}/data/plots/Mean_EnergyMassDiff_graphs.pdf", format="pdf")
plt.close()

# #################################################################################
# tt = EventSelector(tt)
# pile_up = EventSelector(pile_up)

# mus = np.arange(0,11,step=1)
# high_PU_no = int(pile_up.max_ID)
# max_jet_no = int(tt.max_ID)
# print("high pu", high_PU_no)
# data_y = np.zeros((3, len(mus)))

# tasks = [
#         (tt, pile_up, mu, max_jet_no, high_PU_no+1) 
#         for mu in mus
#     ]
# energy_data = np.zeros(len(tasks))
# energy_std = np.zeros(len(tasks))
# for indx,task in enumerate(tasks):
#     data = mean_quantity_diff(task)
#     energy_data[indx] += data[0]
#     energy_std[indx] += data[1]
# np.savetxt(f"{CWD}/data/plots/energy_resolution_data.txt", [energy_data, energy_std], fmt="%10.10f",delimiter=",")
# with multiprocessing.Pool(processes=int(len(mus) / 4)) as pool:
#     results = pool.map(mean_quantity_diff, tasks)
# pool.close()
# pool.join()

# for ind,E_total in enumerate(results):
    # data_y[0][ind] += m_total
    
    # data_y[2][ind] += p_T_total

# for name,data in zip(["Mass", "Energy", "pT"], data_y):
#     fig  = plt.figure(figsize=(8,6))
#     plt.tight_layout()
#     plt.plot(MUs, data)
#     plt.xlabel("$\mu$")
#     plt.ylabel(y_qlabel[name.lower()])
#     plt.xlim(0, np.max(MUs))
#     plt.ylim(0 if 0 < np.min(data) else np.min(data), np.max(data))
#     plt.savefig(f"{CWD}/data/plots/Mean_{name}_diff.pdf", format="pdf")
#     plt.close()

# end_time_global = time.time()
# print(f"Global runtime: {end_time_global - start_time_global} seconds")

print("DONE ALL.")
