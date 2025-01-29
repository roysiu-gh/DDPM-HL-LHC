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

# === Read in data
# print("0 :: Loading original data")
# pile_up = np.genfromtxt(
#     PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# tt = np.genfromtxt(
#     TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# print("FINISHED loading data\n")

# #################################################################################
# tt = EventSelector(tt)
# pile_up = EventSelector(pile_up)

mus = np.arange(0,11,step=1)
# high_PU_no = int(pile_up.max_ID)
# max_jet_no = int(tt.max_ID)
# print("high pu", high_PU_no)
# print("max jet", max_jet_no)
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
y_qlabel = {
    "mass": r'$\braket{m_{\mu}^{\text{jet}} - m_{0}^{\text{jet}}}$ [GeV]',
    "energy": r"$\braket{E_{\mu}^{\text{jet}} - E_{0}^{\text{jet}}}$ [GeV]",
    "pt": r"$\braket{p_{T,\mu}^{\text{jet}} - p_{T,0}^{\text{jet}}}$ [GeV]"
}
energy_data = np.loadtxt(f"{CWD}/data/plots/energy_resolution_data.txt", delimiter=",")
energy_mean = energy_data[0]
energy_std = energy_data[1]
# print(energy_data)
fig,axs = plt.subplots(nrows=1,ncols=2, figsize=(8,6))
ax1, ax2 = axs
plt.tight_layout()
ax1.plot(mus, energy_mean)
ax1.set_xlabel("$\mu$")
ax1.set_ylabel(r"$\braket{E_{\mu}^{\text{jet}} - E_{0}^{\text{jet}}}$ [GeV]")
ax2.plot(mus, energy_std)
ax2.set_xlabel("$\mu$")
ax2.set_ylabel(r"$\sigma(E)$ [GeV]")
ax1.set_xlim(0, np.max(mus))
ax2.set_xlim(0, np.max(mus))
ax1.set_ylim(0 if 0 < np.min(energy_mean) else np.min(energy_mean), np.max(energy_mean))
plt.savefig(f"{CWD}/data/plots/Mean_Energy_graphs.pdf", format="pdf")
plt.close()
# end_time_global = time.time()
# print(f"Global runtime: {end_time_global - start_time_global} seconds")

print("DONE ALL.")
