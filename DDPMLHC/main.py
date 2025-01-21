# Package imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.dataset_ops.process_data import *
from DDPMLHC.dataset_ops.generate_intermediate_event_data import process_noisy_data
from DDPMLHC.generate_plots.generate_1d_plots import plot_1d_histograms
from DDPMLHC.generate_plots.visualisation import plot_detections, generate_2dhist
from DDPMLHC.dataset_ops.data_loading import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

MAX_DATA_ROWS = 100000

# ======= global matplotlib params =====
# plt.rcParams["text.usetex"] = False  # Use LaTeX for rendering text
# plt.rcParams["font.size"] = 12  # Set default font size (optional)

# === Read in data
print("0 :: Loading original data")
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

#################################################################################
# jet_no = 493
# plot_data = select_event(tt, jet_no)
# jet_centre = get_axis_eta_phi(plot_data[:,3:])
# mu = 100
# # # event_IDS = np.random.choice(pile_up[:,0], size = mu).astype(int)
# event_IDS = np.random.randint(0, np.max(pile_up[:,0]), size = mu).astype(int)
# selected_pile_ups = [select_event(pile_up,event_ID, filter=True) for event_ID in event_IDS]
# selected_pile_ups = np.vstack(selected_pile_ups)
# zero_p_mask = ~((selected_pile_ups[:, 3] == 0) & (selected_pile_ups[:, 4] == 0) & (selected_pile_ups[:, 5] == 0))
# number_of_false = np.size(zero_p_mask) - np.count_nonzero(zero_p_mask)
# selected_pile_ups = selected_pile_ups[zero_p_mask]
# print("Number of false pile-ups", number_of_false)
# plot_detections(
#     tt_bar=plot_data,
#     centre = jet_centre,
#     pile_ups=selected_pile_ups,
#     jet_no=jet_no,
#     filename=f"eta_phi_jet{jet_no}_Mu{mu}_select_missing_events",
#     base_radius_size=300,
#     momentum_display_proportion=1,
#     cwd=CWD,
#     pdgids=False
# )
# plot_detections(
#     tt_bar=delta_R(jet_centre, plot_data)[0],
#     centre = jet_centre,
#     pile_ups=selected_pile_ups,
#     jet_no=493,
#     filename=f"eta_phi_jet{jet_no}_deltaR_Mu = {mu}",
#     base_radius_size=100,
#     momentum_display_proportion=1,
#     cwd=CWD,
# )
# plot_detections(
#     plot_data=plot_data,
#     centre = jet_centre,
#     filename=f"eta_phi_jet{jet_no}_cropped",
#     base_radius_size=1,
#     momentum_display_proportion=0.9,
#     cwd=CWD,
# )

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

INTERMEDIATE_PATH = f"{CWD}/data/2-intermediate/try/"
OUT_PATH_1D_HIST = f"{CWD}/data/plots/1D_histograms/particles/"

# # === Create noisy events
# print("1 :: Creating noisy events")
# write_combined_csv(range(10000), tt, pile_up, 10, save_path=INTERMEDIATE_PATH)
# write_combined_csv(range(10000), tt, pile_up, 100, save_path=INTERMEDIATE_PATH)
# write_combined_csv(range(10000), tt, pile_up, 200, save_path=INTERMEDIATE_PATH)
# print("FINISHED creating noisy events\n")

# # === Make noisy events with extra data (e.g. transverse mmtm)
# print("2 :: Making noisy events with extra data")
# process_noisy_data(10, INTERMEDIATE_PATH)
# process_noisy_data(100, INTERMEDIATE_PATH)
# process_noisy_data(200, INTERMEDIATE_PATH)
# print("FINISHED making noisy events with extra data\n")

# === Draw 1D histograms
# print("3 :: Drawing 1D histograms")

# print("Loading intermediate data...")
# # Load intermediate data
# tt = np.genfromtxt(
#     TT_EXT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# px = tt[:, 1]
# py = tt[:, 2]
# pz = tt[:, 3]
# eta = tt[:, 4]
# p_T = tt[:, 6]
# p = p_magnitude(px, py, pz)
# hist_data_particles = [
#     {
#         "name": "Momentum $p$ [GeV]",
#         "data": p,
#         "plot_params": {"xlog": True},
#         "save_filename": "p",
#         "save_path": OUT_PATH_1D_HIST,
#     },
#     {
#         "name": "Pseudorapidity $\eta$",
#         "data": eta,
#         "plot_params": {},
#         "save_filename": "eta",
#         "save_path": OUT_PATH_1D_HIST,
#     },
#     {
#         "name": "Transverse Momentum $p_T$ [GeV]",
#         "data": p_T,
#         "plot_params": {"xlog": True},
#         "save_filename": "pT",
#         "save_path": OUT_PATH_1D_HIST,
#     },
# ]

# foobah(mu=0, event_stats_path=JET_PATH)
# foobah(mu=10)
# foobah(mu=100)
# foobah(mu=200)
# foobah(mu=300)
# print("FINISHED drawing 1D histograms\n")

# print("DONE ALL.")


event_IDS_200 = np.random.randint(0, np.max(pile_up[:,0]), size = 200).astype(int)
event_IDS_100 = np.random.randint(0, np.max(pile_up[:,0]), size = 100).astype(int)# event_IDS_200 = np.random.choice(pile_up_data[:,0], size = 100).astype(int)
# max_event_id = np.max(pile_up_data[:,0])

# for jet_no in jet_nos:
# event_IDS = np.random.randint(low = 0, high = max_event_id, size = mu, dtype=np.int32)
# event_IDS = np.mgrid[0:(mu-1):(mu)*1j]
# print(f"Jet_No: {jet_no}, event IDs: {event_IDS}")
# selected_pile_ups now contain 2D arrays
selected_pile_ups_200 = [select_event_deprecated(pile_up, event_ID, filter=True) for event_ID in event_IDS_200]
selected_pile_ups_100 = [select_event_deprecated(pile_up, event_ID, filter=True) for event_ID in event_IDS_100]

# zero_p_mask_200 = ~((selected_pile_ups_200[:, 3] == 0) & (selected_pile_ups_200[:, 4] == 0) & (selected_pile_ups_200[:, 5] == 0))
# zero_p_mask_100 = ~((selected_pile_ups_100[:, 3] == 0) & (selected_pile_ups_100[:, 4] == 0) & (selected_pile_ups_100[:, 5] == 0))
selected_pile_ups_200 = np.vstack(selected_pile_ups_200)
selected_pile_ups_100 = np.vstack(selected_pile_ups_100)
selected_pile_ups_200 = selected_pile_ups_200[selected_pile_ups_200[:,0] != -1]
selected_pile_ups_100 = selected_pile_ups_100[selected_pile_ups_100[:,0] != -1]

# 

# if mu != 0:
#     selected_pile_ups = selected_pile_ups2[selected_pile_ups2[:,0] != -1]
# else:
#     selected_pile_ups = np.array([])
# Remove invalid pile_ups
# false_ids = np.size(selected_pile_ups2, axis=0) - np.size(selected_pile_ups, axis=0)
# print("Number of false Pile-up IDs: ",false_ids)
jet_data = select_event_deprecated(tt, jet_no, max_data_rows=MAX_DATA_ROWS)
data100 = np.vstack((jet_data,selected_pile_ups_100))
data200 = np.vstack((jet_data,selected_pile_ups_200))
# data = selected_pile_ups
# print(len(data))
# data = jet_data
pxj = jet_data[:,3]
pzj = jet_data[:,5]
pyj = jet_data[:,4]
px1 = data100[:,3]
py1 = data100[:,4]
pz1 = data100[:,5]
px2 = data200[:,3]
py2 = data200[:,4]
pz2 = data200[:,5]
pmag1 = p_magnitude(px1, py1, pz1)
pmag2 = p_magnitude(px2, py2, pz2)
pmagj = p_magnitude(pxj, pyj, pzj)

# All columns are passed in, so make sure to select last 3 columns for the 3-momenta
# Find jet axis
px, py, pz = jet_data[:,3], jet_data[:,4], jet_data[:,5]
jet_centre = get_axis_eta_phi(px, py, pz)
# Wrap jet axis phi between -1, 1
# print("centre", jet_centre)
phisj = to_phi(pxj, pyj) 
phis_100 = to_phi(px1, py1)
phis_200 = to_phi(px2, py2)
# Wrap particles w.r.t unwrapped jet axis
phis_100 = wrap_phi(jet_centre[1], phis_100)
phis_200 = wrap_phi(jet_centre[1], phis_200)
phis_j = wrap_phi(jet_centre[1], phisj)
# jet_centre_wrapped = wrap_phi(jet_centre[1], jet_centre[1])
# print("wrapped centre", jet_centre_wrapped)
etas_100 = pseudorapidity(pmag1, pz1)
etas_200 = pseudorapidity(pmag2, pz2)
etas_j = pseudorapidity(pmagj, pzj)
#shift particles so wrapped jet_axis is centred at (0,0)
centre,etas_c1, phis_c1 = centre_on_jet(jet_centre, etas_100, phis_100)
centre,etas_c2, phis_c2 = centre_on_jet(jet_centre, etas_200, phis_200)
centre,etas_cj, phis_cj = centre_on_jet(jet_centre, etas_j, phis_j)
# Delta R is calculated relative to the jet centre, and over all particles including pile-up
bounded_momenta1, etas_100, phis_100 = delta_R(centre, px1, py1, pz1, etas_c1, phis_c1)
bounded_momenta2, etas_200, phis_200 = delta_R(centre, px2, py2, pz2, etas_c2, phis_c2)
bounded_momentaj, etas_j, phis_j = delta_R(centre, pxj, pyj, pzj, etas_cj, phis_cj)
print(len(etas_j))
print(len(phis_j))
# bounded_momenta, etas2, phis2 = [px,py,pz], etas_c, phis_c
masked_px1 = bounded_momenta1[0]
masked_px2 = bounded_momenta2[0]
masked_py1 = bounded_momenta1[1]
masked_py2 = bounded_momenta2[1]
masked_pz1 = bounded_momenta1[2]
masked_pz2 = bounded_momenta2[2]
masked_pxj = bounded_momentaj[2]
masked_pyj = bounded_momentaj[2]
masked_pzj = bounded_momentaj[2]
# print(len(etas) == len(phis))
# masked_energies = np.sqrt(masked_data[:,3]*masked_data[:,3] + masked_data[:,4]*masked_data[:,4]+masked_data[:,5]*masked_data[:,5])
# # energy_normed = normalize_data(energies, energy_norm_factor)
# Unnormalised energies

# print("etas2", etas2)
# print("phis2", phis2)
# print("Orig num of eta: ", len(etas))
# print("Orig num of phi: ", len(phis))
# print("New num of eta: ", len(etas2))
# print("New num of phi: ", len(phis2))
# print("Masked number of particles, etas: ", len(etas) - len(etas2))
# print("Masked number of particles, etas: ", len(phis) - len(phis2))
# print(len(etas2) == len(phis2))
# print(np.min(etas2), np.max(etas2))
# print(np.min(etas), np.max(etas))
# exit(1)
# energy_normed = (masked_energies - energy_min) / energy_norm_denom
# print(energy_normed)
# Function appends "_hist" to the end
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
# fig, axs = plt.subplots(1,3,figsize=(20, 6), sharey=True)
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
cax = fig.add_subplot(gs[3])
# ax1, ax2, ax3 = axs
ax1.set_xlabel(r'$\Delta\eta$', fontsize=16)
ax2.set_xlabel(r'$\Delta\eta$', fontsize=16)
ax3.set_xlabel(r'$\Delta\eta$', fontsize=16)
ax1.set_ylabel(r'$\Delta\phi$', fontsize=16)
ax1.xaxis.set_major_formatter(FormatStrFormatter('$%.2f$'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('$%.2f$'))
ax3.xaxis.set_major_formatter(FormatStrFormatter('$%.2f$'))
# ax2.set_ylabel(r'$\Delta\phi$', fontsize=16)
# ax3.set_ylabel(r'$\Delta\phi$', fontsize=16)
# plt.title(
#     f"$\Delta\phi$ vs $\Delta\eta$ of jet {jet_no}, real_parts={len(etas2)}+false_pileup={false_ids}, bins={bins}"
# )
boundary = 1.0
bins = 32
created_bins = np.mgrid[-boundary:boundary:bins*1j]
save_str = f"./data/hist/combined_etaphi2d"
energies_100 = np.sqrt(masked_px1*masked_px1 + masked_py1*masked_py1+masked_pz1*masked_pz1)
energies_200 = np.sqrt(masked_px2*masked_px2 + masked_py2*masked_py2+masked_pz2*masked_pz2)
energies_jet = np.sqrt(masked_pxj*masked_pxj + masked_pyj*masked_pyj+masked_pzj*masked_pzj)
SD = np.std(energies_200)
scale = 256 / (0.001 * SD)  # Value above which we represent as full brightness (256)
print(scale)
print("std dev", SD)
scaled_energies_200 = np.floor(energies_200 * scale)
scaled_energies_100 = np.floor(energies_100 * scale)
scaled_energies_jet = np.floor(energies_jet * scale)
scaled_energies_200[scaled_energies_200 > 256] = 256  # Maximise at 256
scaled_energies_100[scaled_energies_100 > 256] = 256  # Maximise at 256
scaled_energies_jet[scaled_energies_jet > 256] = 256  # Maximise at 256
scaled_energies_jet = scaled_energies_jet.astype(int)
scaled_energies_100 = scaled_energies_100.astype(int)
scaled_energies_200 = scaled_energies_200.astype(int)
        # print(energies)
# print(energies2)
hist1 = ax1.hist2d(etas_j, phis_j, bins=(created_bins, created_bins), weights=np.log(scaled_energies_jet,where=scaled_energies_jet > 0), cmap='Greys_r',)
hist2 = ax2.hist2d(etas_100, phis_100, bins=(created_bins, created_bins), weights=np.log(scaled_energies_100,where=scaled_energies_100 > 0), cmap='Greys_r',)
hist3 = ax3.hist2d(etas_200, phis_200, bins=(created_bins, created_bins), weights=np.log(scaled_energies_200, where=scaled_energies_200 > 0), cmap='Greys_r',)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(hist3[3], cax=cax,label='Scaled Energies')
cbar.set_ticks([0, 16, 32, 48, 64])

# Calculate bin indices
bin_indices = np.arange(len(created_bins))  # Bin numbers start at 0

# Use bin numbers as axis labels
ax1.set_xlim(left=-1, right=1.001)
bin_ticks_foo = created_bins[:-1] + np.diff(created_bins) / 2
print(bin_ticks_foo)
print(bin_indices)
ax1.set_xticks(np.append(bin_ticks_foo[::8], bin_ticks_foo[-1]))
ax2.set_xticks(np.append(bin_ticks_foo[::8], bin_ticks_foo[-1]))
ax3.set_xticks(np.append(bin_ticks_foo[::8], bin_ticks_foo[-1]))
ax1.set_yticks(np.append(bin_ticks_foo[::8], bin_ticks_foo[-1]))
ax1.set_xticklabels(np.append(bin_indices[::8], [32]))
ax2.set_xticklabels(np.append(bin_indices[::8], [32]))
ax3.set_xticklabels(np.append(bin_indices[::8], [32]))
ax1.set_yticklabels(np.append(bin_indices[::8], [32]))
ax2.set_yticklabels([])
ax3.set_yticklabels([])

plt.tight_layout()
plt.savefig(f"{save_str}_energies.png", dpi=600)
# plt.savefig(f"{save_str}_energies.pdf",bbox_inches="tight")
plt.close()

