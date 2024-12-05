"""Docstring here"""

# Package imports
import numpy as np

# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *

# MU300_PATH = f"{CWD}/data/2-intermediate/noisy_mu300.csv"
# mu = 300
MU10_PATH = f"{CWD}/data/2-intermediate/noisy_mu10.csv"
mu = 10
save_path = f"{CWD}/data/2-intermediate/"

noisy_data = np.genfromtxt(
    MU10_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

NIDs = noisy_data[:, 0].astype(int)
LIDs = noisy_data[:, 1].astype(int)
pxs, pys, pzs = noisy_data[:, 2], noisy_data[:, 3], noisy_data[:, 4]
etas = noisy_data[:, 5]
phis = noisy_data[:, 6]
enes = noisy_data[:, 7]
pTs = to_pT(pxs, pys)

p2s = contraction(enes, pxs, pys, pzs)
masses = np.sqrt(p2s)

# ===== Create Noisy Event Data ===== #

NIDs_unique = np.unique(NIDs)
event_enes, event_pxs, event_pys, event_pzs = calculate_four_momentum_massless(NIDs, pxs, pys, pzs)
event_p2s = contraction(event_enes, event_pxs, event_pys, event_pzs)
event_masses = np.sqrt(event_p2s)
event_etas = pseudorapidity(event_enes, event_pzs)
event_phis = to_phi(event_pxs, event_pys)
event_pTs = to_pT(event_pxs, event_pys)

# Kinda fun to print
for event_id in range(0, len(NIDs_unique), 1):
    four_mmtm = NIDs[event_id]
    p2 = event_p2s[event_id]
    print(f"NID: {event_id}, Total 4-Momenta: [{event_enes[event_id]:.3f}, {event_pxs[event_id]:.3f}, {event_pys[event_id]:.3f}, {event_pzs[event_id]:.3f}], Mass: {event_masses[event_id]:.3f}")

# Save to CSV
combined_array = np.array([NIDs_unique, event_pxs, event_pys, event_pzs, event_etas, event_phis, event_masses, event_pTs]).T
# np.savetxt(f"{save_path}/noisy_event_stats_mu300.csv", combined_array, delimiter=",", header="event_id, px, py, pz, eta, phi, mass, p_T", comments="", fmt="%i, %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f")
np.savetxt(f"{save_path}/noisy_event_stats_mu{mu}.csv", combined_array, delimiter=",", header="event_id, px, py, pz, eta, phi, mass, p_T", comments="", fmt="%i, %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f")
print("Saved.")