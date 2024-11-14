"""Make intermediate data to avoid repeated computation."""

# Import constants
from config import *

# Package imports
import numpy as np

# Local imports
from calculate_quantities import *

# === BEGIN Reading in Data ===
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

# Calculate quantities
# Get data
num = len(tt)
jet_ids = tt[:, 0]
jet_ids_unique = np.unique(jet_ids)
px, py, pz = tt[:, 3], tt[:, 4], tt[:, 5]
# # Basic stats
# p_mag = p_magnitude(px, py, pz)
# eta = pseudorapidity(p_mag, pz)
# p_T = np.sqrt(px**2 + py**2)

# Stats for jets
jet_enes, jet_pxs, jet_pys, jet_pzs = get_jet_four_momentum(jet_ids, px, py, pz)
jet_p2s = contraction2(jet_enes, jet_pxs, jet_pys, jet_pzs)
jet_masses = np.sqrt(jet_p2s)
jet_etas = pseudorapidity(jet_enes, jet_pzs)
jet_phis = to_phi(jet_pxs, jet_pys)
jet_p_Ts = np.sqrt(jet_pxs**2 + jet_pys**2)

# Kinda fun to print
for jet_id in range(0, len(jet_ids_unique), 1):
    four_mmtm = jet_ids[jet_id]
    p2 = jet_p2s[jet_id]
    print(f"Jet ID: {jet_id}, Total 4-Momenta: [{jet_enes[jet_id]:.3f}, {jet_pxs[jet_id]:.3f}, {jet_pys[jet_id]:.3f}, {jet_pzs[jet_id]:.3f}], Mass: {jet_masses[jet_id]:.3f}")

# Save to CSV
save_file = f"{CWD}/data/2-intermediate/ttbar_jets.csv"
combined_array = np.array([jet_ids_unique, jet_pxs, jet_pys, jet_pzs, jet_etas, jet_phis, jet_masses, jet_p_Ts]).T
np.savetxt(save_file, combined_array, delimiter=",", header="jet_id, px, py, pz, eta, phi, mass, p_T", comments="", fmt="%10.10f")
