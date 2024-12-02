"""Make intermediate data to avoid repeated computation."""

# Import constants
from config import *

# Package imports
import numpy as np

# Local imports
from calculate_quantities import *

save_path = f"{CWD}/data/2-intermediate/"

# === BEGIN Reading in Data (tt) ===

tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

jet_ids = tt[:, 0].astype(int)
pxs, pys, pzs = tt[:, 3], tt[:, 4], tt[:, 5]

# ===== Create Individual Particle Data (t-tbar) ===== #

energies = p_magnitude(pxs, pys, pzs)
etas = pseudorapidity(energies, pzs)
pTs = np.sqrt(pxs**2 + pys**2)
phis = to_phi(pxs, pys)
pTs = to_pT(pxs, pys)

p2s = contraction(energies, pxs, pys, pzs)
masses = np.sqrt(p2s)

# Save to CSV
combined_array = np.array([jet_ids, pxs, pys, pzs, etas, phis, pTs]).T
np.savetxt(f"{save_path}/ttbar_extended.csv", combined_array, delimiter=",", header="jet_id, px, py, pz, eta, phi, p_T", comments="", fmt="%i, %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f")

# ===== Create Jet Data (t-tbar) ===== #

jet_ids_unique = np.unique(jet_ids)
jet_enes, jet_pxs, jet_pys, jet_pzs = calculate_four_momentum_massless(jet_ids, pxs, pys, pzs)
jet_p2s = contraction(jet_enes, jet_pxs, jet_pys, jet_pzs)
jet_masses = np.sqrt(jet_p2s)
jet_etas = pseudorapidity(jet_enes, jet_pzs)
jet_phis = to_phi(jet_pxs, jet_pys)
jet_pTs = to_pT(jet_pxs, jet_pys)

# Kinda fun to print
for jet_id in range(0, len(jet_ids_unique), 1):
    four_mmtm = jet_ids[jet_id]
    p2 = jet_p2s[jet_id]
    print(f"Jet ID: {jet_id}, Total 4-Momenta: [{jet_enes[jet_id]:.3f}, {jet_pxs[jet_id]:.3f}, {jet_pys[jet_id]:.3f}, {jet_pzs[jet_id]:.3f}], Mass: {jet_masses[jet_id]:.3f}")

# Save to CSV
combined_array = np.array([jet_ids_unique, jet_pxs, jet_pys, jet_pzs, jet_etas, jet_phis, jet_masses, jet_pTs]).T
np.savetxt(f"{save_path}/ttbar_jets.csv", combined_array, delimiter=",", header="jet_id, px, py, pz, eta, phi, mass, p_T", comments="", fmt="%i, %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f")

# === BEGIN Reading in Data (PU) ===

pu = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)

pileup_ids = pu[:, 0].astype(int)
pxs, pys, pzs = pu[:, 3], pu[:, 4], pu[:, 5]

# ===== Create Individual Particle Data (pile-up) ===== #

energies = p_magnitude(pxs, pys, pzs)
etas = pseudorapidity(energies, pzs)
pTs = np.sqrt(pxs**2 + pys**2)
phis = to_phi(pxs, pys)
pTs = to_pT(pxs, pys)

p2s = contraction(energies, pxs, pys, pzs)
masses = np.sqrt(p2s)

# Save to CSV
combined_array = np.array([pileup_ids, pxs, pys, pzs, etas, phis, pTs]).T
np.savetxt(f"{save_path}/pileup_extended.csv", combined_array, delimiter=",", header="pileup_id, px, py, pz, eta, phi, p_T", comments="", fmt="%i, %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f,  %10.10f")
