import numpy as np
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl

# ======= global matplotlib params =====
plt.rcParams['text.usetex'] = False  # Use LaTeX for rendering text
plt.rcParams['font.size'] = 12      # Set default font size (optional)

file_dir = os.path.dirname(os.path.realpath(__file__))
pile_path = f"{file_dir}/../data/1-initial/pileup.csv"
tt_path = f"{file_dir}/../data/1-initial/ttbar.csv"
# jetnumber , pdgid , charge , px , py , pz
pile_up = np.genfromtxt(pile_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=10)
tt = np.genfromtxt(tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=1000)

max_tt_num = np.max(tt[:,0])
max_pile_num = np.max(pile_up[:,0])
# tt_bar = np.loadtxt(tt_path)
def select_jet(data, num):
  """
  Select data with jets #num from data file
  """
  if max_tt_num <= num:
    print(f"Requested jet {num} is not in data. Max jet number is {max_tt_num}")
    sys.exit(1)
  return data[data[:,0] == num]

def p_magnitude(p):
  """
  p is a 2D NumPy array where each element is a particle's 3-momentum

  This function simply calculates and returns the magnitude of the momenta of each particle and returns it as a 1D NumPy array
  """
  return np.linalg.norm(p, axis=1)

def pseudorapidity(p_mag, p_z):
  """
    p_mag is a 1D NumPy array where each element is the magnitude of momentum for a particle
    p_z is the z-component of momentum of a particle (the component along the beam-axis)

    this function calculates the pseudorapidity and returns it as a 1D NumPy array
    https://en.wikipedia.org/wiki/Pseudorapidity
  """
  if ( np.shape(p_mag) != np.shape(p_z)):
    print("Error: p_mag shape not equal to p_z shape")
    sys.exit(1)
  return np.arctanh(p_z/p_mag)

def to_phi(p_x, p_y):
  """
  p_x, p_y are 1D NumPy arrays of x and y momenta respectively, where each element corresponds to a particle.

  This function finds the angle phi (radians) from the 2 components of transverse momentum p_x,  p_y using arctan(p_y/p_x)
  """
  if (np.shape(p_x) != np.shape(p_y)):
    print("Error: p_x shape not equal to p_y shape")
    sys.exit(1)
  return np.arctan2(p_y, p_x)

jet_no = 0
data = select_jet(tt, jet_no)
print(data)
tt_momenta = data[:,3:]
# print(tt_momenta)
tt_pmag = p_magnitude(tt_momenta)
print(tt_pmag)
tt_pz = data[:,5]
# print(tt_pz)
tt_eta = pseudorapidity(tt_pmag, tt_pz)
tt_phi = to_phi(tt_momenta[:,0], tt_momenta[:,1])
fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(tt_eta, tt_phi, color='blue', marker='o', s=0.1)

# Add titles and labels
plt.title(f"$\phi$ vs $\eta$ of jet {jet_no}")
ax.set_xlabel("$\eta$")
ax.set_ylabel("$\phi$")
plt.savefig("eta_phi.png", format="png", dpi=100)
# plt.close()
# print("done")
sys.exit(0)