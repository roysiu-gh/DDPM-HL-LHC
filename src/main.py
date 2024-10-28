import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# ======= global matplotlib params =====
plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text
plt.rcParams['font.size'] = 12      # Set default font size (optional)

pile_path = "../data/1-initial/pileup.csv"
tt_path = "../data/1-initial/ttbar.csv"
# jetnumber , pdgid , charge , px , py , pz
pile_up = np.genfromtxt(pile_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=10)
tt = np.genfromtxt(tt_path, delimiter=",", encoding="utf-8", skip_header=1, max_rows=10)
# tt_bar = np.loadtxt(tt_path)

def p_magnitude(p_x, p_y, p_z):
  """
  p_x, p_y, p_z are 1D NumPy arrays where each element consists of a particle's component of momentum

  This function simply calculates and returns the magnitude of the momenta of each particle and returns it as a 1D NumPy array
  """
  if (np.shape(p_x) != np.shape(p_y)):
    print("Error: p_x shape not equal to p_y shape")
    sys.exit(1)
  if ( np.shape(p_y) != np.shape(p_z)):
    print("Error: p_y shape not equal to p_z shape")
    sys.exit(1)
  if ( np.shape(p_x) != np.shape(p_z)):
    print("Error: p_x shape not equal to p_z shape")
    sys.exit(1)

  return np.sqrt(p_x * p_x + p_y*p_y + p_z*p_z)

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
  return np.arctan(p_y / p_x)

tt_momenta = tt[:,3:]
tt_pmag = p_magnitude(tt_momenta[:,0], tt_momenta[:,1], tt_momenta[:,2])
tt_pz = tt[:,5]
tt_eta = pseudorapidity(tt_pmag, tt_pz)
tt_phi = to_phi(tt_momenta[:,0], tt_momenta[:,1])
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(tt_eta, tt_phi, color='blue', marker='o')

# Add titles and labels
plt.title("Scatter Plot of $\phi$ vs $\eta$")
ax.set_xlabel("$\eta$")
ax.set_ylabel("$\phi$")
plt.savefig("eta_phi", format="png", dpi=100)
# plt.close()
# print("done")
sys.exit(0)