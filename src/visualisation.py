"""
Contains all plotting and visualisation tools. 
"""

# Package imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Circle

# Local imports
from calculate_quantities import p_magnitude, pseudorapidity, to_phi

PDG_IDS = {
    0: r"$\gamma$ (Photon)",
    11: r"$e^-$ (Electron)",
    -11: r"$e^+$ (Positron)",
    22: r"$\gamma$ (Photon)",
    130: r"$K^0_S$ (K-short)",
    211: r"$\pi^+$ (Pion)",
    -211: r"$\pi^-$ (Pion)",
    321: r"$K^+$ (Kaon)",
    -321: r"$K^-$ (Kaon)",
    2112: r"$n$ (Neutron)",
    -2112: r"$\bar{n}$ (Antineutron)",
    2212: r"$p$ (Proton)",
}

# Global color scheme (tab20 for extended color range)
unique_abs_pdgids = sorted(abs(pdgid) for pdgid in PDG_IDS.keys())
cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, len(unique_abs_pdgids))))
GLOBAL_CMAP = {pid: cmap(i) for i, pid in enumerate(unique_abs_pdgids)}

def plot_detections(
    tt_bar,
    pile_ups,
    # plot_data,
    centre,
    jet_no=0,
    filename="eta_phi",
    base_radius_size=1,
    momentum_display_proportion=1.0,
    verbose=True,
    pdgids=False,
    cwd=".",
) -> None:
    """Plot a jet and output to a PNG. Obvious.

    Parameters
    ----------
    tt_bar: ndarray
        2D dataset containing jets
    pile_ups: ndarray
        2D dataset containing pile_ups
    centre: (float,float)
        (eta,phi) of the jet axis.
    filename: str
        The name to save the file as (PNG)
    base_dot_size: float
        Base size for plot circles
    momentum_display_proportion: flost
        proportion of total momentum to display in plot limits - default 1.0 (display all detections)
    verbose: bool
        print detailed information
    pgdids: bool, default = False
        Whether to highlight PDGIDs in a legend. Useful to showcase decay products, but for just telling where the jets are, useful to leave off, When off (False), 2 colours are used: 1 for al jet constituents, the other for pile-ups
    Returns
    -------
    """
    # Retrieve data for the specified jet and calculate momenta and angles
    plot_data = np.concatenate((tt_bar, pile_ups), axis=0)
    momenta = plot_data[:, 3:]
    pmag = p_magnitude(momenta)
    if verbose:
        print("Constituent momenta magnitudes:\n", pmag)
    pz = plot_data[:, 5]
    eta = pseudorapidity(pmag, pz)
    phi = to_phi(momenta[:, 0], momenta[:, 1])

    # Variable dot sizes, prop to pmag
    radius_sizes = 0.1 * base_radius_size * (pmag / np.max(pmag))

    # Get colours from global cmap based on PDG IDs
    pdgid_values = plot_data[:, 1]
    colours = [GLOBAL_CMAP.get(abs(pid), "black") for pid in pdgid_values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(
        f"$\phi$ vs $\eta$ of jet {jet_no}, tot_num_parts={len(plot_data)}, mmtm_crop={momentum_display_proportion}"
    )
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\phi$")
    # Set phi range to +/-pi and adjust tick marks
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / 4))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda val, pos: f"{(val / np.pi)}$\pi$" if val != 0 else "0"
        )
    )
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.7)
        # Add circle to visualise boundary and jet centre
    ax.plot(centre[0], centre[1], marker="o", color="blue")
    boundary_circle = plt.Circle(centre, 1.0, fill=False)
    ax.add_patch(boundary_circle)

    # Show particles up to the target proportion of total momentum
    sorted_indices = np.argsort(pmag)[::-1]
    if momentum_display_proportion != 1.0:  # Do some calculating for the cutoff idx
        total_momentum = np.sum(pmag)
        cumulative_momentum = np.cumsum(pmag[sorted_indices])
        target_momentum = total_momentum * momentum_display_proportion
        cutoff_index = np.searchsorted(cumulative_momentum, target_momentum, side='right')

        # Delete unwanted particles from plotting data
        pdgid_values = pdgid_values[sorted_indices][:cutoff_index]
        eta = eta[sorted_indices][:cutoff_index]
        phi = phi[sorted_indices][:cutoff_index]
        colours = [colours[i] for i in sorted_indices[:cutoff_index]]
        radius_sizes = radius_sizes[sorted_indices[:cutoff_index]]
        radius_sizes *= 5  # Make larger radius for cropped plots, consider making this variable in future
        linewidth = 1
    else:  # Show all particles
        cutoff_index = None
        linewidth = 0.1

    # Plot centres
    # FIX THIS FOR CROPS
    # dot_sizes = radius_sizes*radius_sizes  # Dots sizes based on area so scale as square

    if pdgids:
        # Plot circles prop to width
        ax.scatter(eta, phi, color=colours, marker='.', edgecolors='none', s=0)
        for pdgid, e, p, color, radius in zip(pdgid_values, eta, phi, colours, radius_sizes):
            linestyle = "-" if pdgid >= 0 else "--"
            circle = Circle((e, p), radius=radius/100, edgecolor=color, facecolor='none', linewidth=linewidth, linestyle=linestyle, fill=False)
            ax.add_patch(circle)
    

        # Add legend for pdgid values and particle names
        # NB this will show all particles in the collision in the legend, even if cropped out (is desired behaviour)
        handles = []
        pdgid_values = plot_data[:, 1]  # Reset toget all PDG IDs again, in case some lost from the crop
        unique_detected_pdgids = sorted(set(pdgid_values))
        unique_abs_detected_pdgids = sorted(set(abs(i) for i in pdgid_values))

        # Arrange legend in ascending abs PDG IDs, with antiparts below if detected
        for abs_pid in unique_abs_detected_pdgids:
            colour = GLOBAL_CMAP.get(abs_pid, "grey")
            if abs_pid in unique_detected_pdgids:
                pid = abs_pid
                particle_name = PDG_IDS.get(pid, "Not in PDGID dict")
                handles.append( Patch(
                    label=f"PDG ID: {int(pid)}, \n{particle_name}",
                    color=colour
                ) )
            if -abs_pid in unique_detected_pdgids:
                pid = -abs_pid
                particle_name = PDG_IDS.get(pid, "Not in PDGID dict")
                handles.append( Patch(
                    label=f"PDG ID: {int(pid)}, \n{particle_name}",
                    edgecolor=colour, facecolor="none"
                ) )
        
        # Resize main plot and put legend to right
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.scatter(eta[0:len(tt_bar)], phi[0:len(tt_bar)], color="red", marker='.', edgecolors='none', s=0, label="Jet particles")
        ax.scatter(eta[len(tt_bar):], phi[len(tt_bar):], color="lightblue", marker='.', edgecolors='none', s=0, label="Pile-ups")
        ax.legend()
        for pdgid, e, p, color, radius in zip(pdgid_values, eta[:len(tt_bar)], phi[:len(tt_bar)], colours, radius_sizes):
            # linestyle = "-" if pdgid >= 0 else "--"
            circle = Circle((e, p), radius=radius/100, edgecolor="red", facecolor='none', linewidth=linewidth, fill=False)
            ax.add_patch(circle)
        for pdgid, e, p, color, radius in zip(pdgid_values, eta[len(tt_bar):], phi[len(tt_bar):], colours, radius_sizes):
            # linestyle = "-" if pdgid >= 0 else "--"
            circle = Circle((e, p), radius=radius/100, edgecolor="blue", facecolor='none', linewidth=linewidth, fill=False)
            ax.add_patch(circle)

    plt.savefig(f"{cwd}/data/plots/test/{filename}.png", dpi=1000)
    plt.savefig(f"{cwd}/data/plots/test/{filename}.pdf",)
    plt.close()

def count_hist(
    eta,
    phi,
    jet_no,
    bins=(10,10),
    filename="eta_phi",
    cwd=".",
) -> None:
    """
    Plots a 2D histogram of particle counts (colour map) against eta and phi bins.

    Parameters
    ----------
    eta: ndarray
        1D dataset containing particle etas
    phi: ndarray
        1D dataset containing particle phis
    jet_no: int,
        Select jet to plot (only useful for the title)
    bins: (int, int)
        Number of (eta,phi) bins to use. Default: (10,10).
    filename: str
        The name to save the file as (PNG & PDF)

    Returns
    ---------
    None
    """
    plt.figure(figsize=(8, 6))
    plt.hist2d(eta, phi, bins=bins, cmap='Greys')  # Use grayscale colormap

    # Customizing the plot
    plt.colorbar(label='Number of Particles')  # Colorbar to show counts
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\phi$')
    plt.title(
        f"$\phi$ vs $\eta$ of jet {jet_no}, tot_num_parts={len(eta)}, bins={bins}"
    )
    plt.savefig(f"{cwd}/data/hist/{filename}_bins_{bins}_hist.png", dpi=600)
    plt.savefig(f"{cwd}/data/hist/{filename}_bins_{bins}_hist.pdf",)
    plt.close()

def energy_hist(
    eta,
    phi,
    energies,
    jet_no,
    bins=(10,10),
    filename="eta_phi",
    cwd=".",
) -> None:
    """
    Plots a 2D histogram of particle counts (colour map) against eta and phi bins.

    Parameters
    ----------
    eta: ndarray
        1D dataset containing particle etas
    phi: ndarray
        1D dataset containing particle phis
    energies: ndarray
        1D dataset containing particle energies (to use for histogram)
    jet_no: int,
        Select jet to plot (only useful for the title)
    bins: (int, int)
        Number of (eta,phi) bins to use. Default: (10,10).
    filename: str
        The name to save the file as (PNG & PDF)

    Returns
    ---------
    None
    """
    plt.figure(figsize=(8, 6))
    plt.hist2d(eta, phi, bins=bins, weights=energies, cmap='Greys',)

    # Customizing the plot
    plt.colorbar(label='Energies')  # Colorbar to show counts
    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$\phi$')
    plt.title(
        f"$\phi$ vs $\eta$ of jet {jet_no}, tot_num_parts={len(eta)}, bins={bins}"
    )
    plt.savefig(f"{cwd}/data/hist/{filename}_bins_{bins}_energies.png", dpi=600)
    plt.savefig(f"{cwd}/data/hist/{filename}_bins_{bins}_energies.pdf",)
    plt.close()