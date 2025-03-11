"""Original code written by Roy Siu. Legend code written by Claude 3.5."""

# Package imports
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Local imports
from DDPMLHC.config import *

def relative_resolution(ground_truth, comparison):
    if len(ground_truth) != len(comparison):
        raise IndexError(f"Lengths mismatched: ground_truth ({len(ground_truth)}) != comparison ({len(comparison)})")
    if np.any(ground_truth == 0):
        raise ZeroDivisionError(f"Zero(s) in ground truth")
    return (comparison - ground_truth) / ground_truth

def load_variable_data(data_path, variable, truth_path=None):
    if truth_path is None:
        truth_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level.csv"
    truth = pl.read_csv(truth_path)[variable].to_numpy()
    data = pl.read_csv(data_path)[variable].to_numpy()
    
    if len(data) != len(truth):
        print(f"WARNING - Length mismatch in data: truth {len(truth)} != data {len(data)}. Using shortest length.")
        min_len = min(len(data), len(truth))
        data = data[:min_len]
        truth = truth[:min_len]
    
    # Delete known bad event
    bad_idx = 24716
    truth = np.delete(truth, bad_idx)
    data = np.delete(data, bad_idx)

    return relative_resolution(truth, data)

def plot_resolutions(mass_resolutions, pt_resolutions, colors=None, 
                    bins=BMAP_SQUARE_SIDE_LENGTH, save_path=None,
                    mass_cutoff=(-1, 4), pt_cutoff=(-1, 2), use_log=False, fig_vinch=4,
                    legend_title=None, show_subtit=True, 
                    mass_text=None, pT_text=None):
    
    if legend_title is None:
        legend_title = rf"${bins}\times {bins}$ grid"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, fig_vinch))
    
    # Mass plot
    hist_lines = []
    for label, res in mass_resolutions.items():
        plot_data = res[(res >= mass_cutoff[0]) & (res < mass_cutoff[1])]
        bias = np.mean(plot_data)
        resolution = np.std(plot_data)
        stat_label = f"{label}\n$\zeta={bias:.3f}$\n$\\rho={resolution:.3f}$"
        
        color = colors.get(label) if colors else None
        line = ax1.hist(plot_data, 
                        bins=np.linspace(mass_cutoff[0], mass_cutoff[1], 50),
                        label=stat_label, 
                        histtype="step",
                        density=True,
                        color=color)
        hist_lines.append(line)
    
    ax1.axvline(x=0, color="black", linestyle="--")
    if show_subtit: ax1.set_xlabel("Mass response")
    ax1.set_ylabel("Frequency density")
    if use_log: ax1.set_yscale("log")
    
    # Add text to top right corner
    if mass_text:
        ax1.text(0.95, 0.95, mass_text, transform=ax1.transAxes, 
                 fontsize=LABEL_FONTSIZE, verticalalignment='top', horizontalalignment='right')
    
    # Create custom legend handles
    handles = []
    for (label, res), line in zip(mass_resolutions.items(), hist_lines):
        plot_data = res[(res >= mass_cutoff[0]) & (res < mass_cutoff[1])]
        bias = np.mean(plot_data)
        resolution = np.std(plot_data)
        stat_label = f"{label}\n$\zeta={bias:.3f}$\n$\\rho={resolution:.3f}$"
        
        color = line[2][0].get_edgecolor()
        
        handles.append(Line2D([0], [0], 
                            color=color,
                            marker='|', 
                            markersize=45, 
                            markeredgewidth=2,
                            label=stat_label))
    
    # legend_font=TICK_AND_LEGEND_FONTSIZE
    if legend_title == "":
        ax1.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                  labelspacing=1.0,
                  handlelength=0,
                  handletextpad=0.5,
                  frameon=False,
                  prop={'size': TICK_AND_LEGEND_FONTSIZE})
    else:
        ax1.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                  labelspacing=1.0,
                  handlelength=0,
                  handletextpad=0.5,
                  title=legend_title,
                  frameon=False,
                  prop={'size': TICK_AND_LEGEND_FONTSIZE})
    
    # pT plot (similar approach)
    hist_lines = []
    for label, res in pt_resolutions.items():
        plot_data = res[(res >= pt_cutoff[0]) & (res < pt_cutoff[1])]
        bias = np.mean(plot_data)
        resolution = np.std(plot_data)
        stat_label = f"{label}\n$\zeta={bias:.3f}$\n$\\rho={resolution:.3f}$"
        
        color = colors.get(label) if colors else None
        line = ax2.hist(plot_data, 
                        bins=np.linspace(pt_cutoff[0], pt_cutoff[1], 50),
                        label=stat_label, 
                        histtype="step",
                        density=True,
                        color=color)
        hist_lines.append(line)
    
    ax2.axvline(x=0, color="black", linestyle="--")
    if show_subtit: ax2.set_xlabel(r"$p_T$ response")
    if use_log: ax2.set_yscale("log")
    
    # Add text to top right corner
    if pT_text:
        ax2.text(0.95, 0.95, pT_text, transform=ax2.transAxes, 
                 fontsize=LABEL_FONTSIZE, verticalalignment='top', horizontalalignment='right')
    
    # Create custom legend handles for pT plot
    handles = []
    for (label, res), line in zip(pt_resolutions.items(), hist_lines):
        plot_data = res[(res >= pt_cutoff[0]) & (res < pt_cutoff[1])]
        bias = np.mean(plot_data)
        resolution = np.std(plot_data)
        stat_label = f"{label}\n$\zeta={bias:.3f}$\n$\\rho={resolution:.3f}$"
        
        color = line[2][0].get_edgecolor()
        
        handles.append(Line2D([0], [0], 
                            color=color,
                            marker='|', 
                            markersize=45, 
                            markeredgewidth=2,
                            label=stat_label))
    
    if legend_title == "":
        ax2.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                  labelspacing=1.0,
                  handlelength=0,
                  handletextpad=0.5,
                  frameon=False,
                  prop={'size': TICK_AND_LEGEND_FONTSIZE})
    else:
        ax2.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                  labelspacing=1.0,
                  handlelength=0,
                  handletextpad=0.5,
                  title=legend_title,
                  frameon=False,
                  prop={'size': TICK_AND_LEGEND_FONTSIZE})
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{CWD}/data/plots/relative_resolutions/resolutions_test.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
