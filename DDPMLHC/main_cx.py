# %%
import torch
from torch import optim
from torch.utils.data import Subset, Dataset, DataLoader, IterableDataset, TensorDataset
import torchvision.transforms as T
import torch.nn.functional as F
# from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime
from torch.amp import autocast
import math
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
# req torch, torchvision, einops, tqdm, ema_pytorch, accelerate
# from IPython.display import display
from einops import rearrange, reduce, repeat
import glob
from ema_pytorch import EMA
from scipy.optimize import linear_sum_assignment
from accelerate import Accelerator
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *
from DDPMLHC.generate_plots.histograms_1d import plot_1d_histograms, plot_event_level_quantities_comparison, plot_particle_level_quantities_comparison, plot_particle_level_quantities_ttbar_only
from DDPMLHC.generate_plots.overlaid_1d import create_overlay_plots
from DDPMLHC.generate_plots.overlaid_debin import create_overlay_plots_debin
from DDPMLHC.generate_plots.bmap import plot_mu_comparison, save_to_bmap
from DDPMLHC.generate_plots.overlaid_general import create_overlay_plots_general
from DDPMLHC.generate_plots.resolution_plots import *


import os
import sys
import gc
CWD = os.getcwd()

# Device stuff
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU Device:", torch.cuda.get_device_name(0))
#     print("Number of GPUs:", torch.cuda.device_count())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Remember to use {device} device from here on")
# print(os.chdir("../"))
# %cd /home/physics/phuqza/E9/DDPM-HL-LHC/
# from DDPMLHC.config import *
# from DDPMLHC.calculate_quantities import *
# from DDPMLHC.data_loading import *
# # from DDPMLHC.generate_plots.overlaid_1d import create_overlay_plots
# from DDPMLHC.generate_plots.bmap import *
# from DDPMLHC.generate_plots.histograms_1d import *
# from DDPMLHC.model_utils import *
# from DDPMLHC.generate_plots.resolution_plots import *
BMAP_SQUARE_SIDE_LENGTH = 16

# # from 
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
pile_up = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = EventSelector(tt)
pile_up = EventSelector(pile_up)

# Some functions from denoising_diffusion_pytorch that are required but couldn't import
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# %%

# MAX_DATA_ROWS = None
# bins=BMAP_SQUARE_SIDE_LENGTH
# %%
# Generate samples
# batch_size = 4
# # NG_jet.reset()
# # NG_pu.reset()
# sampled_images = diffusion.sample(batch_size=batch_size)
# show_tensor_images(sampled_images, scale_factor=10)
# # pats = [ng_for_dataloader[0],ng_for_dataloader[1],ng_for_dataloader[2],ng_for_dataloader[3],ng_for_dataloader[4],ng_for_dataloader[5], ng_for_dataloader[-1]]
# # show_tensor_images(pats, scale_factor=10)

# %%
# Print Diagnostics right before training
mu = 200
# train_batch_size = 200
# # Sampling: only load checkpoint
# num_epochs = 0

# print_params(mode="SAMPLING")
# # this one is to be passed into DataLoader for training
# ng_for_dataloader = NGenForDataloader(NG_jet)
# dataloader = DataLoader(ng_for_dataloader, batch_size=train_batch_size, num_workers=2, shuffle = True, pin_memory = True)
# diffusion_old = PUDiffusion(
#     model = model,
#     puNG = NG_pu,
#     jet_ng= NG_jet,
#     image_size = BMAP_SQUARE_SIDE_LENGTH, 
#     timesteps = 200,  # Number of diffusion steps
#     objective = "pred_x0",
# ).to(device)
beta="001"
# print("beta", beta)
# # save_dir = f"{CWD}/data/ML/Unet{UNET_DIMS}_bins{BMAP_SQUARE_SIDE_LENGTH}_mu{mu}"
# save_dir = f"{CWD}/data/ML/Unet{UNET_DIMS}_bins{BMAP_SQUARE_SIDE_LENGTH}_mu{mu}_beta{beta}"

# print("Begin training")
# xd = load_and_train(diffusion, dataloader, num_epochs=0, device=device, save_dir=save_dir)
# print("Finished training")

# # %%
# # Generate samples
# # NG_jet.reset()
# # NG_pu.reset()
# # sampled_images = diffusion.sample(batch_size=100)
# output_path = f"{CWD}/data/3-grid/Unet{UNET_DIMS}_bins{bins}_mu{mu}_beta{beta}"
# output_filename = f"noisy_mu{mu}_event_level_from_grid{bins}.csv"
# output_filepath = f"{output_path}/{output_filename}"
# histogram_path = f"{output_path}/grid{bins}_hist_beta{beta}"
# # mpl.rcParams.update(MPL_GLOBAL_PARAMS)
# if not(os.path.exists(output_path)):
#     os.makedirs(output_path,exist_ok=True)
# if not(os.path.exists(histogram_path)):
#     os.makedirs(histogram_path,exist_ok=True)
    
# # def tensor_to_data(tensor_images):
# #     # tensor_images_cpu = tensor_images.detach().cpu().numpy()
# #     save_image(tensor_images, f"{histogram_path}/saved_denoised_grids_new.png")


# class OutData():
#     def __init__(self, diffusion, NG_jet, num_jets_to_process, bins=BMAP_SQUARE_SIDE_LENGTH, num_saved=4):
#         self.diffusion = diffusion
#         self.NG_jet = NG_jet
#         self.num_jets = num_jets_to_process
#         self.bins = bins
#         self.num_saved_row =  int(np.sqrt(num_saved))
#         #self.diffusion.begin_sample = 0 # ensure starting from first jet for sampling
#         self.diffusion.reset_sample()
#         self.jets_to_plot = []
#     @torch.inference_mode()
#     def _batch_sample(self, rescale=False):
#         #self.diffusion.reset()
#         #self.diffusion.reset_sample()
#         # while self.diffusion.begin_sample < self.NG_jet._max_TT_no:i=0
#         i = 0
#         while i == 0:
#             i += 1
#             try:
#               sampled_images = self.diffusion.sample(batch_size=4)
#               sampled_images = sampled_images * self.NG_jet.max_energy
#               if rescale:
#                   sampled_images = torch.log1p(sampled_images)
#               sampled_images = sampled_images.detach().cpu()            
#               yield sampled_images
#               del sampled_images
#               gc.collect()
#               torch.cuda.empty_cache()
#               torch.cuda.synchronize()
#             except StopIteration:
#                 break
#         return None
#     def _calculate_event_level(self):
#         print(f"Iterating through dataset, adding noise and letting model denoise...")
#         counter = 0
#         eventid = 0
#         # sampled_images is a generator because of yield
#         # So each "element" in generator is a sample of jets
#         self.diffusion.reset_sample()
#         for sampled_images in self._batch_sample(rescale=False):
#             print("counter", counter)
#             all_data = []

#             if sampled_images is None:
#                 break
#         # rescaled = sampled_images
#         # Save first 4 jets only
        
#         # Remove channel dimension if exists
#             if counter ==0:
#                 sampled = sampled_images[:4]
#                 # print("sampled", sampled)
#                 sys.stdout.flush()
#                 sampled_scaled = torch.log1p(sampled)
#                 # save_image_larger(tensor=sampled_scaled, nrow=self.num_saved_row,fp=f"{histogram_path}/saved_denoised_grids{self.num_saved_row}.png", normalize=True)
#                 self.jets_to_plot = sampled_scaled
#                 # print("samples scaled", sampled_scaled)
#                 counter =1
#             break
#             if len(sampled_images.shape) == 4:  # (batch, channel, height, width)
#                 sampled_images = sampled_images.squeeze(1)
#             for jidx, grid in enumerate(sampled_images):
#                 # enumerate starts from 0
#                 # whereas sampled_images is a batch of grids iterating through jet dataset
#                 # Therefore use eventid as a running id to select the jet
#                 NG_jet.select_jet(eventid)
#                 axis = NG_jet.jet_axis
#                 enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=self.bins)
#                 detas, dphis = decentre(axis, detas, dphis)
#                 pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
#                 # print("OutData eventlevel")
#                 event_quantities = particle_momenta_to_event_level(enes, pxs, pys, pzs)
#                 event_mass, event_px, event_py, event_pz, event_eta, event_phi, event_pT = event_quantities
                
#                 event_level = np.array([
#                     eventid,
#                     event_px,
#                     event_py,
#                     event_pz,
#                     event_eta,
#                     event_phi,
#                     event_mass,
#                     event_pT,
#                 ])
#                 eventid +=1
#                 # combined.append(np.copy(event_level))
#                 all_data.append(np.copy(event_level)) 
           
#             all_data = np.vstack(all_data)
#             yield all_data
#         del sampled_images
#         gc.collect()
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         return None
    
#     def save_event_level(self, output_folder=f"{CWD}/data/4-reconstruction", output_filename=None):
#         # return
#         if output_filename is None:
#             output_filename = f"reconstructed_mu{self.diffusion.mu}_event_level_from_grid{self.bins}_Unet{UNET_DIMS}.csv"
#         output_path = f"{output_folder}/{output_filename}"
#         data = self._calculate_event_level()
#         # return
#         # print(data)
#         with open(output_path, 'w') as f:
#             f.write("event_id,px,py,pz,eta,phi,mass,p_T\n")
#             f.close()
#         total_events = 0
#         with open(output_path, 'a+') as f:
#             for batch_data in data:
#                 if batch_data is None:
#                     break
#                 np.savetxt(f, batch_data,
#                             delimiter=",",
#                             fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f")
#             f.close()
#         print("Done writing")

#         # total_events += len(batch_data)

#         # print(f"Processed {total_events} events so far...")

#         # print("writing")
#         # np.savetxt(
#         #     output_path,
#         #     data,
#         #     delimiter=",",
#         #     header="event_id,px,py,pz,eta,phi,mass,p_T",
#         #     comments="",
#         #     fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
#         # )
        
#         return output_path

# # Number of jets to sample - note very high memory requirement
# # model_cpu = model.


# output_path = f"{CWD}/data/plots/bmap_comparison/"
# ############### PURE JET NOISY JET AND DENOISED SIDE-BY-SIDE
# def compare_denoised(denoised_array, use_log=True):
#     save_path=f"{output_path}/denoised_comparison_linear_{beta}.png" if use_log == False else f"{output_path}/denoised_comparison_{beta}_log1p.png"
#     fig = plt.figure(figsize=(16, 6))  # Increased height slightly for labels
    
#     # Create gridspec to have better control over spacing
#     gs = fig.add_gridspec(1, 3, wspace=0.3)
#     main_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
#     # Get mu values
#     # mu_values = [0, 100, 200]
#     vmin, vmax = float('inf'), -float('inf')

#     # Store random state
#     rng_state = np.random.get_state()
    
#     # First pass to get global scaling across all jets and mu values
#     all_grids = []
#     # Get noisy jets
#     mu = 200
#     # for mu in mu_values:
#         # Reset random state to get same jets
#     np.random.set_state(rng_state)
    
#     mu_grids = []
#     NG = NoisyGenerator(tt, pu, mu=mu)
#     NG.reset()
#     NG.mu = 0
#     for _ in range(4):
#         next(NG)
#         grid = NG.get_grid(normalise=False)
#         if use_log:
#             grid = np.log1p(grid)
#         vmin = min(vmin, grid.min())
#         vmax = max(vmax, grid.max())
#         mu_grids.append(grid)
#     all_grids.append(mu_grids)
#     # gets mu = 200 grids
#     NG.reset()
#     NG.mu = mu
#     for _ in range(4):
#         next(NG)
#         grid = NG.get_grid(normalise=False)
#         if use_log:
#             grid = np.log1p(grid)
#         vmin = min(vmin, grid.min())
#         vmax = max(vmax, grid.max())
#         mu_grids.append(grid)
#     all_grids.append(mu_grids)
#     # denoised_array comes from sampling of first 4 jets from model
#     #  Assume it has been converted to numpy array already
#     # Plot each mu subplot with its 4 jets
#     all_grids.append(denoised_array)
#     # print("????", all_grids[0][0].shape)
#     letters = ['(a)', '(b)', '(c)']
#     for idx, grids in enumerate(all_grids):
#         # Create 2x2 grid for this mu
#         grid_size = grids[0].shape[0]
#         combined_grid = np.zeros((grid_size * 2, grid_size * 2))
        
#         # Fill the 2x2 grid
#         combined_grid[:grid_size, :grid_size] = grids[0]
#         combined_grid[:grid_size, grid_size:] = grids[1]
#         combined_grid[grid_size:, :grid_size] = grids[2]
#         combined_grid[grid_size:, grid_size:] = grids[3]

#         im = main_axes[idx].imshow(combined_grid, 
#                                     cmap='viridis',
#                                     vmin=vmin,
#                                     vmax=vmax,
#                                     norm=None,  # Add this to prevent automatic normalization
#                                     interpolation='nearest')
        
#         # Add grid lines to separate events
#         main_axes[idx].axhline(y=grid_size-0.5, color='white', linewidth=1)
#         main_axes[idx].axvline(x=grid_size-0.5, color='white', linewidth=1)
        
#         # # Move mu label to bottom
#         # main_axes[idx].text(0.5, -0.1, f'{letter} $\mu = {mu}$',
#         #                   transform=main_axes[idx].transAxes,
#         #                   fontsize=18, ha='center')
        
#         main_axes[idx].axis('off')
    
#     print(f"Maximum energy in any pixel: {max([grid.max() for grids in all_grids for grid in grids]):.4f}")

#     # Add colorbar with proper spacing
#     cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#     cbar = fig.colorbar(im, cax=cbar_ax)
#     cbar.set_label('$\ln(1+E)$' if use_log else 'Energy', fontsize=18)
#     # plt.tight_layout()
#     # Save if path provided
#     plt.savefig(save_path, 
#                 bbox_inches='tight', 
#                 dpi=300,
#                 pad_inches=0.2)
#     print(f"Saved figure to {save_path}")
#     plt.close()
# ##### CODE FOR GENERATING MASS/ETA/PT PLOTS####
# # events_dat = np.genfromtxt(
# #         output_path, delimiter=",", encoding="utf-8", skip_header=1
# #     )
# # mass_num_bins = 50
# # mass_max = 400
# # pT_max = 5000
# # pT_num_bins = 50
# # pT_bins = np.mgrid[0:pT_max:(pT_num_bins+1)*1j]
# # mass_bins = np.mgrid[0:mass_max:(mass_num_bins+1)*1j]
# # fig, axs = plt.subplots(1,3,figsize=(14,6))
# # axs[0].hist(events_dat[:,6], bins=mass_bins, density=True)
# # axs[1].hist(events_dat[:,4], bins=50,density=True)
# # # axs[2].hist(events_dat[:,5], bins=50,density=True)
# # axs[2].hist(events_dat[:,7], bins = pT_bins, density=True)
# # plt.savefig(f"{CWD}/data/3-grid/grid{bins}/test.pdf")
# # plt.close()

# ##### CODE TO GENERATE RESOLUTION PLOTS #####
# def generate_event_level_gridded_jets(NG: NoisyGenerator, save_dir=INTERMEDIATE_PATH):
#     NG.reset()
#     # NG.bins = BMAP_SQUARE_SIDE_LENGTH
#     gt_file = f"{save_dir}/noisy_mu0_event_level_grid{NG.bins}.csv"
#     combined = []
#     for idx, _ in enumerate(NG):
#         # next(NG)
#         grid = NG.get_grid(normalise=False)
#         # NG.select_jet(idx)
#         axis = NG.jet_axis
#         enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=NG.bins)
#         detas, dphis = decentre(axis, detas, dphis)
#         pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
#         # print("???")
#         event_quantities = particle_momenta_to_event_level(enes, pxs, pys, pzs)
#         event_mass, event_px, event_py, event_pz, event_eta, event_phi, event_pT = event_quantities
        
#         event_level = np.array([
#             idx,
#             event_px,
#             event_py,
#             event_pz,
#             event_eta,
#             event_phi,
#             event_mass,
#             event_pT,
#         ])
        
#         combined.append(np.copy(event_level))
            
#     all_data = np.vstack(combined)
#     np.savetxt(
#             gt_file,
#             all_data,
#             delimiter=",",
#             header="event_id,px,py,pz,eta,phi,mass,p_T",
#             comments="",
#             fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
#     )
#     return all_data, gt_file
# generate_event_level_gridded_jets(NoisyGenerator(tt, pu, bins=4, mu=0))
# generate_event_level_gridded_jets(NoisyGenerator(tt, pu, bins=8, mu=0))
# generate_event_level_gridded_jets(NoisyGenerator(tt, pu, bins=64, mu=0))
# generate_event_level_gridded_jets(NoisyGenerator(tt, pu, bins=256, mu=0))
# sys.exit()
# print("Done generating grid")
# def mass_energy_diff(save_dir=INTERMEDIATE_PATH, mu=200):
#     """
#     Finds the relative difference between the model's denoised images and the binned jets as ground truths
#     """
#     gt_file = f"{save_dir}/noisy_mu0_event_level_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"
#     if not os.path.isfile(gt_file):
#         generate_event_level_gridded_jets(NG_jet)

#     jet_quantities = pl.read_csv(gt_file)
#     jets_px = jet_quantities['px']
#     jets_py = jet_quantities['py']
#     jets_pz = jet_quantities['pz']
#     jets_mass = jet_quantities['mass'].to_numpy()
# # print(your[24717:])
#     jets_mass = np.concatenate((jets_mass[0:24716], jets_mass[24717:]))
        
#     jet_energy = (jets_px ** 2) + (jets_py ** 2) + (jets_pz ** 2)
#     jet_energy = jet_energy.to_numpy()
#     jet_energy = np.concatenate((jet_energy[0:24716], jet_energy[24717:]))
#     jet_energy = np.sqrt(jet_energy)
#     csv_file_paths = [f"./data/4-reconstruction/reconstructed_mu{mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}_Unet{UNET_DIMS}.csv"]
#     fig,axs = plt.subplots(nrows=len(csv_file_paths),ncols=2, figsize=(8,6))
#     #  = axs
#     energy_counts = []
#     mass_counts = []
#     mean_mass_diffs = []
#     std_mass_diffs = []
#     data_array = [pl.read_csv(csv_file_path) for csv_file_path in csv_file_paths]

#     for idx,data in enumerate(data_array):
#         df = data
#         max_id = df['event_id'].max() + 1
#         # get jet indices
#         # Will select from ground truth jets
#         # jet_indices = df['event_id']
#         px = df['px']
#         py = df['py']
#         pz = df['pz']
#         # print(px)
#         mass = df['mass']
#         mass = mass.to_numpy()
#         # mass = np.concatenate((mass[0:24716], mass[24717:]))
#         # massless limit
#         energy = (px ** 2) + (py ** 2) + (pz ** 2)
#         energy = energy.to_numpy()
#         # energy = np.concatenate((energy[0:24716], energy[24717:]))

#         energy = np.sqrt(energy)
#         # print(energy)
#         # Find energy difference between jet+pile-up and jet for feach jet_id
#         jet_energy1 = jet_energy[:max_id]
#         jets_mass1 = jets_mass[:max_id]
#         energy_diffs = energy - jet_energy1
#         energy_diffs = energy_diffs / jet_energy1
#         mass_diffs = mass - jets_mass1
#         mass_diffs2 = mass_diffs/ jets_mass1
#         # print(mass_diffs2)
#         # mean_energy_diff = np.sum(energy_diffs) / (max_id - 1)
#         # en_bins = np.mgrid[np.min(energy_diffs):np.max(energy_diffs):(len(energy_diffs)+1)*1j]
#         # mass_bins = np.mgrid[np.min(mass_diffs2):np.max(mass_diffs2):(len(mass_diffs2)+1)*1j]
#         # mass_bins = np.mgrid[0:mass_max:(mass_num_bins+1)*1j]
#         axs[0].hist(energy_diffs, bins = 50, label=f"Reconstructed", edgecolor="black")
#         axs[1].hist(mass_diffs2[mass_diffs2<5], bins = 50,label=f"Reconstructed", edgecolor="black")
#         axs[0].set_ylabel(r"Counts")
#         # axs[idx][0].set_ylabel(r"Counts")

#         axs[0].legend(prop={'size': 14})
#         axs[1].legend(prop={'size': 14})
#         # std_energy_diff = np.std(energy_diffs)
#         # print(std_energy_diff)
#         # print(mean_energy_diff)

#         # mean_mass_diff = np.sum(mass_diffs2) / (max_id - 1)
#         # std_mass_diff = np.std(mass_diffs2)

#         # mean_energy_diffs.append(mean_energy_diff)
#         # std_energy_diffs.append(std_energy_diff)
#         # mean_mass_diffs.append(mean_mass_diff)
#         # std_mass_diffs.append(std_mass_diff)
#     axs[0].set_xlabel(r"$\frac{E_{\mu}^{j} - E_{0}^{j}}{E_{0}^{j}}$")
#     axs[1].set_xlabel(r"$\frac{m_{\mu}^{j} - m_{0}^{j}}{m_{0}^{j}}$")
#     #### Plot differences for binned and unbinned data
    
#     NG_default.reset()
#     energy_bin_diffs = []
#     mass_bin_diffs = []
#     for i in range(jets_to_sample):
#         next(NG_default)
#         grid = NG_default.get_grid()
#         axis = NG_default.jet_axis
#         enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=NG_default.bins)
#         detas, dphis = decentre(axis, detas, dphis)
#         pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
#         # print("???")
#         event_quantities = particle_momenta_to_event_level(enes, pxs, pys, pzs)
#         event_mass, event_px, event_py, event_pz, event_eta, event_phi, event_pT = event_quantities
#         energy_bin = (event_px ** 2) + (event_py ** 2) + (event_pz ** 2)
#         energy_bin_diffs.append(energy_bin - jet_energy[i])
#         mass_bin_diffs.append(event_mass - jets_mass[i])
#     axs[0].hist(energy_bin_diffs, bins = 50, label=f"Best case", edgecolor="black")
#     axs[1].hist(mass_bin_diffs, bins = 50,label=f"Best case", edgecolor="black")
#     plt.tight_layout()
#     plt.savefig(f"{CWD}/data/3-grid/Unet{UNET_DIMS}_bins{bins}_mu{mu}/grid{bins}_hist/hist_energymasscounts_model.pdf", format="pdf")
#     # plt.savefig(f"{CWD}/data/plots/hist_energymasscounts_model.png", format="png", dpi=600)
#     plt.close()
# mass_energy_diff()


# %%
# import numpy as np
# import matplotlib.pyplot as plt

# data_arrays = []
# vmin, vmax =0,0

# rng_state = np.random.get_state()
# np.random.set_state(rng_state)
# use_log=True
# NG = NoisyGenerator(tt, pu, mu=mu)
# NG.reset()
# NG.mu = 0
# for _ in range(4):
#     next(NG)
#     grid = NG.get_grid(normalise=False)
#     if use_log:
#         grid = np.log1p(grid)
#     vmin = min(vmin, grid.min())
#     vmax = max(vmax, grid.max())
#     # mu_grids.append(grid)
#     data_arrays.append(grid)
# # gets mu = 200 grids
# NG.reset()
# NG.mu = mu
# for _ in range(4):
#     next(NG)
#     grid = NG.get_grid(normalise=False)
#     if use_log:
#         grid = np.log1p(grid)
#     vmin = min(vmin, grid.min())
#     vmax = max(vmax, grid.max())
#     data_arrays.append(grid)

# print("lem  data arrays before tensors", len(data_arrays))
# with torch.inference_mode():
#     model.eval()
#     diffusion.eval()
#     # sampled_images = diffusion.sample(batch_size=batch_size)
#     # rescaled = sampled_images * NG_jet.max_energy
#     # tensor_to_data(rescaled)
#     output_folder=f"{CWD}/data/4-reconstruction/beta{beta}"
#     output_filename = f"reconstructed_mu{diffusion.mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}_Unet{UNET_DIMS}.csv"

#     OD = OutData(diffusion, NG_jet, 4)
#     output_path = OD.save_event_level(output_folder=output_folder, output_filename=output_filename)
#     jets_to_plot = OD.jets_to_plot
#     # print("???",jets_to_plot)
#     denoised_jets = jets_to_plot.detach().cpu().numpy()
#     # print(denoised_kets.shape)
#     denoised_jets = np.squeeze(denoised_jets, axis=1)
#     print(denoised_jets.shape)
#     for jet in denoised_jets:
#         print("single jet", jet.shape)
#         # print("grid?", jet)
#         data_arrays.append(jet)

#     # compare_denoised(jets_to_plot)

# torch.cuda.empty_cache()



# # print(data_arrays[9])# print(data_arrays)
# print("DONE")
# # for grid in data_arrays:
#     # print("grid?", grid)

# # Assume you have 12 2D arrays (1 row × 3 columns, each with 4 plots)
# # Replace this with your actual data
# # data_arrays = [np.random.rand(64, 64) for _ in range(12)]



# # %%
# # Create a figure with 1 row, 3 columns
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# # Loop through each subplot (column)
# for subplot_idx in range(3):
#     # Create a 2x2 grid within the current subplot
#     for i in range(2):
#         for j in range(2):
#             # Calculate the position in the grid
#             plot_idx = i * 2 + j
            
#             # Create an axis for each grid position with small padding
#             ax = axs[subplot_idx].inset_axes([
#                 j*0.5 + 0.01,   # Add small horizontal padding
#                 1-(i+1)*0.5 + 0.01,  # Add small vertical padding
#                 0.48,  # Reduce width slightly to account for padding
#                 0.48   # Reduce height slightly to account for padding
#             ])
            
#             # Get the corresponding data array
#             data = data_arrays[subplot_idx * 4 + plot_idx]
            
#             # Plot the 2D array using imshow
#             im = ax.imshow(data, cmap='viridis', aspect='auto')
#             ax.axis('off')  # Hide axes

# # Remove all whitespace and borders
# plt.subplots_adjust(wspace=0.1, hspace=0, left=0, right=0.9, top=1, bottom=0)

# # Remove all axes and borders
# for ax in axs:
#     ax.axis('off')
# # plt.colorbar(im, cax=axs[-1])
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.set_label(r'\ln(1+E)' if use_log else r'Energy', fontsize=18)
# # plt.show()
# plt.savefig(f"{CWD}/storage/physics/phuftc/DDPM-HL-LHC/data/plots/bmap_comparison/comparison_log1p.png")


# from DDPMLHC.generate_plots.histograms_1d import *
# from itertools import product

# # plot_event_level_quantities_comparison(save_path=f"{CWD}/data/plots/event_level_quantities_comparison.png")
# # plot_particle_level_quantities_comparison(tt, pile_up, f"{CWD}/data/plots/particle_level_quantities_comparison.png")
# # plot_particle_level_quantities_ttbar_only(f"{CWD}/data/plots/particle_level_ttbar.png")
# # for x in product([16,32,64],["0.5", "001"]):
# #     bins  = x[0]
# #     beta = x[1]
# #     best_case_path = f"{CWD}/data/2-intermediate/noisy_mu0_event_level_grid{bins}.csv"
# #     noisy_path = f"{CWD}/data/2-intermediate/noisy_mu200_event_level_grid{bins}.csv"
# #     reconstructed_path = f"{CWD}/data/4-reconstruction/beta{beta}/reconstructed_mu{200}_event_level_from_grid{bins}_Unet{UNET_DIMS}.csv"
# #     mass_resolutions_best = {
# #         "Noisy, $\mu = 200$": load_variable_data(noisy_path, "mass", truth_path=best_case_path),
# #         "Denoised": load_variable_data(reconstructed_path, "mass", truth_path=best_case_path)
# #     }

# #     pt_resolutions_best = {
# #         "Noisy, $\mu = 200$": load_variable_data(noisy_path, "p_T", truth_path=best_case_path),
# #         "Denoised": load_variable_data(reconstructed_path, "p_T", truth_path=best_case_path)
# #     }

# #     plot_resolutions(
# #         mass_resolutions_best, pt_resolutions_best,
# #         colors={
# #             "Noisy, $\mu = 200$": "red",
# #             "Denoised": "blue"
# #         },
# #         save_path = f"{CWD}/data/plots/relative_resolutions/beta{beta}/resolution_grid{bins}_Unet{UNET_DIMS}_mass_gtBEST.pdf",
# #         show_subtit=True,
# #         mass_text="",
# #         pT_text=""
# #     )



# # # Resplots for impact of gridding, compare against cts pure
# NG = NoisyGenerator(tt, pu, mu=0)
# bins = [4, 8, 16, 256]
# save_path = f"{CWD}/data/plots/relative_resolutions/resolution_compare_grids_{'_'.join(map(str, bins))}.pdf"

# paths = [f"{CWD}/data/2-intermediate/noisy_mu0_event_level_grid{bin}.csv" for bin in bins]
# mass_resolutions_grids = { rf"$b={bin}$" : load_variable_data(paths[i], "mass")
#                           for i, bin in enumerate(bins) }
# pt_resolutions_grids = { rf"$b={bin}$" : load_variable_data(paths[i], "p_T")
#                           for i, bin in enumerate(bins) }

# colors={
#         f"$b={bins[0]}$": PLOT_COLOURS[1],
#         f"$b={bins[1]}$": PLOT_COLOURS[2],
#         f"$b={bins[2]}$": PLOT_COLOURS[3],
#         f"$b={bins[3]}$": PLOT_COLOURS[4],
#     }


# plot_resolutions(
#     mass_resolutions_grids, pt_resolutions_grids,
#     save_path = save_path,
#     mass_cutoff=(-1, 4),
#     pt_cutoff=(-0.1, 0.1),
#     use_log=True,
#     legend_title="",
#     fig_vinch=4.5,
#     colors=colors,
#     vert_line_colour="black",
# )

# files = [
#     f"{CWD}/data/2-intermediate/noisy_mu0_event_level.csv",
#     f"{CWD}/data/2-intermediate/noisy_mu{0}_event_level_grid{64}.csv",
#     f"{CWD}/data/2-intermediate/noisy_mu200_event_level.csv",
#     f"{CWD}/data/4-reconstruction/beta001/reconstructed_mu200_event_level_from_grid64_Unet64.csv",
# ]
# labels = ["Original", "Best case", "Noisy", "Denoised"]
# save_path = f"{CWD}/data/plots/1D_histograms/overlaid_from_model/cx_overlaid_poster.pdf"
# create_overlay_plots_general(files, labels, mass_max=350, save_path=save_path)

output_path = f"{CWD}/data/plots/bmap_comparison/"

plot_mu_comparison(tt, pile_up, 
                  use_log=True,
                  save_path=f"{output_path}/mu_comparison_b{16}_log_cx.pdf")


