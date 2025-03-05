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
from denoising_diffusion_pytorch import Unet
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
import polars as pl
import os
import gc
CWD = os.getcwd()

# Device stuff
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Remember to use {device} device from here on")
# print(os.chdir("../"))
# %cd /home/physics/phuqza/E9/DDPM-HL-LHC/
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *
from DDPMLHC.generate_plots.overlaid_1d import *
from DDPMLHC.generate_plots.bmap import *
from DDPMLHC.generate_plots.histograms_1d import *
from DDPMLHC.model_utils import *

# DATA LOADING
MAX_DATA_ROWS = None
bins=BMAP_SQUARE_SIDE_LENGTH
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

print_params(mode="SAMPLING")
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

save_dir = f"{CWD}/data/ML/Unet{UNET_DIMS}_bins{bins}_mu{mu}"

print("Begin training")
xd = load_and_train(diffusion, dataloader, num_epochs=0, device=device, save_dir=save_dir)
print("Finished training")

# # %%
# # Generate samples
# # NG_jet.reset()
# # NG_pu.reset()
# # sampled_images = diffusion.sample(batch_size=100)
output_path = f"{CWD}/data/3-grid/Unet{UNET_DIMS}_bins{bins}_mu{mu}"
output_filename = f"noisy_mu{mu}_event_level_from_grid{bins}.csv"
output_filepath = f"{output_path}/{output_filename}"
histogram_path = f"{output_path}/grid{bins}_hist"
# mpl.rcParams.update(MPL_GLOBAL_PARAMS)
if not(os.path.exists(output_path)):
    os.mkdir(output_path)
if not(os.path.exists(histogram_path)):
    os.mkdir(histogram_path)
    
def tensor_to_data(tensor_images):
    # tensor_images_cpu = tensor_images.detach().cpu().numpy()
    save_image(tensor_images, f"{histogram_path}/saved_denoised_grids_new.png")


class OutData():
    def __init__(self, diffusion, NG_jet, num_jets_to_process, bins=BMAP_SQUARE_SIDE_LENGTH, num_saved=4):
        self.diffusion = diffusion
        self.NG_jet = NG_jet
        self.num_jets = num_jets_to_process
        self.bins = bins
        self.num_saved_row =  int(np.sqrt(num_saved))
        #self.diffusion.begin_sample = 0 # ensure starting from first jet for sampling
        self.diffusion.reset_sample()
    @torch.inference_mode()
    def _batch_sample(self, rescale=False):
        #self.diffusion.reset()
        #self.diffusion.reset_sample()
        while self.diffusion.begin_sample < self.NG_jet._max_TT_no - 1:
            try:
              sampled_images = self.diffusion.sample(batch_size=SAMPLE_BATCH)
              sampled_images = sampled_images * self.NG_jet.max_energy
              if rescale:
                  sampled_images = torch.log1p(sampled_images)
              sampled_images = sampled_images.detach().cpu()            
              yield sampled_images
              del sampled_images
              gc.collect()
              torch.cuda.empty_cache()
              torch.cuda.synchronize()
            except StopIteration:
                break
        return None
    def _calculate_event_level(self):
        print(f"Iterating through dataset, adding noise and letting model denoise...")
        counter = 0
        # sampled_images is a generator because of yield
        # So each "element" in generator is a sample of jets
        self.diffusion.reset_sample()
        for sampled_images in self._batch_sample(rescale=False):
            all_data = []

            if sampled_images is None:
                break
        # rescaled = sampled_images
        # Save first 4 jets only
        
        # Remove channel dimension if exists
            if counter ==0:
                sampled = sampled_images[:4]
                sampled_scaled = torch.log1p(sampled)
                save_image(tensor=sampled_scaled, nrow=self.num_saved_row,fp=f"{histogram_path}/saved_denoised_grids{self.num_saved_row}.png", normalize=True)
            if len(sampled_images.shape) == 4:  # (batch, channel, height, width)
                sampled_images = sampled_images.squeeze(1)

            counter +=1
            # print(f"rescaled.shape {rescaled.shape}")
            
            # combined = []
            for idx, grid in enumerate(sampled_images):
                NG_jet.select_jet(idx)
                axis = NG_jet.jet_axis
                enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=self.bins)
                detas, dphis = decentre(axis, detas, dphis)
                pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
                # print("OutData eventlevel")
                event_quantities = particle_momenta_to_event_level(enes, pxs, pys, pzs)
                event_mass, event_px, event_py, event_pz, event_eta, event_phi, event_pT = event_quantities
                
                event_level = np.array([
                    idx,
                    event_px,
                    event_py,
                    event_pz,
                    event_eta,
                    event_phi,
                    event_mass,
                    event_pT,
                ])
                
                # combined.append(np.copy(event_level))
                all_data.append(np.copy(event_level)) 
           
            all_data = np.vstack(all_data)
            yield all_data

        print(len(all_data))
        del sampled_images
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return None
    
    def save_event_level(self, output_folder=f"{CWD}/data/4-reconstruction", output_filename=None):
        if output_filename is None:
            output_filename = f"reconstructed_mu{self.diffusion.mu}_event_level_from_grid{self.bins}_Unet{UNET_DIMS}.csv"
        output_path = f"{output_folder}/{output_filename}"
        data = self._calculate_event_level()
        with open(output_path, 'w') as f:
            f.write("event_id,px,py,pz,eta,phi,mass,p_T\n")
            f.close()
        total_events = 0
        with open(output_path, 'a+') as f:
            for batch_data in data:
                if batch_data is None:
                    break
                np.savetxt(f, batch_data,
                            delimiter=",",
                            fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f")
            f.close()
        print("Done writing")

        # total_events += len(batch_data)

        # print(f"Processed {total_events} events so far...")

        # print("writing")
        # np.savetxt(
        #     output_path,
        #     data,
        #     delimiter=",",
        #     header="event_id,px,py,pz,eta,phi,mass,p_T",
        #     comments="",
        #     fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
        # )
        
        return output_path

# Number of jets to sample - note very high memory requirement
# model_cpu = model.
with torch.inference_mode():
    model.eval()
    diffusion.eval()
    # sampled_images = diffusion.sample(batch_size=batch_size)
    # rescaled = sampled_images * NG_jet.max_energy
    # tensor_to_data(rescaled)
    output_folder=f"{CWD}/data/4-reconstruction"
    output_filename = f"reconstructed_mu{diffusion.mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}_Unet{UNET_DIMS}.csv"

    OD = OutData(diffusion, NG_jet, jets_to_sample)
    output_path = OD.save_event_level(output_folder=output_folder, output_filename=output_filename)

torch.cuda.empty_cache()
print("DONE")
##### CODE FOR GENERATING MASS/ETA/PT PLOTS####
# events_dat = np.genfromtxt(
#         output_path, delimiter=",", encoding="utf-8", skip_header=1
#     )
# mass_num_bins = 50
# mass_max = 400
# pT_max = 5000
# pT_num_bins = 50
# pT_bins = np.mgrid[0:pT_max:(pT_num_bins+1)*1j]
# mass_bins = np.mgrid[0:mass_max:(mass_num_bins+1)*1j]
# fig, axs = plt.subplots(1,3,figsize=(14,6))
# axs[0].hist(events_dat[:,6], bins=mass_bins, density=True)
# axs[1].hist(events_dat[:,4], bins=50,density=True)
# # axs[2].hist(events_dat[:,5], bins=50,density=True)
# axs[2].hist(events_dat[:,7], bins = pT_bins, density=True)
# plt.savefig(f"{CWD}/data/3-grid/grid{bins}/test.pdf")
# plt.close()

##### CODE TO GENERATE RESOLUTION PLOTS #####
# def generate_event_level_gridded_jets(NG: NoisyGenerator, save_dir=INTERMEDIATE_PATH):
#     NG.reset()
#     NG.bins = BMAP_SQUARE_SIDE_LENGTH
#     gt_file = f"{save_dir}/noisy_mu0_event_level_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"
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
