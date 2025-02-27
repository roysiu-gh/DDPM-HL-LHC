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
train_batch_size = 200
# Sampling: only load checkpoint
num_epochs = 0

print_params(mode="SAMPLING")
# this one is to be passed into DataLoader for training
ng_for_dataloader = NGenForDataloader(NG_jet)
dataloader = DataLoader(ng_for_dataloader, batch_size=train_batch_size, num_workers=2, shuffle = True, pin_memory = True)
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
    def __init__(self, diffusion, NG_jet, num_jets_to_process, bins=BMAP_SQUARE_SIDE_LENGTH):
        self.diffusion = diffusion
        self.NG_jet = NG_jet
        self.num_jets = num_jets_to_process
        self.bins = bins
    
    def _calculate_event_level(self):
        print(f"Generating {self.num_jets} jets...")
        
        sampled_images = diffusion.sample(batch_size=self.num_jets)
        # show_tensor_images(sampled_images[:5] * NG_jet.max_energy, scale_factor=10)
        # print(len(sampled_images))
        # print(sampled_images[0])
        # save_image(sampled_images[:4], f"{histogram_path}/saved_denoised_grids2.png")

        rescaled = sampled_images * self.NG_jet.max_energy
        tensor_to_data(rescaled[:48])

        # print(len(rescaled))
        # print(rescaled[0])
        rescaled = rescaled.cpu().numpy()  # PyTorch to numpy
        
        # Remove channel dimension if exists
        print(f"rescaled.shape {rescaled.shape}")
        if len(rescaled.shape) == 4:  # (batch, channel, height, width)
            rescaled = rescaled.squeeze(1)
        print(f"rescaled.shape {rescaled.shape}")
        
        combined = []
        for idx, grid in enumerate(rescaled):
            enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=self.bins)
            pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
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
            
            combined.append(np.copy(event_level))
                
        all_data = np.vstack(combined)
        del rescaled
        del sampled_images
        return all_data
    
    def save_event_level(self, output_folder=f"{CWD}/data/4-reconstruction", output_filename=None):
        if output_filename is None:
            output_filename = f"reconstructed_mu{self.diffusion.mu}_event_level_from_grid{self.bins}.csv"
        output_path = f"{output_folder}/{output_filename}"
        
        data = self._calculate_event_level()
        np.savetxt(
            output_path,
            data,
            delimiter=",",
            header="event_id,px,py,pz,eta,phi,mass,p_T",
            comments="",
            fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
        )
        
        return output_path

batch_size = 100
# Number of jets to sample - note very high memory requirement
jets_to_sample = 1000
# model_cpu = model.
with torch.inference_mode():
    model.eval()
    diffusion.eval()
    # sampled_images = diffusion.sample(batch_size=batch_size)
    # rescaled = sampled_images * NG_jet.max_energy
    # tensor_to_data(rescaled)
    output_folder=f"{CWD}/data/4-reconstruction"
    output_filename = f"reconstructed_mu{diffusion.mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"

    OD = OutData(diffusion, NG_jet, jets_to_sample)
    output_path = OD.save_event_level(output_folder=output_folder, output_filename=output_filename)

torch.cuda.empty_cache()
##### CODE FOR GENERATING MASS/ETA/PT PLOTS####
events_dat = np.genfromtxt(
        output_path, delimiter=",", encoding="utf-8", skip_header=1
    )
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
def generate_event_level_gridded_jets(NG: NoisyGenerator, save_dir=INTERMEDIATE_PATH):
    NG.reset()
    gt_file = f"{save_dir}/noisy_mu0_event_level_gridded.csv"
    combined = []
    for idx, _ in enumerate(NG):
        # next(NG)
        grid = NG.get_grid(normalise=False)
        enes, detas, dphis = grid_to_ene_deta_dphi(grid, N=NG.bins)
        pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
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
        
        combined.append(np.copy(event_level))
            
    all_data = np.vstack(combined)
    np.savetxt(
            gt_file,
            all_data,
            delimiter=",",
            header="event_id,px,py,pz,eta,phi,mass,p_T",
            comments="",
            fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
    )
    return all_data, gt_file
def mass_energy_diff(save_dir=INTERMEDIATE_PATH, mu=200):
    """
    Finds the relative difference between the model's denoised images and the binned jets as ground truths
    """
    gt_file = f"{save_dir}/noisy_mu0_event_level_gridded.csv"
    if not os.path.isfile(gt_file):
        generate_event_level_gridded_jets(NG_jet)

    jet_quantities = pl.read_csv(gt_file)
    jets_px = jet_quantities['px']
    jets_py = jet_quantities['py']
    jets_pz = jet_quantities['pz']
    jets_mass = jet_quantities['mass'].to_numpy()
# print(your[24717:])
    jets_mass = np.concatenate((jets_mass[0:24716], jets_mass[24717:]))
        
    jet_energy = (jets_px ** 2) + (jets_py ** 2) + (jets_pz ** 2)
    jet_energy = jet_energy.to_numpy()
    jet_energy = np.concatenate((jet_energy[0:24716], jet_energy[24717:]))
    jet_energy = np.sqrt(jet_energy)
    csv_file_paths = [f"./data/4-reconstruction/reconstructed_mu{mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"]
    fig,axs = plt.subplots(nrows=len(csv_file_paths),ncols=2, figsize=(8,6))
    #  = axs
    energy_counts = []
    mass_counts = []
    mean_mass_diffs = []
    std_mass_diffs = []
    data_array = [pl.read_csv(csv_file_path) for csv_file_path in csv_file_paths]

    for idx,data in enumerate(data_array):
        df = data
        max_id = df['event_id'].max() + 1
        # get jet indices
        # Will select from ground truth jets
        # jet_indices = df['event_id']
        px = df['px']
        py = df['py']
        pz = df['pz']
        # print(px)
        mass = df['mass']
        mass = mass.to_numpy()
        # mass = np.concatenate((mass[0:24716], mass[24717:]))
        # massless limit
        energy = (px ** 2) + (py ** 2) + (pz ** 2)
        energy = energy.to_numpy()
        # energy = np.concatenate((energy[0:24716], energy[24717:]))

        energy = np.sqrt(energy)
        # print(energy)
        # Find energy difference between jet+pile-up and jet for feach jet_id
        jet_energy1 = jet_energy[:max_id]
        jets_mass1 = jets_mass[:max_id]
        energy_diffs = energy - jet_energy1
        energy_diffs = energy_diffs / jet_energy1
        mass_diffs = mass - jets_mass1
        mass_diffs2 = mass_diffs/ jets_mass1
        # print(mass_diffs2)
        # mean_energy_diff = np.sum(energy_diffs) / (max_id - 1)
        # en_bins = np.mgrid[np.min(energy_diffs):np.max(energy_diffs):(len(energy_diffs)+1)*1j]
        # mass_bins = np.mgrid[np.min(mass_diffs2):np.max(mass_diffs2):(len(mass_diffs2)+1)*1j]
        # mass_bins = np.mgrid[0:mass_max:(mass_num_bins+1)*1j]
        axs[0].hist(energy_diffs, bins = 50, label=f"$\\mu = {mu}$", edgecolor="black")
        axs[1].hist(mass_diffs2[mass_diffs2<5], bins = 50,label=f"$\\mu = {mu}$", edgecolor="black")
        axs[0].set_ylabel(r"Counts")
        # axs[idx][0].set_ylabel(r"Counts")

        axs[0].legend(prop={'size': 14})
        axs[1].legend(prop={'size': 14})
        # std_energy_diff = np.std(energy_diffs)
        # print(std_energy_diff)
        # print(mean_energy_diff)

        # mean_mass_diff = np.sum(mass_diffs2) / (max_id - 1)
        # std_mass_diff = np.std(mass_diffs2)

        # mean_energy_diffs.append(mean_energy_diff)
        # std_energy_diffs.append(std_energy_diff)
        # mean_mass_diffs.append(mean_mass_diff)
        # std_mass_diffs.append(std_mass_diff)
    axs[0].set_xlabel(r"$\frac{E_{\mu}^{j} - E_{0}^{j}}{E_{0}^{j}}$")
    axs[1].set_xlabel(r"$\frac{m_{\mu}^{j} - m_{0}^{j}}{m_{0}^{j}}$")
    plt.tight_layout()
    plt.savefig(f"{CWD}/data/plots/hist_energymasscounts_model.pdf", format="pdf")
    # plt.savefig(f"{CWD}/data/plots/hist_energymasscounts_model.png", format="png", dpi=600)
    plt.close()
mass_energy_diff()