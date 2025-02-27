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
from DDPMLHC.model_utils import load_and_train

# DATA LOADING
MAX_DATA_ROWS = None
bins=BMAP_SQUARE_SIDE_LENGTH

# === Read in data
print("0 :: Loading original data")
tt = np.genfromtxt(
    TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
pu = np.genfromtxt(
    PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
)
tt = EventSelector(tt)
pu = EventSelector(pu)
print("FINISHED loading data\n")

# Ground truth ttbar jets
NG_jet = NoisyGenerator(TTselector=tt, PUselector=pu, bins=bins, mu=0)
# Second one to randomly generate and return pile-up events ONLY
NG_pu = NoisyGenerator(TTselector=tt, PUselector=pu, bins=bins, mu=0, pu_only=True)

model = Unet(
    dim=UNET_DIMS,                  # Base dimensionality of feature maps
    dim_mults=(1, 2, 4, 8),  # Multipliers for feature dimensions at each level
    channels=1,              # E.g. 3 for RGB
).to(device)

# 
diffusion = PUDiffusion(
    model = model,
    puNG = NG_pu,
    jet_ng= NG_jet,
    image_size = bins,  # Size of your images (ensure your images are square)
    timesteps = 200,  # Number of diffusion steps
    objective = "pred_x0",
    sampling_timesteps = None
).to(device)

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

print("#############################")
print("DIAGNOSTIC PARAMETERS")
print("#############################")
print("MODE: SAMPLING")
print(f"Training Batch Size: {train_batch_size}")
print(f"mu: {mu}")
print(f"Image Size/bins: {BMAP_SQUARE_SIDE_LENGTH}")
print(f"UNET DIMS: {UNET_DIMS}")
print(f"TOTAL EPOCHS: {num_epochs}")
print(f"TOTAL DIFFUSION TIMESTEPS: {diffusion.timesteps}")
print(f"DEVICE: {device.type}")
print("#############################")
print("END DIAGNOSTIC PARAMETERS")
print("#############################")

# this one is to be passed into DataLoader for training
ng_for_dataloader = NGenForDataloader(NG_jet)
dataloader = DataLoader(ng_for_dataloader, batch_size=train_batch_size, num_workers=2, shuffle = True, pin_memory = True)
save_dir = f"{CWD}/data/ML/Unet{UNET_DIMS}_bins{bins}_mu{mu}"

print("Begin training")
xd = load_and_train(diffusion, dataloader, num_epochs=num_epochs, device=device, save_dir=save_dir, lr=1e-5)
print("Finished training")

# %%
# Set model to evaluation mode
# model_cpu = model.to(torch.device("cpu"))
# diffusion_cpu = diffusion.to(torch.device("cpu"))

# model_cpu.eval()
# diffusion_cpu.eval()

# # %%
# MPL_GLOBAL_PARAMS = {
#     'text.usetex' : False, # use latex text
#     'text.latex.preamble' : r'\usepackage{type1cm}\usepackage{braket}\usepackage{amssymb}\usepackage{amsmath}\usepackage{txfonts}', # latex packages
#     'font.size' : 24,
#     'figure.dpi' : 600,
#     'figure.figsize' : (4, 3),
#     'figure.autolayout' : True, # tight layout (True) or not (False)
#     'axes.labelpad' : 5,
#     'axes.xmargin' : 0,
#     'axes.ymargin' : 0,
#     'axes.grid' : False,
#     'axes.autolimit_mode' : 'round_numbers', # set axis limits by rounding min/max values
#     # 'axes.autolimit_mode' : 'data', # set axis limits as min/max values
#     'xtick.major.pad' : 10,
#     'ytick.major.pad' : 10,
#     'xtick.labelsize': label_fontsize,
#     'ytick.labelsize': label_fontsize,
#     'lines.linewidth' : 1.3,
#     'xtick.direction' : 'in',
#     'ytick.direction' : 'in',
#     'xtick.top' : True,
#     'ytick.right' : True,
#     'xtick.minor.visible' : True,
#     'ytick.minor.visible' : True,
#     'axes.prop_cycle': cycler(color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*3),
#     'legend.framealpha': None
#   }
# mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# import glob
# import re
# import os

# def get_losses_from_checkpoints(checkpoint_dir='./data/ML/second'):
#     # Get absolute path to checkpoint directory
#     print(os.listdir("."))
#     checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*_loss_*.pth')
    
#     # Get all matching files
#     files = glob.glob(checkpoint_pattern)
#     print(files)
#     losses = []
#     pattern = r'loss_([\d.]+)\.pth'
    
#     for file in files:
#         match = re.search(pattern, file)
#         if match:
#             loss = float(match.group(1))
#             losses.append(loss)
    
#     return len(files), losses

# # Usage:
# num_files, losses = get_losses_from_checkpoints()  
# print(f"Number of checkpoint files: {num_files}")
# print(f"Losses: {losses}")

# fig = plt.figure()
# # num,losses = get_losses_from_checkpoints()
# plt.plot(list(range(num_files)),losses)
# plt.show()

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
# model_cpu = model.
with torch.inference_mode():
    model.eval()
    diffusion.eval()
    # sampled_images = diffusion.sample(batch_size=batch_size)
    # rescaled = sampled_images * NG_jet.max_energy
    # tensor_to_data(rescaled)
    output_folder=f"{CWD}/data/4-reconstruction"
    output_filename = f"reconstructed_mu{diffusion.mu}_event_level_from_grid{BMAP_SQUARE_SIDE_LENGTH}.csv"

    OD = OutData(diffusion, NG_jet, 2000)
    output_path = OD.save_event_level(output_folder=output_folder, output_filename=output_filename)

torch.cuda.empty_cache()
events_dat = np.genfromtxt(
        output_path, delimiter=",", encoding="utf-8", skip_header=1
    )
mass_num_bins = 50
mass_max = 400
pT_max = 5000
pT_num_bins = 50
pT_bins = np.mgrid[0:pT_max:(pT_num_bins+1)*1j]

mass_bins = np.mgrid[0:mass_max:(mass_num_bins+1)*1j]
fig, axs = plt.subplots(1,3,figsize=(14,6))
axs[0].hist(events_dat[:,6], bins=mass_bins, density=True)
axs[1].hist(events_dat[:,4], bins=50,density=True)
# axs[2].hist(events_dat[:,5], bins=50,density=True)
axs[2].hist(events_dat[:,7], bins = pT_bins, density=True)
plt.savefig(f"{CWD}/data/3-grid/grid{bins}/test.pdf")
plt.close()

# # rescaled_np = rescaled.numpy()

# # Move model to CPU after training

# # # print(sampled_images)
# # show_tensor_images(rescaled[0], scale_factor=10)




# # %%
# # Data post-processing
# mu = 200

# # generator = NoisyGenerator(tt, pile_up, mu=mu)
# combined = []
# def grid_to_ene_deta_dphi(grid, N=BMAP_SQUARE_SIDE_LENGTH):
#     enes = np.zeros(N*N)
#     detas = np.zeros(N*N)
#     dphis = np.zeros(N*N)
#     # xbin and ybin may be wrong way around
#     for xbin in range(N):
#         for ybin in range(N):
#             idx = xbin*N + ybin
#             deta = 2*xbin/N - 1
#             dphi = 2*ybin/N - 1
#             enes[idx] = grid[xbin, ybin]
#             detas[idx] = deta
#             dphis[idx] = dphi
#     return enes, detas, dphis
#     for idx,grid in enumerate(tensor_images_cpu):
#         # Each grid is 1 x bins x bins
#         hxW = grid[0] # Selects bins x bins
#         enes, detas, dphis = grid_to_ene_deta_dphi(hxW, N=bins)
#         pxs, pys, pzs = deta_dphi_to_momenta(enes, detas, dphis)
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

#     # Final check before saving
#     if np.any(np.isnan(all_data)):
#         print("\nWarning: NaN values in final data:")
#         print(f"Total NaN count: {np.sum(np.isnan(all_data))}")
#         print("NaN locations (row, column):")
#         nan_rows, nan_cols = np.where(np.isnan(all_data))
#         column_names = ['event_id', 'px', 'py', 'pz', 'eta', 'phi', 'mass', 'p_T']
#         for row, col in zip(nan_rows, nan_cols):
#             print(f"Row {row}, Column {column_names[col]}")
    

#     np.savetxt(
#             output_filepath,
#             all_data,
#             delimiter=",",
#             header="event_id,px,py,pz,eta,phi,mass,p_T",
#             comments="",
#             fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
#         )
    
#     # Here mu is the initial mu we set simulations to start from
#     plot_1d_histograms(mu, event_stats_path=output_filepath, output_path=f"{output_path}/grid{bins}_hist")

#     # Tensors are current;y in BxCxHxW tensor
#     # Want to ex
# # for idx, _ in enumerate(generator):
# #     grid = generator.get_grid()
    
    
# # rescaled = sampled_images * NG_jet.max_energy
# # np.savetxt(f"tensor_data_denoised.txt", rescaled_np,delimiter=",")

# tensor_to_data(rescaled)

