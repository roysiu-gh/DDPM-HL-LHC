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
from DDPMLHC.model_utils import *

# Print Diagnostics right before training
mu = 200
train_batch_size = 200
num_epochs = 500

print_params(mode="TRAINING")
# this one is to be passed into DataLoader for training
save_dir = f"{CWD}/data/ML/Unet{UNET_DIMS}_bins{BMAP_SQUARE_SIDE_LENGTH}_mu{mu}"
print("Begin training")
xd = load_and_train(diffusion, dataloader, num_epochs=num_epochs, device=device, save_dir=save_dir)
print("Finished training")


