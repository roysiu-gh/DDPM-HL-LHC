# Package imports
import numpy as np
import matplotlib as mpl
# Local imports
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *

mpl.rcParams.update(MPL_GLOBAL_PARAMS)

# MAX_DATA_ROWS = None

# # === Read in data
# print("0 :: Loading original data")
# pile_up = np.genfromtxt(
#     PILEUP_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# tt = np.genfromtxt(
#     TT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=10*MAX_DATA_ROWS
# )
# print("FINISHED loading data\n")

# tt = EventSelector(tt)
# pu = EventSelector(pile_up)

# NG = NoisyGenerator(TTselector=tt, PUselector=pu, bins=16)
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)
def load_and_train(
    diffusion,
    dataloader,
    num_epochs,
    device,
    save_dir,
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get last epoch number
    checkpoint_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
    last_epoch = 0
    if checkpoint_files:
        epoch_numbers = []
        for f in checkpoint_files:
            try:
                epoch_num = int(f.split('epoch_')[1].split('_loss')[0])
                epoch_numbers.append(epoch_num)
            except:
                continue
        last_epoch = max(epoch_numbers) if epoch_numbers else 0

    # Load checkpoint if exists
    if last_epoch > 0:
        checkpoint_pattern = os.path.join(save_dir, f'checkpoint_epoch_{last_epoch}_*.pth')
        checkpoint_file = glob.glob(checkpoint_pattern)[0]
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Starting fresh training")

    optimizer = optim.Adam(diffusion.parameters(), lr=1e-4)

    for epoch in range(last_epoch, last_epoch + num_epochs):
        print(f"\nEpoch {epoch + 1}/{last_epoch + num_epochs}")
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        running_loss = 0.0
        for i, images in progress_bar:
            images = images.to(device)
            optimizer.zero_grad()
            loss = diffusion(images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})

        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
# trainer = Trainer(
#     diffusion,
#     'path/to/your/images',
#     train_batch_size = 32,
#     train_lr = 8e-5,
#     train_num_steps = 700000,         # total training steps
#     gradient_accumulate_every = 2,    # gradient accumulation steps
#     ema_decay = 0.995,                # exponential moving average decay
#     amp = True,                       # turn on mixed precision
#     calculate_fid = True              # whether to calculate fid during training
# )

# trainer.train()
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import torch
from torch import optim
from torch.utils.data import Subset, Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# req torch, torchvision, einops, tqdm, ema_pytorch, accelerate
from IPython.display import display
from tqdm import tqdm

import glob
model = Unet1D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 128,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = torch.rand(64, 32, 128) # features are normalized from 0 to 1

loss = diffusion(training_seq)
loss.backward()
#################################################################################

# print("4 :: Drawing overlaid histograms with varying mu")
# print("Loading intermediate data...")
# tt = np.genfromtxt(
#     TT_EXT_PATH, delimiter=",", encoding="utf-8", skip_header=1, max_rows=MAX_DATA_ROWS
# )
# px = tt[:, 1]
# py = tt[:, 2]
# pz = tt[:, 3]
# eta = tt[:, 4]
# p_T = tt[:, 6]
# p = p_magnitude(px, py, pz)
# OUT_PATH_1D_HIST = f"{CWD}/data/plots/1D_histograms/particles/"
# hist_data_particles = [
#     {
#         "name": "Momentum $p$ [GeV]",
#         "data": p,
#         "plot_params": {"xlog": True},
#         "save_filename": "p",
#         "save_path": OUT_PATH_1D_HIST,
#     },
#     {
#         "name": "Pseudorapidity $\eta$",
#         "data": eta,
#         "plot_params": {},
#         "save_filename": "eta",
#         "save_path": OUT_PATH_1D_HIST,
#     },
#     {
#         "name": "Transverse Momentum $p_T$ [GeV]",
#         "data": p_T,
#         "plot_params": {"xlog": True},
#         "save_filename": "pT",
#         "save_path": OUT_PATH_1D_HIST,
#     },
# ]
# print("FINISHED drawing overlaid histograms\n")

print("DONE ALL.")
