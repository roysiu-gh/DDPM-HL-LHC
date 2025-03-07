# %%
import torch
from torch import optim
from torch.utils.data import Subset, Dataset, DataLoader, IterableDataset, TensorDataset
from torchvision.utils import make_grid
import torchvision.transforms as T
import torch.nn.functional as F
# from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm import tqdm
from datetime import datetime
from torch.amp import autocast
from denoising_diffusion_pytorch import Unet
from einops import rearrange, reduce, repeat
from ema_pytorch import EMA
from scipy.optimize import linear_sum_assignment
from accelerate import Accelerator
import math
import glob
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
import os
import re
from typing import Literal
from PIL import Image
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
# from 

# def show_tensor_images(tensor_images, scale_factor=8):
#     to_pil = T.ToPILImage()
#     pil_images = [to_pil(image) for image in tensor_images]

#     for img in pil_images:
#         # Upscale the image
#         upscaled_img = img.resize(
#             (img.width * scale_factor, img.height * scale_factor), 
#             Image.NEAREST  # or Image.BOX for smoother results
#         )
#         display(upscaled_img)

def constant_beta_schedule(beta_end, num_diffusion_timesteps=TIMESTEPS):
    # betas = beta_end * torch.ones([1,num_diffusion_timesteps], dtype=np.float64)
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    return betas

class PUDiffusion(GaussianDiffusion):
    def __init__(self, model, image_size, timesteps, puNG: NoisyGenerator, jet_ng: NoisyGenerator, mu=200, **kwargs):
        super(PUDiffusion, self).__init__(model=model, image_size=image_size, timesteps=timesteps, **kwargs)
        self.puNG = puNG
        self.jetNG = jet_ng
        self.channels = model.channels
        self.mu_counter = 1
        self.timesteps = timesteps
        self.mu = mu
        
        self.begin_sample = 0
        # Override beta scheduler with constant scheduler
        betas = constant_beta_schedule(beta_end=1)
        # betas = constant_beta_schedule(beta_end=0.0102)
        betas = torch.from_numpy(betas)
        betas = betas.to(self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

    def reset_sample(self):
        self.jetNG.reset()
        self.begin_sample = 0
    #############################################################################################
    def cond_noise(self, x_shape, noise, t):
        return self.pu_to_tensor(x_shape, t=t).to(self.device) if noise is None else noise
        # return torch.zeros_like(x_start) if noise is None else noise
    def generate_data(self, shape, NG: NoisyGenerator):
        """
        This function generates image data matched to the correct shape
        """
        # Start next jet
        next(NG)
        selected = NG.get_grid()
        # If empty pile-up, return array of 0s instead since model should account for this
        if selected.size == 0:
            return  "Error in PUDiffusion.generate_jet"
        # print(selected_pu.shape)
        pu_tensor = torch.from_numpy(selected).float()

        pu_tensor = torch.unsqueeze(pu_tensor,0)
        # This tensor has dimensions BxCxHxW to match x_start
        pu_tensor = torch.unsqueeze(pu_tensor,0)
        pu_tensor = pu_tensor.expand(shape[0], shape[1], -1, -1) 
        # pu_tensor = torch.zeros(shape)
        # pu_tensor = pu_tensor.to(self.device)
        return pu_tensor
    # @torch.inference_mode()
    def pu_to_tensor(self, shape, t):
        # Select random number of pile-ups (mu) to generate, max 200 for now since HL-LHC expected to do up to this
        # We are doing it per batch
        # Align jetIDs for correct centering of pile-up
        self.puNG._next_jetID = self.jetNG._next_jetID
        NG = self.puNG
        # Ensures if t is array valued, select the value
        NG.mu = int(t[0]) if isinstance(t, (torch.Tensor)) else int(t)
        # NG.reset()
        # next(self.puNG)
        pu_tensor = self.generate_data(shape=shape, NG=NG)
        return pu_tensor
    def jet_to_tensor(self, shape):
        NG = self.jetNG
        # Align jetIDs for correct centering of pile-up
        self.puNG._next_jetID = self.jetNG._next_jetID
        # next(NG)
        pu_tensor = self.generate_data(shape=shape, NG=self.jetNG)
        return pu_tensor
    
    #############################################################################################
    
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        # print("batched times", t)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        ######## MODIFY
        noise = self.pu_to_tensor(x.shape, t=t).to(self.device) if t > 0 else 0 # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = self.cond_noise(x_shape=x_start.shape, noise=noise, t=t)

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]
        # print("q_sample t", t)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    @torch.inference_mode()
    def generate_noise(self,shape):
        batch, device = shape[0], self.device
        jets = []
        self.puNG.mu = self.mu
        end_sample = min(self.begin_sample + batch, self.jetNG._max_TT_no)
        if end_sample > self.jetNG._max_TT_no:
            end_sample = self.jetNG._max_TT_no
        for i in range(self.begin_sample, end_sample):
            # random_jet_no = np.random.randint(low=0, high=self.jetNG._max_TT_no, size=None)
            self.jetNG._next_jetID = i
            self.jetNG.select_jet(i)  # or however you select jets
            jet = torch.from_numpy(self.jetNG.get_grid()).unsqueeze(0)
            # Now to add pile-up
            # random_pu_no = np.random.randint(low=0, high=self.jetNG._max_TT_no, size=None)
            self.puNG._next_jetID = i
            # Start from 200 pileups
            # Generate them
            next(self.puNG)
            selected_pu = self.puNG.get_grid()
            pu_tensor = torch.from_numpy(selected_pu)
            pu_tensor = torch.unsqueeze(pu_tensor,0)
            noised_jet = jet + pu_tensor # add energies element wise for each bin
            noised_jet =noised_jet.float()
            jets.append(noised_jet)
        self.begin_sample = end_sample
        # Should now  be batch x 1 x grid x grid
        jets = torch.stack(jets)
        jets = jets.to(self.device)
        return jets

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device
        img = self.generate_noise(shape)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)  # Returns intermediate imgs?

        ret = self.unnormalize(ret)
        print("final timestep: ", self.num_timesteps)
        return ret
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)


    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape
        noise = self.cond_noise(x_start.shape, noise=noise, t=t)
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    def forward(self, img, *args, **kwargs):
        # img = img.squeeze(0)
        # print("???", *img.shape)
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


# %%
# Base code for training generated by Claude 3.5. Since modified for our purposes.
# Changes from originally-generated response:
#  - Tracking of loss per epoch
# - Checkpoint saved in intervals rather than every epoch
# - Changed handling of range and loading when 0 epochs
## Custom reimplementation of Trainer from DDPM
## Avoid subclassing because we do not want to pass in literal files

def load_and_train(
    diffusion,
    dataloader,
    num_epochs,
    device,
    save_dir,
    lr=2e-4
):
    os.makedirs(save_dir, exist_ok=True)
    loss_array = []
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

    optimizer = optim.Adam(diffusion.parameters(), lr=lr)
    epoch_range = range(last_epoch, last_epoch + num_epochs)
    final_epoch = list(epoch_range)[-1] if len(list(epoch_range))> 0 else last_epoch

    for epoch in epoch_range:
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
            progress_bar.set_postfix({'Loss': f'{avg_loss:.10f}'})
        loss_array.append(avg_loss)
        # Save checkpoint at the end of each epoch
        if epoch % 50 == 0 or epoch == final_epoch:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_loss_{avg_loss:.10f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    return loss_array


## Plot losses from available checkpoints

def get_losses_from_checkpoints(checkpoint_dir='./data/ML/second'):
    # Get absolute path to checkpoint directory
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*_loss_*.pth')
    
    # Get all matching files
    files = glob.glob(checkpoint_pattern)
    losses = []
    pattern = r'loss_([\d.]+)\.pth'
    
    for file in files:
        match = re.search(pattern, file)
        if match:
            loss = float(match.group(1))
            losses.append(loss)
    
    return len(files), losses

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
NG_jet = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=0)
# Second one to randomly generate and return pile-up events ONLY
NG_pu = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=0, pu_only=True)
# Default NG just for convenience
NG_default = NoisyGenerator(TTselector=tt, PUselector=pu, bins=BMAP_SQUARE_SIDE_LENGTH, mu=200)
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
    image_size = BMAP_SQUARE_SIDE_LENGTH, 
    timesteps = TIMESTEPS,  # Number of diffusion steps
    objective = "pred_x0",
).to(device)

ng_for_dataloader = NGenForDataloader(NG_jet)
dataloader = DataLoader(ng_for_dataloader, batch_size=BATCH_SIZE, num_workers=2, shuffle = True, pin_memory = True)


def print_params(mode: Literal["TRAINING", "SAMPLING"],num_epochs=EPOCHS, mu=200):
    print("#############################")
    print("DIAGNOSTIC PARAMETERS")
    print("#############################")
    print(F"MODE: {mode}")
    print(f"Training Batch Size: {BATCH_SIZE}")
    print(f"mu: {mu}")
    print(f"Image Size/bins: {BMAP_SQUARE_SIDE_LENGTH}")
    print(f"UNET DIMS: {UNET_DIMS}")
    print(f"TOTAL EPOCHS: {num_epochs}")
    print(f"TOTAL DIFFUSION TIMESTEPS: {TIMESTEPS}")
    print(f"LEARNING RATE: {LR}")
    print(f"DEVICE: {device.type}")
    print("#############################")
    print("END DIAGNOSTIC PARAMETERS")
    print("#############################")

################
# Custom save_image function since 
@torch.no_grad()
def save_image_larger(
    tensor,
    fp,
    format="png",
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    width, height = im.size 
    scale_factor=4
    # im = im.resize((width*scale_factor, height*scale_factor))
    im.save(fp, format=format, dpi=(600,600))

