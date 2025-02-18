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
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# req torch, torchvision, einops, tqdm, ema_pytorch, accelerate
from IPython.display import display
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
# %cd /home/physics/phuqza/E9/DDPM-HL-LHC/
from DDPMLHC.config import *
from DDPMLHC.calculate_quantities import *
from DDPMLHC.data_loading import *
from DDPMLHC.generate_plots.overlaid_1d import create_overlay_plots
from DDPMLHC.generate_plots.bmap import save_to_bmap
from DDPMLHC.generate_plots.histograms_1d import *
# from 


# Some functions from denoising_diffusion_pytorch that are required but couldn't import
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


CWD = os.getcwd()

# Device stuff
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Remember to use {device} device from here on")
print(os.chdir("../"))

# Training loop
def load_and_train(
    diffusion,
    dataloader,
    num_epochs,
    device,
    save_dir,
    lr=1e-4
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
            progress_bar.set_postfix({'Loss': f'{avg_loss:.7f}'})
        loss_array.append(avg_loss)
        # Save checkpoint at the end of each epoch
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_loss_{avg_loss:.7f}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
    # Return array of final losses for plotting
    return loss_array


MAX_DATA_ROWS = None

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
bins=32
# Ground truth ttbar jets
NG_jet = NoisyGenerator(TTselector=tt, PUselector=pu, bins=bins, mu=0)
# Second one to randomly generate and return pile-up events ONLY
## Will use np.random.randint to generate NoisyGenerator.mu and then call next
NG_pu = NoisyGenerator(TTselector=tt, PUselector=pu, bins=bins, mu=0, pu_only=True)

class NGenForDataloader(Dataset):
    def __init__(self, noisy_generator, njets=100):
        self.ng = noisy_generator
        self.jets = []
        self.njets = njets
        # next(self.ng)
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.ng._max_TT_no - 1
    
    def __getitem__(self, idx):
        self.ng.select_jet(idx)
        x = torch.from_numpy( self.ng.get_grid() )
        # x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        # self.jets.append(x)
        # print("x jet img", x.shape)
        # y = x
        return x
        #
model = Unet(
    dim=128,                  # Base dimensionality of feature maps
    dim_mults=(1, 2, 4, 8),  # Multipliers for feature dimensions at each level
    channels=1,              # E.g. 3 for RGB
).to(device)

class PUDiffusion(GaussianDiffusion):
    def __init__(self, model, image_size, timesteps, puNG: NoisyGenerator, jet_ng: NoisyGenerator, **kwargs):
        super(PUDiffusion, self).__init__(model=model, image_size=image_size, timesteps=timesteps, **kwargs)
        self.puNG = puNG
        self.jetNG = jet_ng
        self.channels = model.channels
        self.mu_counter = 1
        self.timesteps = timesteps
    def cond_noise(self, x_shape, noise):
        return self.pu_to_tensor(x_shape) if noise is None else noise
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
        pu_tensor = torch.from_numpy(selected)

        pu_tensor = torch.unsqueeze(pu_tensor,0)
        # This tensor has dimensions BxCxHxW to match x_start
        pu_tensor = torch.unsqueeze(pu_tensor,0)
        pu_tensor = pu_tensor.expand(shape[0], shape[1], -1, -1) 
        # pu_tensor = torch.zeros(shape)
        pu_tensor = pu_tensor.to(self.device)
        return pu_tensor
    def pu_to_tensor(self, shape):
        # Select random number of pile-ups (mu) to generate, max 200 for now since HL-LHC expected to do up to this
        # We are doing it per batch
        # if isinstance(t, int):
        #     mu = np.random.randint(low=1, high=200, size=None)
        # else:
        #     mu = np.random.randint(low=1, high=200, size=None)
        mu = 1
        # print(mu)
        # Align jetIDs for correct centering of pile-up
        self.puNG._next_jetID = self.jetNG._next_jetID
        NG = self.puNG
        NG.mu = mu
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
    # TODO: ddim_sample???
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        # print("batched times", t)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        ######## MODIFY
        noise = self.pu_to_tensor(x.shape) if t > 0 else 0 # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = self.cond_noise(x_shape=x_start.shape, noise=noise)

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]
        # print("q_sample t", t)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    def generate_noise(self,shape):
        batch, device = shape[0], self.device

        # img = torch.randn(shape, device = device)
        # This function is called in PUTrainer to sample jets
        # img = self.jet_to_tensor(shape=shape) # Geenerates a jet
        # Choose random jet
        jets = []
        # jet_indices = [16897, 54328, 7898, 2854]
        for i in range(batch):
            # random_jet_no = np.random.randint(low=0, high=self.jetNG._max_TT_no, size=None)
            self.jetNG._next_jetID = i
            # Prints jets to make images from
            print(self.jetNG._next_jetID)
            # self.jetNG.select_jet(random_jet_no)  # or however you select jets
            self.jetNG.select_jet(i)  # or however you select jets
            
            jet = torch.from_numpy(self.jetNG.get_grid()).unsqueeze(0)
            # Now to add pile-up
            # random_pu_no = np.random.randint(low=0, high=self.jetNG._max_TT_no, size=None)
            self.puNG._next_jetID = self.jetNG._next_jetID
            # Start from 200 pileups
            self.puNG.mu = self.timesteps
            # Generate them
            next(self.puNG)
            selected_pu = self.puNG.get_grid()
            # If empty pile-up, return array of 0s instead since model should account for this
            # if selected.size == 0:
            #     return  "Error in PUDiffusion.generate_jet"
            # print(selected_pu.shape)
            pu_tensor = torch.from_numpy(selected_pu)
            # Get same shape, 1 x  grid x grid
            pu_tensor = torch.unsqueeze(pu_tensor,0)
            # pu_tensor = torch.unsqueeze(pu_tensor,0)
            # pu_tensor = pu_tensor.expand(shape[0], shape[1], -1, -1) 
            # pu_tensor = torch.zeros(shape)
            noised_jet = jet + pu_tensor # add energies element wise for each bin
            noised_jet = noised_jet.to(self.device)
            jets.append(noised_jet)
        # Should now  be batch x 1 x grid x grid
        jets = torch.stack(jets)
        jets = jets.to(self.device)
        return jets
        # x = torch.from_numpy( self.jetNG.get_grid() )
        # x = torch.unsqueeze(x,0)
        # # This tensor has dimensions BxCxHxW to match x_start
        # x = torch.unsqueeze(x,0)
        # x = x.expand(shape[0], shape[1], -1, -1) 
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(0)
        # pu_tensor = torch.zeros(shape)
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        # print("p sample loop")
        # print(shape)
        batch, device = shape[0], self.device
        img = self.generate_noise(shape)
        # img = torch.randn(shape, device = device)
        # This function is called in PUTrainer to sample jets
        # img = self.jet_to_tensor(shape=shape) # Geenerates a jet
        # Choose random jet
        # random_jet_no = np.random.randint(low=0, high=self.jetNG._max_TT_no, size=None)
        # self.jetNG.select_jet(random_jet_no)
        # print(self.jetNG._next_jetID)
        # x = torch.from_numpy( self.jetNG.get_grid() )
        # x = torch.unsqueeze(x,0)
        # # This tensor has dimensions BxCxHxW to match x_start
        # x = torch.unsqueeze(x,0)
        # x = x.expand(shape[0], shape[1], -1, -1) 
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(0)
        # pu_tensor = torch.zeros(shape)
        # img = self.generate_noise()
        # img = img.to(self.device)
        
        # img = img.to(self.device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            self.mu_counter +=1
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        print("final mu: ", self.mu_counter)
        print("final timestep: ", self.num_timesteps)
        return ret
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        # print("is ddim")
        return
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        # img = self.cond_noise(x_start, sampling_timesteps)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # noise = torch.randn_like(img)
            # noise = self.cond_noise(self_cond, noi)
            noise = self.pu_to_tensor(x_start.shape)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret


    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape
        # print(x_start.shape)
        # Select one pile-up at a time for each timestep $t$
        # single_pileup = self.pu.select_event(np.random.randint(low=0, high=self.pu.max_ID, size=1))
        # print("p_losses t", t)

        noise = self.cond_noise(x_start.shape, noise=noise)
        # noise = torch.zeros_like(x_start)

        # if noise is None: 
        # noise = default(noise, lambda: torch.randn_like(x_start))
        # offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        # if offset_noise_strength > 0:
        #     offset_noise = torch.randn(x_start.shape[:2], device = self.device)
        #     noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

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


# diffusion = GaussianDiffusion(
#     model = model,
#     image_size = 16,  # Size of your images (ensure your images are square)
#     timesteps = 1000,  # Number of diffusion steps
#     objective = "pred_x0",
# ).to(device)

diffusion = PUDiffusion(
    model = model,
    puNG = NG_pu,
    jet_ng= NG_jet,
    image_size = bins,  # Size of your images (ensure your images are square)
    timesteps = 200,  # Number of diffusion steps
    objective = "pred_x0",
    sampling_timesteps = None
).to(device)



ng_for_dataloader = NGenForDataloader(NG_jet)
dataloader = DataLoader(ng_for_dataloader, batch_size=train_batch_size, num_workers=2, shuffle = True, pin_memory = True)

save_dir = f"{CWD}/data/checkpoints"
num_epochs = 200
xd = load_and_train(diffusion, dataloader, num_epochs=num_epochs, device=device, save_dir=save_dir, lr=1e-4)

print("Finished training")

# Set model to evaluation mode
model.eval()
diffusion.eval()


#############


MPL_GLOBAL_PARAMS = {
    'text.usetex' : False, # use latex text
    'text.latex.preamble' : r'\usepackage{type1cm}\usepackage{braket}\usepackage{amssymb}\usepackage{amsmath}\usepackage{txfonts}', # latex packages
    'font.size' : 24,
    'figure.dpi' : 600,
    'figure.figsize' : (4, 3),
    'figure.autolayout' : True, # tight layout (True) or not (False)
    'axes.labelpad' : 5,
    'axes.xmargin' : 0,
    'axes.ymargin' : 0,
    'axes.grid' : False,
    'axes.autolimit_mode' : 'round_numbers', # set axis limits by rounding min/max values
    # 'axes.autolimit_mode' : 'data', # set axis limits as min/max values
    'xtick.major.pad' : 10,
    'ytick.major.pad' : 10,
    'xtick.labelsize': label_fontsize,
    'ytick.labelsize': label_fontsize,
    'lines.linewidth' : 1.3,
    'xtick.direction' : 'in',
    'ytick.direction' : 'in',
    'xtick.top' : True,
    'ytick.right' : True,
    'xtick.minor.visible' : True,
    'ytick.minor.visible' : True,
    'axes.prop_cycle': cycler(color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*3),
    'legend.framealpha': None
  }
mpl.rcParams.update(MPL_GLOBAL_PARAMS)

import glob
import re
import os

def get_losses_from_checkpoints(checkpoint_dir='./data/checkpoints'):
    # Get absolute path to checkpoint directory
    print(os.listdir("."))
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*_loss_*.pth')
    
    # Get all matching files
    files = glob.glob(checkpoint_pattern)
    print(files)
    losses = []
    pattern = r'loss_([\d.]+)\.pth'
    
    for file in files:
        match = re.search(pattern, file)
        if match:
            loss = float(match.group(1))
            losses.append(loss)
    
    return len(files), losses

# Usage:
num_files, losses = get_losses_from_checkpoints()  
print(f"Number of checkpoint files: {num_files}")
print(f"Losses: {losses}")

fig = plt.figure(figsize=(5,2))
# num,losses = get_losses_from_checkpoints()
plt.plot(list(range(num_files)),losses)
# plt.show()
plt.savefig("loss.png",dpi=600)
sampled_images = diffusion.sample(batch_size=self.jetNG._max_TT_no - 1)

# Data post-processing
mu = 200
output_path = f"{CWD}/data/3-grid"
output_filename = f"noisy_mu{mu}_event_level_from_grid{bins}.csv"
output_filepath = f"{output_path}/{output_filename}"
histogram_path = f"{output_path}/grid{bins}_hist"
mpl.rcParams.update(MPL_GLOBAL_PARAMS)
if not(os.path.exists(output_path)):
    os.mkdir(output_path)
if not(os.path.exists(histogram_path)):
    os.mkdir(histogram_path)
    
# generator = NoisyGenerator(tt, pile_up, mu=mu)
combined = []
def tensor_to_data(tensor_images):
    tensor_images = tensor_images.detach().cpu().numpy()
    tensor_saves = tensor_images[:10]
    save_image(tensor_saves, "saved_denoised_grids.png")
    for idx,grid in enumerate(tensor_images):
        # Each grid is 1 x bins x bins
        hxW = grid[0] # Selects bins x bins
        enes, detas, dphis = grid_to_ene_deta_dphi(hxW, N=bins)
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

    # Final check before saving
    if np.any(np.isnan(all_data)):
        print("\nWarning: NaN values in final data:")
        print(f"Total NaN count: {np.sum(np.isnan(all_data))}")
        print("NaN locations (row, column):")
        nan_rows, nan_cols = np.where(np.isnan(all_data))
        column_names = ['event_id', 'px', 'py', 'pz', 'eta', 'phi', 'mass', 'p_T']
        for row, col in zip(nan_rows, nan_cols):
            print(f"Row {row}, Column {column_names[col]}")
    

    np.savetxt(
            output_filepath,
            all_data,
            delimiter=",",
            header="event_id,px,py,pz,eta,phi,mass,p_T",
            comments="",
            fmt="%i,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f,%10.10f"
        )
    
    # Here mu is the initial mu we set simulations to start from
    plot_1d_histograms(mu, event_stats_path=output_filepath, output_path=f"{output_path}/grid{bins}_hist")

    # Tensors are current;y in BxCxHxW tensor
    # Want to ex
# for idx, _ in enumerate(generator):
#     grid = generator.get_grid()
    
    

tensor_to_data(sampled_images)
print("Finished running completely.")