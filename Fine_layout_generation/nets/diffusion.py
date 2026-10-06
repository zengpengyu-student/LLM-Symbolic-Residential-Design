import numpy as np
import torch
import torch.nn as nn

from functools import partial
from copy import deepcopy


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, img_size, img_channels, betas=[],
    ):
        super().__init__()
        self.model      = model
        self.ema_model  = deepcopy(model)

        self.img_size       = img_size
        self.img_channels   = img_channels
        self.num_timesteps  = len(betas)

        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas)

        # Convert to a torch.tensor for processing
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # betas             [0.0001, 0.00011992, 0.00013984 ... , 0.02]
        self.register_buffer("betas", to_torch(betas))
        # alphas            [0.9999, 0.99988008, 0.99986016 ... , 0.98]
        self.register_buffer("alphas", to_torch(alphas))
        # alphas_cumprod    [9.99900000e-01, 9.99780092e-01, 9.99640283e-01 ... , 4.03582977e-05]
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # sqrt(1 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        # sqrt(1 / alphas)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)
