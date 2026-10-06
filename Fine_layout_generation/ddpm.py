import torch

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.utils import show_config


class Diffusion(object):
    _defaults = {
        #-----------------------------------------------#
        #   Model weights path
        #-----------------------------------------------#
        "model_path"        : '../Checkpoint/Fine_layout_generation.pth',
        #-----------------------------------------------#
        #   Convolution channel settings
        #-----------------------------------------------#
        "channel"           : 64,
        #-----------------------------------------------#
        #   Input image size settings
        #-----------------------------------------------#
        "input_shape"       : (64, 64),
        #-----------------------------------------------#
        #   Beta-related parameters
        #-----------------------------------------------#
        "schedule"          : "linear",
        "num_timesteps"     : 1000,
        "schedule_low"      : 1e-4,
        "schedule_high"     : 0.02,
        #-------------------------------#
        #   Whether to use CUDA
        #   Set to False if no GPU is available
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   Initialize Diffusion
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
            self._defaults[name] = value 
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        #----------------------------------------#
        #   Create the Diffusion model
        #----------------------------------------#
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )
            
        self.net    = GaussianDiffusion(UNet(3, self.channel), self.input_shape, 3, betas=betas)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

