# Compact inference path: direct conditional Euler sampling is shown here;
# optional guidance logic is not included in this example.

import torch
from einops import rearrange
from torchvision.utils import make_grid, save_image


class Normalizer:
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            return (x.clip(-1, 1) + 1) * 0.5
        return x * self.std.to(x.device) + self.mean.to(x.device)


class MeanFlow:
    """Sampling helpers for the trained MeanFlow layout model."""

    def __init__(self, channels=1, image_size=32, num_classes=10,
                 normalizer=('minmax', None, None)):
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.normer = Normalizer.from_list(normalizer)

    def save_images(self, image, img_save_path):
        with torch.no_grad():
            save_image(make_grid(image, nrow=8), img_save_path)

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=5, device='cuda', c_test=None):
        model.eval()

        if classes is None:
            labels = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            labels = torch.tensor(classes, device=device).repeat(n_per_class)

        image = torch.randn(
            labels.shape[0], self.channels, self.image_size, self.image_size,
            device=device,
        )
        time_steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t = torch.full((image.size(0),), time_steps[i], device=device)
            r = torch.full((image.size(0),), time_steps[i + 1], device=device)
            velocity = model(image, t, r, c_test)
            image = image - (t - r).view(-1, 1, 1, 1) * velocity

        return self.normer.unnorm(image)
