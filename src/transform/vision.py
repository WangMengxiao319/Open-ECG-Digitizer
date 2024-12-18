import torch
from torch import nn
from typing import Tuple, Union
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import random
import os
import matplotlib.pyplot as plt


class RandomShiftTextTransform(nn.Module):
    """Randomly overlays a shifted version of the text in the image."""

    def __init__(
        self,
        roll_x_range: Tuple[int, int] = (-1000, 1000),
        roll_y_range: Tuple[int, int] = (-1000, 1000),
        opacity_range: Tuple[float, float] = (0.4, 0.8),
    ):
        super().__init__()
        self.roll_x_range = roll_x_range
        self.roll_y_range = roll_y_range
        self.opacity_range = opacity_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        roll_x = random.randint(*self.roll_x_range)
        roll_y = random.randint(*self.roll_y_range)
        opacity = random.uniform(*self.opacity_range)

        text_mask = (mask[1:2] > 0).to(img.dtype)

        text_region = img * text_mask

        rolled_text_mask = torch.roll(text_mask, shifts=(roll_x, roll_y), dims=(-2, -1))
        rolled_text_region = torch.roll(text_region, shifts=(roll_x, roll_y), dims=(-2, -1))

        blended_img = torch.where(rolled_text_mask.bool(), (1 - opacity) * img + opacity * rolled_text_region, img)

        background_mask = (mask[2:] == 0).bool()
        updated_text_mask = torch.where(
            background_mask & (rolled_text_mask[0] > 0), torch.max(opacity * torch.ones_like(mask[1]), mask[1]), mask[1]
        )
        mask[1] = updated_text_mask

        return blended_img, mask


class RandomTextOverlayTransform(nn.Module):
    """Reads a set of text images and overlays them on the input image."""

    def __init__(self, text_path: str, opacity_range: Tuple[float, float] = (0.1, 0.8)):
        super().__init__()
        self.texts = [
            torch.tensor(plt.imread(os.path.join(text_path, f))).float().permute(2, 0, 1)[:3]
            for f in os.listdir(text_path)
        ]
        self.opacity_range = opacity_range
        self.rotation = T.RandomRotation(degrees=180, fill=(1,), interpolation=T.InterpolationMode.BILINEAR)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text = None
        while text is None or text.shape[-1] < img.shape[-1] or text.shape[-2] < img.shape[-2]:
            text = self.texts[random.randint(0, len(self.texts) - 1)]
        text = self.rotation(text)

        if img.shape[0] == 1:
            text = text.mean(0, keepdim=True)

        c1 = random.randint(0, text.shape[-2] - img.shape[-2])
        c2 = random.randint(0, text.shape[-1] - img.shape[-1])
        text_crop = text[:, c1 : c1 + img.shape[-2], c2 : c2 + img.shape[-1]]

        binary_text_mask = (text_crop < 0.8).sum(0) > 0
        opacity = random.uniform(*self.opacity_range)

        img = torch.where(binary_text_mask, opacity * (text_crop) + (1 - opacity) * img, img)

        mask_channel_1_condition = binary_text_mask & (mask[2:] == 0).all(0)
        mask[1] = torch.where(mask_channel_1_condition, torch.max(opacity * torch.ones_like(mask[1]), mask[1]), mask[1])

        return img, mask.float()


class RandomFlipTransform(nn.Module):
    """Randomly flips the image and mask."""

    def __init__(self, p_ud: float = 0.5, p_lr: float = 0.5):
        super().__init__()
        self.p_ud = p_ud
        self.p_lr = p_lr

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p_ud:
            img, mask = F.vflip(img), F.vflip(mask)
        if random.random() < self.p_lr:
            img, mask = F.hflip(img), F.hflip(mask)
        return img, mask


class GaussianBlurTransform(nn.Module):
    """Applies Gaussian blur to the image."""

    def __init__(self, kernel_size: int = 7, sigma: Tuple[float, float] = (10.0, 10.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = random.uniform(*self.sigma_range)
        img = F.gaussian_blur(img, kernel_size=self.kernel_size, sigma=sigma)
        return img, mask


class RandomSharpnessTransform(nn.Module):
    """Adjusts the sharpness of the image."""

    def __init__(self, sharpness_range: Tuple[float, float] = (1.0, 3.0)):
        super().__init__()
        self.sharpness_range = sharpness_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sharpness = random.uniform(*self.sharpness_range)
        img = F.adjust_sharpness(img, sharpness_factor=sharpness)
        return img, mask


class RandomBlurOrSharpnessTransform(nn.Module):
    """Randomly applies either Gaussian blur or sharpness adjustment."""

    def __init__(
        self,
        p: float = 0.5,
        sharpness_range: Tuple[float, float] = (1.0, 2.0),
        kernel_size: int = 7,
        sigma: Tuple[float, float] = (0.0, 1.5),
    ):
        super().__init__()
        self.p = p
        self.blur = GaussianBlurTransform(kernel_size=kernel_size, sigma=sigma)
        self.sharpness = RandomSharpnessTransform(sharpness_range=sharpness_range)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            img, mask = self.blur(img, mask)
        else:
            img, mask = self.sharpness(img, mask)
        return img, mask


class RandomGammaTransform(nn.Module):
    """Applies gamma correction to the image."""

    def __init__(self, gamma_range: Tuple[float, float] = (0.5, 2.0)):
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = random.uniform(*self.gamma_range)
        img = F.adjust_gamma(img, gamma=gamma)
        return img, mask


class RandomResizedCropTransform(nn.Module):
    """Crops and resamples to specified size."""

    def __init__(self, size: Tuple[int, int] = (1700, 2200), scale: Tuple[float, float] = (0.5, 1.0)):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=self.scale, ratio=(1.0, 1.0))
        img = F.resized_crop(img, i, j, h, w, self.size)
        mask = F.resized_crop(mask, i, j, h, w, self.size)
        return img, mask


class RandomRotation(nn.Module):
    """Randomly rotates both image and mask."""

    def __init__(self, degrees: Union[float, Tuple[float, float]] = 10):
        super().__init__()
        self.degrees = degrees

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angle = (
            random.uniform(-self.degrees, self.degrees)
            if isinstance(self.degrees, (int, float))
            else random.uniform(*self.degrees)
        )
        img = F.rotate(img, angle, fill=1, interpolation=T.InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, fill=[1, 0, 0], interpolation=T.InterpolationMode.BILINEAR)
        return img, mask


class RandomJPEGCompression(nn.Module):
    """Simulates JPEG compression artifacts."""

    def __init__(self, quality: Union[int, Tuple[int, int]] = (5, 100)):
        super().__init__()
        self.jpeg_transform = T2.JPEG(quality=quality)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img = (img * 255).clamp(0, 255).byte()
        img = self.jpeg_transform(img)
        img = img.float() / 255
        return img, mask


class RandomGradientOverlay(nn.Module):
    """Applies a gradient overlay to simulate uneven lighting."""

    def __init__(self, p: float = 0.5, opacity_range: Tuple[float, float] = (-0.3, 0.1)):
        super().__init__()
        self.p = p
        self.opacity_range = opacity_range

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = img.shape[1], img.shape[2]
        if random.random() < self.p:
            gradient = torch.linspace(0, 1, w).unsqueeze(0)
            gradient = gradient.repeat(h, 1).unsqueeze(0)  # (1, H, W)
        else:
            gradient = torch.linspace(0, 1, h).unsqueeze(1)
            gradient = gradient.repeat(1, w).unsqueeze(0)  # (1, H, W)

        opacity = random.uniform(*self.opacity_range)

        img = img + opacity * gradient.to(img.device)
        img = img.clamp(0, 1)
        return img, mask


class RefineMask(nn.Module):
    """Ensures that the mask is valid after transformations."""

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask / mask.sum(0, keepdim=True)
        return img, mask


class ComposedTransform(nn.Module):
    """Applies transformations in sequence."""

    def __init__(self, transforms: list[nn.Module]):
        super(ComposedTransform, self).__init__()
        self.transforms = transforms

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask


class ScanTransform(nn.Module):
    """Default composed transformation for ECG scans."""

    def __init__(self) -> None:
        super(ScanTransform, self).__init__()
        self.transform: ComposedTransform = ComposedTransform(
            [
                RandomShiftTextTransform(),
                RandomTextOverlayTransform(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay_images")),
                RandomFlipTransform(),
                RandomBlurOrSharpnessTransform(),
                RandomGammaTransform(),
                RandomResizedCropTransform(),
                RandomRotation(),
                RandomJPEGCompression(),
                RandomGradientOverlay(),
                RefineMask(),
            ]
        )

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(img, mask)  # type: ignore
