import torch
from torch import nn
from typing import Tuple, Union, Dict, Any, List
from src.utils import import_class_from_path
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import random
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import perspective
from torchvision.io import read_image


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
        mask[1] = torch.where(
            background_mask & (rolled_text_mask[0] > 0), torch.max(opacity * torch.ones_like(mask[1]), mask[1]), mask[1]
        )[0]
        mask[0] = torch.where(
            background_mask & (rolled_text_mask[0] > 0),
            torch.min((1 - opacity) * torch.ones_like(mask[0]), mask[0]),
            mask[0],
        )[0]

        return blended_img, mask


class RandomTextOverlayTransform(nn.Module):
    """Reads a set of text images and overlays them on the input image."""

    def __init__(self, text_path: Union[str | None] = None, opacity_range: Tuple[float, float] = (0.1, 0.8)):
        super().__init__()
        self.text_path = text_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlay_images")
        self.texts = [
            torch.tensor(plt.imread(os.path.join(self.text_path, f))).float().permute(2, 0, 1)[:3]
            for f in os.listdir(self.text_path)
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
        mask[0] = torch.where(
            mask_channel_1_condition, torch.min((1 - opacity) * torch.ones_like(mask[1]), mask[0]), mask[0]
        )

        return img, mask.float()


class RandomPerspectiveWithImageTransform(nn.Module):
    def __init__(self, image_path: str, distortion_scale: float = 0.15) -> None:
        super().__init__()
        self.image_path = image_path
        self.distortion_scale = distortion_scale
        self.image_paths = os.listdir(image_path)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        index: int = int(torch.randint(0, len(self.image_paths), (1,)).item())
        random_image_path = self.image_paths[index]
        random_image = read_image(os.path.join(self.image_path, random_image_path)).float().div(255).clamp(0, 1)

        startpoints = [[0, 0], [img.shape[2], 0], [img.shape[2], img.shape[1]], [0, img.shape[1]]]
        offset_range = 0.5 * self.distortion_scale * img.shape[2]
        endpoints = torch.tensor(startpoints) + torch.rand((4, 2)).mul(2 * offset_range) - offset_range / 2
        midpoint = torch.tensor([img.shape[2] / 2, img.shape[1] / 2])
        endpoints = endpoints * 0.95 + midpoint[None, :] * 0.05

        fill_value = torch.rand((1,)).item()

        img = perspective(img.unsqueeze(0), startpoints, endpoints, fill=fill_value).squeeze(0)
        mask = perspective(mask.unsqueeze(0), startpoints, endpoints, fill=0.0).squeeze(0)

        if random_image.shape[1] <= img.shape[1] or random_image.shape[2] <= img.shape[2]:
            random_image = F.resize(random_image, (img.shape[1] + 5, img.shape[2] + 5))

        c11: int = int(torch.randint(0, random_image.shape[1] - img.shape[1], (1,)).item())
        c12: int = c11 + img.shape[1]
        c21: int = int(torch.randint(0, random_image.shape[2] - img.shape[2], (1,)).item())
        c22: int = c21 + img.shape[2]
        random_image = random_image[:, c11:c12, c21:c22]

        img[img == fill_value] = random_image[img == fill_value]
        mask_is_transformed = (mask == 0.0).sum(0) == 3
        mask[0][mask_is_transformed] = 0.5
        mask[1][mask_is_transformed] = 0.5
        mask[2][mask_is_transformed] = 0.0

        return img, mask


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
    """Crops and resamples to a random size within specified height and width ranges."""

    def __init__(
        self,
        h_range: Tuple[int, int] = (1360, 2550),
        w_range: Tuple[int, int] = (1760, 3300),
        scale: Tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()
        self.h_range = h_range
        self.w_range = w_range
        self.scale = scale

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_height = random.randint(*self.h_range)
        target_width = random.randint(*self.w_range)
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=self.scale, ratio=(target_width / target_height, target_width / target_height)
        )
        img = F.resized_crop(img, i, j, h, w, (target_height, target_width))
        mask = F.resized_crop(mask, i, j, h, w, (target_height, target_width))
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

    def __init__(self, transform_config: List[Dict[Any, Any]]):
        """
        Initializes a composed transform based on the given configuration.

        Args:
            transforms (dict): Configuration containing class path and transforms.
        """
        super(ComposedTransform, self).__init__()
        transforms = []
        for transform_def in transform_config:
            transform_class = import_class_from_path(transform_def["class_path"])
            transform = transform_class(**transform_def.get("KWARGS", {}))
            transforms.append(transform)
        self.transforms = transforms

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            img, mask = transform.forward(img, mask)
        return img, mask


class GreyscaleTransform(nn.Module):
    """Converts the image to greyscale."""

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img = F.rgb_to_grayscale(img, num_output_channels=3)
        return img, mask
