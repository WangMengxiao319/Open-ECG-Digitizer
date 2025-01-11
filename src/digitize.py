import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.io import decode_image
from src.config.default import get_cfg
from src.utils import import_class_from_path
from yacs.config import CfgNode as CN
from collections.abc import Iterable


def load_file_paths(config: CN) -> Iterable[str]:
    candidate_filenames = os.listdir(config.images_path)
    candidate_paths = map(lambda x: os.path.join(config.images_path, x), candidate_filenames)
    file_paths = filter(lambda x: x.endswith(tuple(config.image_extensions)), candidate_paths)
    return file_paths


def adjust_image_shape(image: torch.Tensor) -> torch.Tensor:
    C, H, W = image.shape
    if C == 1:
        image = image.expand(3, H, W)
    elif C == 4:
        image = image[:3]
    return image.unsqueeze(0)


def main(config: CN) -> None:
    inference_wrapper_class = import_class_from_path(config.MODEL.class_path)
    inference_wrapper = inference_wrapper_class(**config.MODEL.KWARGS)

    loading_bar = tqdm(load_file_paths(config.DATA))

    for file_path in loading_bar:
        image = decode_image(file_path, mode="RGB")
        image = adjust_image_shape(image)
        got_values = inference_wrapper(image)

        if config.DATA.get("output_path") is not None:
            fig, axs = plt.subplots(2, 2, figsize=(20, 14))
            axs[0, 0].imshow(got_values["image"].squeeze().permute(1, 2, 0).cpu().numpy())
            axs[0, 1].imshow(got_values["image_aligned"].squeeze().permute(1, 2, 0).cpu().numpy())
            axs[1, 0].imshow(got_values["signal_probabilities_aligned"].squeeze().cpu().numpy())
            axs[1, 1].plot(got_values["snake"].cpu().numpy().T)
            axs[1, 1].invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                os.path.join(config.DATA.output_path, os.path.basename(file_path)), bbox_inches="tight", dpi=150
            )
            plt.close()


if __name__ == "__main__":
    cfg = get_cfg("src/config/inference_wrapper.yml")
    main(cfg)
