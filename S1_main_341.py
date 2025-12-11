import torch
import glob
import os.path
import yaml
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch import Tensor
# Imports from https://github.com/Ahus-AIM/Open-ECG-Digitizer
from src.model.unet import UNet
from src.model.perspective_detector import PerspectiveDetector
from src.model.cropper import Cropper
from src.model.pixel_size_finder import PixelSizeFinder
from src.model.signal_extractor import SignalExtractor
from src.model.lead_identifier import LeadIdentifier
from src.model.lead_identifier import LeadIdentifier
import matplotlib.pyplot as plt

# Resample size: trade-off between accuracy and speed/vram
resample_size = 3000 # maximum no. pixels along the longest dimension

# PixelSizeFinder
PixelSizeFinderKwargs = dict(
    mm_between_grid_lines=5,
    samples=1000,
    min_number_of_grid_lines=20,
    max_number_of_grid_lines=100,
    max_zoom=10,
    zoom_factor=10.0,
    lower_grid_line_factor=0.1,
)

# SignalExtractor
SignalExtractorKwargs = dict(
    threshold_sum=10.0,
    threshold_line_in_mask=0.8,
    label_thresh=0.05,
    max_iterations=4,
    split_num_stripes=4,
    candidate_span=10,
    lam=0.5,
    min_line_width=30,
)

# PerspectiveDetector
PerspectiveDetectorKwargs = dict(
    num_thetas=200, # Higher -> more accurate but slower and more VRAM
    max_num_nonzero=10_000,
)

# LeadIdentifier
LeadIdentifierKwargs = dict(
    layouts=None,  # should be a dict[str, Any]
    unet=None,     # should be a torch.nn.Module
    device=None,   # should be a torch.device
    possibly_flipped=False,    # TODO: 注意，这里如果为True，有概率会误颠倒，因此建议设为False
    target_num_samples=None,
    required_valid_samples=2,
    debug=False,
)

# Cropper
CropperKwargs = dict(
    granularity=50,
    percentiles=(0.03, 0.97),
    alpha=0.95,
)

# Layout UNet (you CAN change this but then you need to retrain the net)
# https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/main/src/train.py
# https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/main/src/config/lead_name_unet.yml
LayoutUNetKwargs = dict(
    weights_path="weights/lead_name_unet_weights_07072025.pt",
    num_in_channels=1,
    num_out_channels=13,
    dims=[32, 64, 128, 256, 256],
    depth=2,
)

# Segmentation UNet (you CAN change this but then you need to retrain the net)
# https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/main/src/train.py
# https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/main/src/config/unet.yml
LoadModelKwargs = dict(
    weights_path="weights/unet_weights_07072025.pt",
    num_in_channels=3,
    num_out_channels=4,
    dims=[32, 64, 128, 256, 320, 320, 320, 320],
    depth=2,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device',device)

def add_noise_to_image(input_img, sigma=1.0, opacity=0.85):
    noise = torch.sigmoid(torch.randn_like(input_img) * sigma)
    input_img = (opacity)*input_img + (1-opacity) * noise
    return input_img

def load_model(**kwargs):
    weights_path = kwargs.pop("weights_path", None)  # safely extract
    model = UNet(**kwargs)
    state_dict = torch.load(weights_path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def load_png_file(path):
    img = read_image(path)
    img = img.float() / 255.0
    img = img.unsqueeze(0)
    if img.shape[1] > 3:
        img = img[:, :3, :, :]
    if img.shape[2]>img.shape[3]:
        img = img.transpose(2, 3).flip(3) 
    return img


def _crop_y(
    image: Tensor,
    signal_prob: Tensor,
    grid_prob: Tensor,
    text_prob: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    def get_bounds(tensor: Tensor) -> tuple[int, int]:
        prob = torch.clamp(
            tensor.squeeze().sum(dim=tensor.dim() - 3)
            - tensor.squeeze().sum(dim=tensor.dim() - 3).mean(),
            min=0,
        )
        non_zero = (prob > 0).nonzero(as_tuple=True)[0]
        if non_zero.numel() == 0:
            return 0, tensor.shape[2] - 1
        return int(non_zero[0].item()), int(non_zero[-1].item())

    y1, y2 = get_bounds(signal_prob + grid_prob)

    slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(None))
    return (
        image[slices],
        signal_prob[slices],
        grid_prob[slices],
        text_prob[slices],
    )


def _align_feature_maps(
    cropper: Cropper,
    image: Tensor,
    signal_prob: Tensor,
    grid_prob: Tensor,
    text_prob: Tensor,
    source_points: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    aligned_signal_prob = cropper.apply_perspective(
        signal_prob,
        source_points,
        fill_value=0,
    )
    aligned_image = cropper.apply_perspective(
        image,
        source_points,
        fill_value=0,
    )
    aligned_grid_prob = cropper.apply_perspective(
        grid_prob,
        source_points,
        fill_value=0,
    )
    aligned_text_prob = cropper.apply_perspective(
        text_prob,
        source_points,
        fill_value=0,
    )
    (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    ) = _crop_y(
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    )

    return (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    )


def plot_segmentation_and_image(
    image,
    segmentation,
    aligned_signal,
    aligned_grid,
    lines,
    output_dir,
    file_name="",
):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 1. 原始图像
    # -----------------------------
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(image_np)
    plt.axis("off")
    plt.tight_layout()
    # plt.savefig(output_dir+"1_raw_image.png", dpi=300)
    plt.close()

    # -----------------------------
    # 2. segmentation feature map
    # -----------------------------
    probs = segmentation.squeeze(0).cpu()
    show_featuremap = torch.ones(probs.shape[1], probs.shape[2], 3)
    probs[2] /= probs[2].max()
    show_featuremap[:, :, [0, 1, 2]] -= 2 * probs[2].unsqueeze(-1)
    show_featuremap[:, :, [1, 2]] -= probs[0].unsqueeze(-1)
    show_featuremap = torch.clamp(show_featuremap, 0, 1).numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(show_featuremap)
    plt.axis("off")
    plt.tight_layout()
    # plt.savefig(output_dir+"2_segmentation_featuremap.png", dpi=300)
    plt.close()

    # -----------------------------
    # 3. straightened image
    # -----------------------------
    straightened_featuremap = torch.ones(
        aligned_signal.shape[2],
        aligned_signal.shape[3],
        3,
        device=aligned_signal.device,
    )
    aligned_signal /= aligned_signal.max()
    straightened_featuremap[:, :, [0, 1, 2]] -= 2 * aligned_signal[0, 0].unsqueeze(-1)
    aligned_grid /= aligned_grid.max()
    straightened_featuremap[:, :, [1, 2]] -= aligned_grid[0, 0].unsqueeze(-1)
    straightened_featuremap = torch.clamp(straightened_featuremap, 0, 1)

    plt.figure(figsize=(6,6))
    plt.imshow(straightened_featuremap.cpu())
    plt.axis("off")
    plt.tight_layout()
    # plt.savefig(output_dir+"3_straightened_featuremap.png", dpi=300)
    plt.close()

    # -----------------------------
    # 4. lines 图
    # -----------------------------
    plt.figure(figsize=(6,6))
    if lines.numel() > 0:
        offsets = [-0, -10.5, -7, -0, -3.5, -7, -0, -3.5, -7, -0, -3.5, -7]   # standard_3x4_with_r1
        # offsets = [-0, -21, -7, -10.5, -14, -17.5, -0, -3.5, -7, -10.5, -14, -17.5]  # standard_6*2_with_r1
        plt.plot(lines.T.cpu().numpy() + offsets[: lines.shape[0]])
        # plt.plot(lines[1,:2500].T.cpu().numpy() -3.5)
    plt.axis("off")
    plt.tight_layout()
    # plt.savefig(output_dir+"4_lines.png", dpi=300)
    plt.close()

    # -----------------------------
    # 5. 保存四宫格大图（原始你的版本）
    # -----------------------------
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax[0, 0].imshow(image_np)
    ax[0, 0].axis("off")

    ax[0, 1].imshow(show_featuremap)
    ax[0, 1].axis("off")

    ax[1, 0].imshow(straightened_featuremap.cpu())
    ax[1, 0].axis("off")

    if lines.numel() > 0:
        ax[1, 1].plot(lines.T.cpu().numpy() + offsets[: lines.shape[0]])
        ax[1, 1].plot(lines[1,:2500].T.cpu().numpy() -3.5) 
    ax[1, 1].axis("off")

    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_name = os.path.join(output_dir,f"{file_name}_full_digitalization.png")
    fig.savefig(output_name, dpi=200)
    plt.close(fig)
    # plt.show()




def crop_image(image, probs):
    perspective_detector = PerspectiveDetector(**PerspectiveDetectorKwargs)

    cropper = Cropper(**CropperKwargs)

    alignment_params = perspective_detector(probs[0, 0])

    source_points = cropper(probs[0, 1], alignment_params)

    signal_prob, grid_prob, text_prob = (
        probs[:, [2]],
        probs[:, [0]],
        probs[:, [1]],
    )

    (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    ) = _align_feature_maps(
        cropper,
        image,
        signal_prob,
        grid_prob,
        text_prob,
        source_points,
    )

    return (
        aligned_image,
        aligned_signal_prob,
        aligned_grid_prob,
        aligned_text_prob,
    )


def extract_signals(
    aligned_signal_prob: Tensor,
    aligned_grid_prob: Tensor,
    aligned_text_prob: Tensor,
    target_num_samples: int,
) -> Tensor:
    pixel_size_finder = PixelSizeFinder(**PixelSizeFinderKwargs)
    signal_extractor = SignalExtractor(**SignalExtractorKwargs)

    layout_unet = load_model(
        **LayoutUNetKwargs,
    )

    layouts = yaml.safe_load(
        open("src/config/lead_layouts_george-moody-2024.yml", "r"),
    )

    identifier_kwargs = LeadIdentifierKwargs.copy()
    identifier_kwargs.update(
        dict(
            layouts=layouts,
            unet=layout_unet,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            target_num_samples=target_num_samples,
        )
    )
    identifier = LeadIdentifier(**identifier_kwargs)
    mm_per_pixel_x, mm_per_pixel_y = pixel_size_finder(aligned_grid_prob)

    avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2 # Is there a better way?
    print('prob',aligned_signal_prob.shape)
    signals = signal_extractor(aligned_signal_prob.squeeze())
    
    signals = identifier(
        signals,
        aligned_text_prob,
        avg_pixel_per_mm=avg_pixel_per_mm,
    )

    return signals

def resample_image(image: Tensor, resample_size: int) -> Tensor:
    height, width = image.shape[2], image.shape[3]
    min_dim = min(height, width)
    max_dim = max(height, width)

    if isinstance(resample_size, int):
        if max_dim > resample_size:
            scale = resample_size / max_dim
            new_size = (int(height * scale), int(width * scale))
            return F.interpolate(image, size=new_size, mode="bilinear", align_corners=False, antialias=True)
        return image

    if isinstance(resample_size, tuple):
        interpolated = F.interpolate(
            image, size=resample_size, mode="bilinear", align_corners=False, antialias=True
        )
        return interpolatedkaggle16

    raise ValueError(f"Invalid resample_size: {resample_size}. Expected int or tuple of (height, width).")

leads_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
def get_slice(lead_name: str, number_of_rows: int) -> slice:
    assert lead_name in leads_names
    if lead_name in ("II",):
        return slice(0, number_of_rows)
    if lead_name in (("I", "III")):
        return slice(0, number_of_rows)
    if lead_name in (("aVR", "aVF", "aVL")):
        return slice(1*number_of_rows, 2*number_of_rows)
    if lead_name in (("V1", "V2", "V3")):
        return slice(2*number_of_rows, 3*number_of_rows)
    if lead_name in (("V4", "V5", "V6")):
        return slice(3*number_of_rows, 4*number_of_rows)

def digitize_image(input_img: Tensor, resample_size: int, target_num_samples: int) -> Tensor:
    # input_img = add_noise_to_image(input_img) # The UNet is trained for "real" images. Sometimes it performs better with added noise for generated images.
    input_img = resample_image(image=input_img, resample_size=resample_size) # higher resample size is (probably) better but watch out for VRAM and time consraints
    
    with torch.no_grad():
        logits = model(input_img.to(device))
        print('unet1 output,',logits.shape)    
        output_probs = torch.softmax(logits, dim=1) # (B, 4, H, W),  Grid, lead waveform, text, background
        print()
        aligned_image, aligned_signal, aligned_grid, aligned_text = crop_image(input_img, output_probs)
        lines = extract_signals(aligned_signal, aligned_grid, aligned_text, target_num_samples=target_num_samples)
        lines = lines["canonical_lines"] * 1e-3  # microvolt to millivolt

    return output_probs, aligned_signal, aligned_grid, lines.float()
    

if __name__ == "__main__":
    model = load_model(**LoadModelKwargs)

    # paths = [
    #     # "/kaggle/input/physionet-ecg-image-digitization/train/1006427285/1006427285-0004.png",
    #     "/kaggle/input/physionet-ecg-image-digitization/train/1006427285/1006427285-0005.png",
    #     # "/kaggle/input/physionet-ecg-image-digitization/train/1006427285/1006427285-0006.png",
    # ]
    input_dir = 'data/PICMAN_ECG/Electrocardiograms_paper'
    output_path = "output/PICMAN_ECG/Electrocardiograms_digitized"
    # get all png and jpg files in the input_dir
    paths = glob.glob(os.path.join(input_dir, '*.png'))+glob.glob(os.path.join(input_dir, '*.jpg'))

    leads_names =    ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    number_of_rows = [2500,10000,2500,2500,2500,2500,2500,2500,2500,2500,2500,2500]
    
    # print(paths)
    # 取list的第一个元素作为示例
    os.makedirs(output_path, exist_ok=True)
    for i, path in enumerate(paths):
        # if i == 0:  # for test only
        file_name = os.path.basename(path).split('.')[0]
        input_img = load_png_file(path)
        if not os.path.exists(path):
            continue

        ### SIGNAL EXTRACTION ###
        print(f"Segmenting {path}...")
        
        output_probs, aligned_signal, aligned_grid, lines = digitize_image(input_img, 4000, 10000)
        print(lines.shape)  # （12，10000）
        plot_segmentation_and_image(input_img, output_probs, aligned_signal, aligned_grid, lines,output_dir=output_path, file_name=file_name)

        ### SAVING ###
        # save to csv
        # 假设总共有 10000 个时间点
        n_rows = 10000
        df = pd.DataFrame(index=range(n_rows))
        
        for i,lead_name in enumerate(leads_names):
            lead_data = lines[i]
            lead_slice = get_slice(lead_name, number_of_rows[i])
            print(lead_slice)
            # 将 Tensor 转成 numpy
            lead_values = lead_data[lead_slice].squeeze().cpu().numpy()
            print(lead_values.shape)

            # 如果是 slice 类型：
            rows = df.index[lead_slice]  # 这样会得到一个长度严格匹配的 Index 对象
            df.loc[rows, lead_name] = lead_values

        csv_path = os.path.join(output_path, os.path.basename(path).replace(".png", ".csv").replace(".jpg", ".csv"))
        df.to_csv(csv_path, index=False)