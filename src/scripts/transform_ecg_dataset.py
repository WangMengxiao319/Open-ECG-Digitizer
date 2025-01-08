import os
import shutil
import matplotlib.pyplot as plt
import torch
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from src.config.default import get_cfg
from src.utils import import_class_from_path


def get_file_id(file: str) -> str:
    return file.split(".")[0]


def remove_and_create_folder(folder: str, delete_folder: bool) -> None:
    if delete_folder and os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def get_files(folder: str, extension: str) -> List[str]:
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(extension)]
    files.sort(key=get_file_id)
    return files


def transform_from_np(scan: NDArray[Any], mask: NDArray[Any], transform: Any) -> Tuple[NDArray[Any], NDArray[Any]]:
    scan_th = torch.tensor(scan).float().permute(2, 0, 1)
    mask_th = torch.nn.functional.one_hot(torch.tensor(mask).long(), NUM_CLASSES).permute(2, 0, 1).float()

    if transform:
        scan_th, mask_th = transform(scan_th, mask_th)

    mask = mask_th.numpy()
    scan = scan_th.permute(1, 2, 0).numpy()

    return scan, mask


def get_transform(config: Dict[Any, Any]) -> Any:
    if "TRANSFORM" not in config:
        raise ValueError("TRANSFORM key not found in config.")
    transform_config = config["TRANSFORM"]
    transform_class = import_class_from_path(transform_config["class_path"])
    return transform_class(**transform_config.get("KWARGS", {}))


if __name__ == "__main__":
    torch.manual_seed(42)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    DELETE_PREVIOUS_DST_FOLDERS = False
    if not DELETE_PREVIOUS_DST_FOLDERS:

        print(
            """
Set DELETE_PREVIOUS_DST_FOLDERS to True when generating the dataset to ensure that no entries from previous runs are
 left in the destination folders. Defaulting to False to prevent accidental deletion of data.
            """
        )

    # Change the following paths to match the location of the ECG dataset on
    # your local machine. The paths should be relative to the location of this
    # file.
    NUM_TRANSFORMS_PER_FILE = 10
    TIMESERIES_FOLDER = "/data/ecg_dataset/ecg_timeseries_1000"
    MASKS_FOLDER = "/data/ecg_dataset/ecg_masks"
    SCANS_FOLDER = "/data/ecg_dataset/ecg_scans"

    TRANSFORM_FOLDERS = ["train", "val"]
    NUM_CLASSES = 3

    config = get_cfg("../config/photo_transform.yml")
    transform = get_transform(config)

    for transform_folder in TRANSFORM_FOLDERS:
        src_folder_masks = f"{MASKS_FOLDER}/{transform_folder}"
        src_folder_scans = f"{SCANS_FOLDER}/{transform_folder}"
        dst_folder_masks = f"{MASKS_FOLDER}/{transform_folder}_transformed"
        dst_folder_scans = f"{SCANS_FOLDER}/{transform_folder}_transformed"

        remove_and_create_folder(dst_folder_masks, DELETE_PREVIOUS_DST_FOLDERS)

        masks_files = get_files(src_folder_masks, ".npy")
        scans_files = get_files(src_folder_scans, ".png")

        for mask_file in tqdm(masks_files, desc=f"Generating {transform_folder} transformed dataset"):
            file_id = get_file_id(mask_file)
            scan_file = f"{file_id}.png"
            if scan_file not in scans_files:
                raise FileNotFoundError(f"Scan file {scan_file} not found in {src_folder_scans}")

            for i in range(NUM_TRANSFORMS_PER_FILE):
                dst_file_id = f"{file_id}_T{i}"
                dst_mask_file = f"{dst_file_id}.npy"
                dst_scan_file = f"{dst_file_id}.png"

                mask = np.load(f"{src_folder_masks}/{mask_file}")
                scan = plt.imread(f"{src_folder_scans}/{scan_file}")

                scan, mask = transform_from_np(scan, mask, transform)

                np.save(f"{dst_folder_masks}/{dst_mask_file}", mask)
                plt.imsave(f"{dst_folder_scans}/{dst_scan_file}", scan)

    print("Finished generating transformed ECG dataset.")
