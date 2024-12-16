from tqdm import tqdm
from typing import Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


class ECGScanDataset(torch.utils.data.Dataset[Any]):
    def __init__(self, ecg_scan_path: str, ecg_mask_path: str):
        self.ecg_scan_path = ecg_scan_path
        self.ecg_mask_path = ecg_mask_path

        # Prepend current software directory if path is not absolute.
        if not os.path.isabs(self.ecg_scan_path):
            self.ecg_scan_path = os.path.join(os.path.dirname(__file__), self.ecg_scan_path)
        if not os.path.isabs(self.ecg_mask_path):
            self.ecg_mask_path = os.path.join(os.path.dirname(__file__), self.ecg_mask_path)

        self.ecg_scan_files, self.ecg_mask_files = self._find_ecg_files()

        self.ecg_scans = self._load_scans(self.ecg_scan_files)
        self.ecg_masks = self._load_masks(self.ecg_mask_files)

    def _find_ecg_files(self) -> Tuple[List[str], List[str]]:
        for path in [self.ecg_scan_path, self.ecg_mask_path]:
            if not os.path.exists(path):
                raise ValueError(f"Path {path} does not exist.")

        def is_no_underscore_npy_file(file: str) -> bool:
            return not file.startswith("_") and file.endswith(".npy")

        ecg_mask_filenames = list(
            map(lambda x: x.split(".")[0], filter(is_no_underscore_npy_file, os.listdir(self.ecg_mask_path)))
        )

        ecg_scan_files = list(map(lambda x: os.path.join(self.ecg_scan_path, f"{x}.png"), ecg_mask_filenames))
        ecg_mask_files = list(map(lambda x: os.path.join(self.ecg_mask_path, f"{x}.npy"), ecg_mask_filenames))
        return ecg_scan_files, ecg_mask_files

    def _load_scans(self, files: List[str]) -> List[torch.Tensor]:
        loaded_scans = []
        for file in tqdm(files, desc="Loading ECG scans"):
            loaded_scans.append(
                torch.tensor(plt.imread(os.path.join(self.ecg_scan_path, file))).float().permute(2, 0, 1)
            )
        return loaded_scans

    def _load_masks(self, files: List[str]) -> List[torch.Tensor]:
        loaded_files = []
        for file in tqdm(files, desc="Loading ECG grid masks"):
            loaded_files.append(torch.tensor(np.load(file)).float())
        loaded_files = self._mask_to_one_hot(loaded_files)
        return loaded_files

    def _mask_to_one_hot(self, masks: List[torch.Tensor]) -> List[torch.Tensor]:
        num_classes = max([int(torch.max(masks[i]).item()) for i in range(len(masks))]) + 1
        one_hot_mask = []
        for i in range(len(masks)):
            one_hot_mask.append(torch.nn.functional.one_hot(masks[i].long(), num_classes).permute(2, 0, 1))
        return one_hot_mask

    def __len__(self) -> int:
        return len(self.ecg_scan_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ecg_scans[idx], self.ecg_masks[idx]
