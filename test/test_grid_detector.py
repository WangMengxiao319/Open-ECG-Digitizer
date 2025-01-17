from src.model.grid_detector import MultiscaleGridDetector, GridDetector
import torch
import matplotlib.pyplot as plt
from typing import Dict, List


def test_multiscale_grid_detector() -> None:
    image: torch.Tensor = torch.tensor(plt.imread("./test/test_data/data/ecg_data/10_1.png"))
    image = image.mean(2).float()
    binary_image: torch.Tensor = (image > 0.6) & (image < 0.95)

    grid_detector: GridDetector = GridDetector(n_iter=2, num_thetas=40, smoothing_sigma=1)
    msc_grid_detector: MultiscaleGridDetector = MultiscaleGridDetector(grid_detector, depth=3, base=3)

    output: Dict[str, List[torch.Tensor]] = msc_grid_detector(binary_image)

    for key in output:
        assert output[key][0].shape == torch.Size([1, 1])
        assert output[key][1].shape == torch.Size([3, 3])
        assert output[key][2].shape == torch.Size([9, 9])
