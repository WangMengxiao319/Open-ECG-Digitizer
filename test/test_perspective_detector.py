from src.model.perspective_detector import PerspectiveDetector
from src.model.cropper import Cropper
import torch
import matplotlib.pyplot as plt


def test_perspective_detector() -> None:
    image: torch.Tensor = torch.tensor(plt.imread("./test/test_data/data/ecg_data/10_1.png")).permute(2, 0, 1)
    image = image[:, 500:-500, 500:-500]  # Save time by cropping the image
    image = (image - image.min()) / (image.max() - image.min())

    pd = PerspectiveDetector(num_thetas=100)
    params = pd(image)
    cropper = Cropper(granularity=100, percentiles=(0.01, 0.99))

    signal_probabilities = (image[0, :, :] < 0.2).float()
    source_points = cropper(signal_probabilities, params)
    cropped_image = cropper.apply_perspective(image, source_points, fill_value=image.mean().item())

    expected_source_points = torch.tensor(
        [[-5.8998, 10.0548], [1196.3998, -21.3052], [1211.3795, 604.9304], [9.0557, 635.2782]]
    )

    assert torch.allclose(source_points, expected_source_points, atol=50)
    assert cropped_image.shape == image.shape
