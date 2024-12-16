from typing import Tuple
import torch
import torch.nn.functional as F


class GridDetector:
    def __init__(self, n_iter: int, num_thetas: int, smoothing_sigma: int, hann_window_scale: float):
        self.n_iter = n_iter
        self.num_thetas = num_thetas
        self.smoothing_sigma = smoothing_sigma
        self.hann_window_scale = hann_window_scale

    def create_gaussian_kernel(self, sigma: int) -> torch.Tensor:
        size = 5 * sigma
        x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        x, y = torch.meshgrid(x, y, indexing="ij")
        kernel = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
        kernel /= kernel.sum()
        return kernel.view(1, 1, size, size)

    def hough_transform(
        self, image: torch.Tensor, thetas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = image.device
        H, W = image.shape
        diag_len = int(torch.sqrt(torch.tensor(H**2 + W**2, device=device)))
        rhos = torch.linspace(-diag_len, diag_len, 2 * diag_len, device=device)
        num_thetas = len(thetas)
        num_rhos = len(rhos)

        y_idxs, x_idxs = torch.nonzero(image, as_tuple=True)
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)

        x_idxs = x_idxs.view(-1, 1).float()
        y_idxs = y_idxs.view(-1, 1).float()
        rhos_vals = x_idxs * cos_thetas + y_idxs * sin_thetas
        rhos_idxs = torch.round((rhos_vals - rhos[0]) / (rhos[1] - rhos[0])).int()
        rhos_idxs = rhos_idxs.clamp(0, len(rhos) - 1)

        accumulator = torch.zeros(num_rhos * num_thetas, dtype=torch.int32, device=device)
        idxs_flat = rhos_idxs * num_thetas + torch.arange(num_thetas, device=device).reshape(1, -1)

        idxs_flat = idxs_flat.flatten()
        idxs_flat = idxs_flat[idxs_flat < num_rhos * num_thetas]

        accumulator.index_add_(0, idxs_flat, torch.ones_like(idxs_flat, dtype=torch.int32))
        accumulator = accumulator.view(len(rhos), len(thetas))

        return accumulator, rhos, thetas

    @torch.compile
    def detect_angles(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = image.device
        thetas = torch.linspace(0, torch.pi, self.num_thetas, device=device) - torch.pi / 4

        kernel = self.create_gaussian_kernel(self.smoothing_sigma).to(device)

        for i in range(self.n_iter):
            accumulator, rhos, thetas = self.hough_transform(image, thetas)

            accumulator = (
                torch.nn.functional.conv2d(accumulator.float().unsqueeze(0).unsqueeze(0), kernel, padding="same")
                .squeeze(0)
                .squeeze(0)
            )

            fft_accumulator: torch.Tensor = torch.fft.rfft(accumulator, dim=0).abs()
            projected_accumulator = fft_accumulator.sum(0)

            hann = torch.hann_window(int(fft_accumulator.shape[1] * self.hann_window_scale), device=device)
            hann = torch.cat([hann, hann])
            projected_accumulator *= hann

            n = len(projected_accumulator)
            first_peak = projected_accumulator[: n // 2].argmax()
            second_peak = projected_accumulator[n // 2 :].argmax() + n // 2

            thetas1 = torch.linspace(
                thetas[first_peak - 3], thetas[first_peak + 3], self.num_thetas // 2, device=device
            )
            thetas2 = torch.linspace(
                thetas[second_peak - 3], thetas[second_peak + 3], self.num_thetas // 2, device=device
            )
            thetas = torch.cat([thetas1, thetas2])

        return (
            thetas[first_peak].float(),
            thetas[second_peak].float(),
            fft_accumulator[:, first_peak],
            fft_accumulator[:, second_peak],
        )

    @torch.compile
    def detect_grid(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        theta1, theta2, fft_accumulator1, fft_accumulator2 = self.detect_angles(image)
        accumulator1 = torch.fft.rfft(fft_accumulator1, n=fft_accumulator1.shape[0] * 2).abs()[:500]
        accumulator2 = torch.fft.rfft(fft_accumulator2, n=fft_accumulator2.shape[0] * 2).abs()[:500]
        peaks1 = self.find_local_maxima(accumulator1)
        peaks2 = self.find_local_maxima(accumulator2)

        # NOTE perhaps this can be done in a more robust way?
        dist1 = torch.diff(torch.sort(peaks1).values).float()
        dist2 = torch.diff(torch.sort(peaks2).values).float()
        mean_dist1 = dist1.mean()
        mean_dist2 = dist2.mean()

        return theta1, theta2, mean_dist1, mean_dist2

    def find_local_maxima(self, tensor: torch.Tensor) -> torch.Tensor:
        padded_tensor = F.pad(tensor, (1, 1), value=float("-inf"))
        local_maxima = (padded_tensor[1:-1] > padded_tensor[:-2]) & (padded_tensor[1:-1] > padded_tensor[2:])
        indices = local_maxima.nonzero(as_tuple=True)[0]
        return indices
