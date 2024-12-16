import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks


class Snake(nn.Module):
    def __init__(
        self,
        preds: torch.Tensor,
        num_peaks: int = 6,
        sigma: float = 10.0,
        left_percentile: float = 0.01,
        right_percentile: float = 0.99,
        top_percentile: float = 0.01,
        bottom_percentile: float = 0.99,
        horizontal_margin: int = 1,
        vertical_margin: int = 50,
    ):
        super(Snake, self).__init__()
        self.num_peaks = num_peaks
        self.sigma = sigma
        self.left_percentile = left_percentile
        self.right_percentile = right_percentile
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.horizontal_margin = horizontal_margin
        self.vertical_margin = vertical_margin
        self.device = preds.device
        self.cropped_preds = self._crop_preds(preds)
        self._check_input()
        self.shape = (self.num_peaks, self.cropped_preds.shape[1])
        self.snake = nn.Parameter(self._initialize_snake())

    def _check_input(self) -> None:
        assert 0 <= self.cropped_preds.min(), "preds values must be in range 0,1"
        assert 1 >= self.cropped_preds.max(), "preds values must be in range 0,1"
        assert 0 <= self.left_percentile <= 1, "left_percentile must be in range 0,1"
        assert 0 <= self.right_percentile <= 1, "right_percentile must be in range 0,1"
        assert 0 <= self.top_percentile <= 1, "top_percentile must be in range 0,1"
        assert 0 <= self.bottom_percentile <= 1, "bottom_percentile must be in range 0,1"
        assert 0 < self.num_peaks <= 10, "num_peaks must be in range 1,10"
        assert 0 <= self.sigma <= 100, "sigma must be in range 0,100"

    def _crop_preds(self, preds: torch.Tensor) -> torch.Tensor:
        x_projection = preds.sum(0)
        y_projection = preds.sum(1)
        cumulative_sum_x = x_projection.cumsum(0) / x_projection.cumsum(0)[-1]
        cumulative_sum_y = y_projection.cumsum(0) / y_projection.cumsum(0)[-1]

        left_bound = int((cumulative_sum_x - self.left_percentile).abs().argmin().item())
        right_bound = int((cumulative_sum_x - self.right_percentile).abs().argmin().item())
        top_bound = int((cumulative_sum_y - self.top_percentile).abs().argmin().item())
        bottom_bound = int((cumulative_sum_y - self.bottom_percentile).abs().argmin().item())

        left_bound = max(0, left_bound - self.horizontal_margin)
        right_bound = min(preds.shape[1], right_bound + self.horizontal_margin)
        top_bound = max(0, top_bound - self.vertical_margin)
        bottom_bound = min(preds.shape[0], bottom_bound + self.vertical_margin)

        cropped_preds = preds[top_bound:bottom_bound, left_bound:right_bound]

        return cropped_preds

    def _initialize_snake(self) -> torch.Tensor:
        filtered_preds = gaussian_filter(self.cropped_preds.cpu().numpy(), sigma=(self.sigma, 1))
        snake_indices = torch.full(self.shape, torch.nan, device=self.device)

        for col in range(self.shape[1]):
            peaks, _ = find_peaks(filtered_preds[:, col], distance=10, threshold=1e-9)
            peaks = peaks[np.argsort(filtered_preds[peaks, col])[-self.num_peaks :]]
            peaks = np.sort(peaks)
            snake_indices[:, col] = torch.tensor(peaks).float() if len(peaks) == self.num_peaks else torch.nan

        mean_indices = torch.nanmean(snake_indices, dim=1)
        for i in range(snake_indices.shape[0]):
            snake_indices[i][torch.isnan(snake_indices[i])] = mean_indices[i]

        snake_indices = median_filter(snake_indices.cpu().numpy(), size=(1, 5))
        return torch.tensor(snake_indices, device=self.device)

    def forward(self) -> torch.Tensor:
        self.snake.data = torch.sort(self.snake.data, dim=0)[0]
        return self.snake
