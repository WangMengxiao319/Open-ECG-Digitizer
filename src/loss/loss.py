from torch import nn
from torch.nn import functional as F
import torch


SIGNAL_CLASS: int = 2


class WeightedDiceLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, signal_class: int = SIGNAL_CLASS) -> None:
        """
        Args:
            alpha (float): Extra weight assigned to the signal class.
            signal_class (int, optional): The class that is considered the signal class.
        """
        super(WeightedDiceLoss, self).__init__()
        self.alpha: float = alpha
        self.signal_class: int = signal_class

    def forward(self, pred: torch.Tensor, target_one_hot: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred_probs: torch.Tensor = F.softmax(pred, dim=1)

        intersection: torch.Tensor = torch.sum(pred_probs * target_one_hot, dim=(0, 2, 3))
        union: torch.Tensor = torch.sum(pred_probs, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))
        dice: torch.Tensor = 1 - (2 * intersection + eps) / (union + eps)

        multiplier: torch.Tensor = torch.ones_like(dice).to(target_one_hot.device)
        multiplier[self.signal_class] = self.alpha
        multiplier /= multiplier.sum()

        dice = dice * multiplier

        return dice.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, signal_class: int = SIGNAL_CLASS) -> None:
        """
        Args:
            alpha (float, optional): Extra weight assigned to the signal class.
                Defaults to 1.0, which means that all classes are weighted equally.
            signal_class (int, optional): The class that is considered the signal class.
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.alpha: float = alpha
        self.signal_class: int = signal_class

    def forward(self, pred: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        prob: torch.Tensor = F.softmax(pred, dim=1)
        log_prob: torch.Tensor = F.log_softmax(pred, dim=1)

        loss: torch.Tensor = -torch.sum(log_prob * target_one_hot, dim=1)
        loss = loss * (1 + (target_one_hot[:, self.signal_class] + prob[:, self.signal_class]) * self.alpha)

        return loss.mean()


class FourierLoss(nn.Module):
    def forward(self, snakes: torch.Tensor) -> torch.Tensor:
        snakes = snakes - snakes.mean(dim=1, keepdim=True)

        frequency_snakes = torch.fft.rfft(snakes, dim=1).abs()
        weight_filter = torch.linspace(0, 1, frequency_snakes.shape[1], device=snakes.device).pow(2)
        filtered_frequency: torch.Tensor = frequency_snakes * weight_filter.unsqueeze(0)
        loss: torch.Tensor = filtered_frequency.mean()

        return loss


class VarianceLoss(nn.Module):
    def __init__(self, beta: float):
        """
        Args:
            beta (float): The beta parameter for the sigmoid function, which controls the steepness of the curve.
        """
        super(VarianceLoss, self).__init__()
        self.beta = beta

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * self.beta)

    def forward(self, snakes: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        cumulative_sum = preds.cumsum(dim=0)
        magnitudes = cumulative_sum[-1].clone()
        cumulative_sum = cumulative_sum / magnitudes[None, :]

        snakemap = torch.zeros((snakes.shape[0] + 1, preds.shape[0], preds.shape[1]), device=preds.device)
        row_indices = torch.arange(preds.shape[0], device=preds.device).float().unsqueeze(1)

        snakemap[0] = self._sigmoid((snakes[0] - row_indices))
        for i in range(1, snakes.shape[0]):
            snakemap[i] = self._sigmoid((snakes[i] - row_indices)) - snakemap[:i].sum(dim=0)
        snakemap[-1] = 1 - self._sigmoid((snakes[-1] - row_indices))
        snakemap = torch.clamp(snakemap, min=0) / snakemap.sum(dim=0, keepdim=True)

        weighted_mean = (snakemap * cumulative_sum.unsqueeze(0)).mean(dim=1) / snakemap.mean(dim=1)
        weighted_std = ((cumulative_sum - weighted_mean.unsqueeze(1)).pow(2) * snakemap).mean(dim=(0, 1))
        loss: torch.Tensor = weighted_std[magnitudes > 1].mean()

        return loss


class SnakeLoss(nn.Module):
    def __init__(self, fourier_weight: float = 1e-6, beta: float = 1):
        """
        Args:
            fourier_weight (float, optional): The weight of the fourier loss. Defaults to 1e-6.
            beta (float, optional): The beta parameter for the sigmoid function, which controls the steepness of the curve. Defaults to 1.
        """
        super(SnakeLoss, self).__init__()
        self.fourier_weight = fourier_weight
        self.beta = beta
        self.fourier_loss = FourierLoss()
        self.variance_loss = VarianceLoss(self.beta)

    def forward(self, snake: torch.Tensor, preds: torch.Tensor, iteration: int) -> torch.Tensor:
        fourier = self.fourier_loss(snake)
        variance = self.variance_loss(snake, preds)
        loss: torch.Tensor = self.fourier_weight * fourier + variance
        return loss
