from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Optional
import torch


SIGNAL_CLASS: int = 2


class MulticlassBinaryLoss(nn.Module):
    def __init__(
        self,
        multiclass_loss: type[nn.Module],
        split_name: str = "",
        signal_class: int = SIGNAL_CLASS,
        **multiclass_loss_kwargs: Any,
    ) -> None:
        """
        Args:
            alpha (float): Extra weight assigned to the signal class.
            signal_class (int, optional): The class that is considered the signal class.
        """
        super(MulticlassBinaryLoss, self).__init__()
        self.split_name: str = split_name
        self.signal_class: int = signal_class
        self.multiclass_loss_kwargs: Dict[Any, Any] = multiclass_loss_kwargs
        self.binary_signal_class: int = 0  # As this class collapses the problem to binary classification.
        self.multiclass_loss_obj = multiclass_loss(signal_class=self.binary_signal_class, **multiclass_loss_kwargs)

    def forward(
        self, pred: torch.Tensor | List[torch.Tensor], target_one_hot: torch.Tensor | List[torch.Tensor]
    ) -> torch.Tensor:
        if type(pred) is not type(target_one_hot):
            raise ValueError("Pred and target_one_hot must be of the same type.")

        if isinstance(pred, torch.Tensor) and isinstance(target_one_hot, torch.Tensor):
            return self.compute_multiclass_binary_loss(pred, target_one_hot)

        total_loss = 0.0
        for curr_pred, curr_target_one_hot in zip(pred, target_one_hot):
            total_loss += self.compute_multiclass_binary_loss(curr_pred, curr_target_one_hot)  # type: ignore
        loss: torch.Tensor = total_loss / len(pred)  # type: ignore
        return loss

    def compute_multiclass_binary_loss(self, pred: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
        pred_signal = pred[:, self.signal_class, :, :].unsqueeze(1)
        prob_signal = F.softmax(pred, dim=1)[:, self.signal_class, :, :].unsqueeze(1)
        target_one_hot_signal = target_one_hot[:, self.binary_signal_class, :, :].unsqueeze(1)
        loss: torch.Tensor = self.multiclass_loss_obj(pred_signal, target_one_hot_signal, probs=prob_signal)
        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, signal_class: int = SIGNAL_CLASS, union_exponent: int = 1) -> None:
        """
        Args:
            alpha (float): Extra weight assigned to the signal class.
            signal_class (int, optional): The class that is considered the signal class.
            union_exponent (int, optional): The exponent to raise the union to. Set to 2 to match the loss
                function for V-Net.
        """
        super(WeightedDiceLoss, self).__init__()
        self.alpha: float = alpha
        self.signal_class: int = signal_class
        self.union_exponent: int = union_exponent

    def forward(
        self, pred: torch.Tensor, target_one_hot: torch.Tensor, eps: float = 1e-6, probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if bool(probs is not None) != bool(target_one_hot.shape[1] == 1):
            raise ValueError("If probs is provided, the targets must be binary and vice versa.")
        is_binary = probs is not None

        pred_probs: torch.Tensor = F.softmax(pred, dim=1) if not is_binary else probs  # type: ignore

        # Must cast to double in order to avoid overflow.
        intersection: torch.Tensor = torch.sum(pred_probs.double() * target_one_hot.double(), dim=(2, 3))
        union: torch.Tensor = torch.sum(pred_probs.double().pow(self.union_exponent), dim=(2, 3)) + torch.sum(
            target_one_hot.double().pow(self.union_exponent), dim=(2, 3)
        )
        dice: torch.Tensor = 1 - (2 * intersection) / (union + eps)

        if not is_binary:
            multiplier: torch.Tensor = torch.ones_like(dice).to(target_one_hot.device)
            multiplier[:, self.signal_class] = self.alpha
            multiplier /= multiplier.mean()

            dice = dice * multiplier

        return dice.mean()


class WeightedDiceLossSquared(WeightedDiceLoss):
    def __init__(self, alpha: float = 1.0, signal_class: int = SIGNAL_CLASS) -> None:
        super(WeightedDiceLossSquared, self).__init__(alpha=alpha, signal_class=signal_class, union_exponent=2)


class MulticlassBinaryDiceLoss(MulticlassBinaryLoss):
    def __init__(
        self, split_name: str = "", alpha: float = 1.0, signal_class: int = SIGNAL_CLASS, union_exponent: int = 1
    ) -> None:
        super(MulticlassBinaryDiceLoss, self).__init__(
            WeightedDiceLoss,
            split_name=split_name,
            signal_class=signal_class,
            alpha=alpha,
            union_exponent=union_exponent,
        )
        self.alpha = alpha
        self.union_exponent = union_exponent

    @property
    def __name__(self) -> str:
        return f"{self.split_name}_multiclass_binary_dice_loss_alpha={self.alpha:.2f}_union_exp={self.union_exponent}_signal={self.signal_class}"


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

    def forward(
        self, pred: torch.Tensor, target_one_hot: torch.Tensor, probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if bool(probs is not None) != bool(target_one_hot.shape[1] == 1):
            raise ValueError("If probs is provided, the targets must be binary and vice versa.")
        is_binary = probs is not None

        prob: torch.Tensor = F.softmax(pred, dim=1) if not is_binary else probs  # type: ignore

        if is_binary:
            binary_loss: torch.Tensor = nn.BCEWithLogitsLoss()(pred, target_one_hot)
            return binary_loss

        with torch.no_grad():
            w = 1 + (target_one_hot[:, self.signal_class] + prob[:, self.signal_class]) * (self.alpha - 1)
            w = w / w.mean()

        log_prob: torch.Tensor = F.log_softmax(pred, dim=1)

        loss: torch.Tensor = -torch.sum(log_prob * target_one_hot, dim=1)
        loss = w * loss

        return loss.mean()


class MulticlassBinaryCrossEntropyLoss(MulticlassBinaryLoss):
    def __init__(self, split_name: str = "", alpha: float = 1.0, signal_class: int = SIGNAL_CLASS) -> None:
        super(MulticlassBinaryCrossEntropyLoss, self).__init__(
            WeightedCrossEntropyLoss, split_name=split_name, signal_class=signal_class, alpha=alpha
        )
        self.alpha = alpha

    @property
    def __name__(self) -> str:
        return (
            f"{self.split_name}_multiclass_binary_cross_entropy_loss_alpha={self.alpha:.2f}_signal={self.signal_class}"
        )


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
