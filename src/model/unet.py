from torch import nn
from typing import List
import torch


class UNet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        out_channel: int = 3,
        depth: int = 3,
        dims: List[int] = [96, 192, 384, 768],
    ):
        super(UNet, self).__init__()

        self.depth = depth
        self.dims = dims

        # Encoder blocks
        self.encoders = nn.ModuleList(
            [self._make_encoder_block(in_chans if i == 0 else dims[i - 1], dims[i], depth) for i in range(len(dims))]
        )

        # Decoder blocks
        self.decoders = nn.ModuleList(
            [self._make_decoder_block(dims[i] + dims[i - 1], dims[i - 1]) for i in range(len(dims) - 1, 0, -1)]
        )

        # Final output layer
        self.final_conv = nn.Conv2d(dims[0], out_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        # Forward through encoders
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = nn.MaxPool2d(2)(x)

        # Reverse skips for decoding
        skips = skips[::-1]
        x = skips[0]  # Start decoding from the last skip

        # Forward through decoders
        for i, decoder in enumerate(self.decoders):
            x = self._upsample(x, skips[i + 1])
            x = decoder(torch.cat([x, skips[i + 1]], dim=1))

        # Final convolution to match output channels
        x = self.final_conv(x)
        return x

    def _make_encoder_block(self, in_channels: int, out_channels: int, num_layers: int) -> nn.Sequential:
        layers = [self._conv_bn_relu(in_channels, out_channels)]
        for _ in range(num_layers - 1):
            layers.append(self._conv_bn_relu(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            self._conv_bn_relu(in_channels, out_channels),
            self._conv_bn_relu(out_channels, out_channels),
        )

    def _conv_bn_relu(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="replicate",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _upsample(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return x
