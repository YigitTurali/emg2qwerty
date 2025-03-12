# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F
import kymatio
import numpy as np

class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class ResidualBlock(nn.Module):
    """Residual block for CNN with skip connection"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        x = self.relu(x)
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        T, N, bands, C, freq = x.shape  # (Time, Batch, Bands, Channels, Frequency)
        attn = self.fc1(x.mean(dim=4))  # Global pooling over frequency
        attn = self.fc2(F.relu(attn))
        attn = self.sigmoid(attn).unsqueeze(4)  # Reshape for broadcasting
        return x * attn  # Apply attention


class WaveletTransform(nn.Module):
    def __init__(self, J=5):
        super().__init__()
        self.J = J
        self.wavelet = None  # Initialize wavelet transform as None, set dynamically in forward

    def forward(self, x):
        T, N, bands, C, freq = x.shape  # (Time, Batch, Bands, Channels, Frequency)
        
        # Ensure wavelet transform is initialized with the correct frequency shape
        if self.wavelet is None:
            self.wavelet = kymatio.Scattering1D(J=self.J, shape=(freq,), frontend='torch').to(x.device)
        
        # Ensure x is in (batch, time) format before wavelet processing
        x = x.reshape(T * N * bands * C, freq).contiguous()
        
        # Apply wavelet transform
        x = self.wavelet(x)

        # Restore original shape
        return x.view(T, N, bands, C, -1)
    
class SincConv1D(nn.Module):
    def __init__(self, out_channels, kernel_size=101, sample_rate=2000, min_low_hz=20, min_band_hz=50, max_freq=850):
        super(SincConv1D, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.max_freq = max_freq

        # Learnable frequency bands
        self.low_hz = nn.Parameter(torch.rand(out_channels) * (max_freq - min_low_hz) + min_low_hz)
        self.band_hz = nn.Parameter(torch.rand(out_channels) * (max_freq - min_band_hz) + min_band_hz)

        # Hamming window
        n_lin = torch.linspace(0, kernel_size, steps=kernel_size)
        self.window = 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / kernel_size)

        self.n = (kernel_size - 1) / 2

    def sinc(self, x):
        return torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x)

    def forward(self, x):
        T, N, bands, C, freq = x.shape
        x = x.reshape(T * N * bands, C, freq)  

        # Compute frequency filters
        low = self.min_low_hz + torch.abs(self.low_hz)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz), self.min_low_hz, self.max_freq)

        band_pass_left = (2 * high[:, None] * self.sinc(2 * high[:, None] * (self.n - self.n))) - \
                         (2 * low[:, None] * self.sinc(2 * low[:, None] * (self.n - self.n)))
        band_pass_left = band_pass_left * self.window

        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = band_pass_left + band_pass_right

        x = F.conv1d(x, band_pass.view(self.out_channels, 1, self.kernel_size), stride=1, padding=self.kernel_size // 2, groups=C)

        return x.view(T, N, bands, self.out_channels, -1)

def rotate_electrodes(x):
    """
    Randomly shifts electrode positions by -1, 0, or +1.
    Args:
        x: Tensor of shape (T, N, bands, C, freq)
    Returns:
        Tensor with rotated channels
    """
    shift = torch.randint(-1, 2, (1,)).item()  # Shift by -1, 0, or +1
    return torch.roll(x, shifts=shift, dims=3)  # Rotate along channel axis (C)


def time_warp(x, warp_factor_range=(0.9, 1.1), p=0.5):
    """
    Applies non-linear time warping to simulate variations in typing speed.
    
    Args:
        x: Tensor of shape (T, N, bands, C, freq)
        warp_factor_range: Tuple defining min/max time stretch factors.
        p: Probability of applying time warping.

    Returns:
        Warped tensor of same shape.
    """
    if torch.rand(1).item() > p:
        return x  # Skip augmentation with probability (1 - p)
    
    T, N, bands, C, freq = x.shape
    device = x.device
    
    # Generate a random warping factor for each batch sample
    warp_factors = torch.rand(N, device=device) * (warp_factor_range[1] - warp_factor_range[0]) + warp_factor_range[0]
    
    # Prepare output tensor
    output = torch.zeros_like(x)
    
    for b in range(N):
        factor = warp_factors[b].item()
        src_time = torch.arange(T, dtype=torch.float32, device=device)
        
        if factor < 1.0:
            # Compress time
            target_t = int(T * factor)
            dst_time = torch.linspace(0, T-1, target_t, device=device)
            warped = F.interpolate(
                x[:, b].reshape(T, -1).permute(1, 0).unsqueeze(0),
                size=target_t,
                mode='linear',
                align_corners=True
            )
            warped = warped.squeeze(0).permute(1, 0)
            padding = torch.zeros(T - target_t, bands * C * freq, device=device)
            warped = torch.cat([warped, padding], dim=0)
        
        else:
            # Stretch time
            target_t = int(T * factor)
            dst_time = torch.linspace(0, target_t - 1, T, device=device)
            temp = F.interpolate(
                x[:, b].reshape(T, -1).permute(1, 0).unsqueeze(0),
                size=target_t,
                mode='linear',
                align_corners=True
            )
            indices = torch.linspace(0, target_t-1, T, device=device).long()
            warped = temp.squeeze(0).permute(1, 0)[indices]

        # Reshape back
        output[:, b] = warped.reshape(T, bands, C, freq)

    return output