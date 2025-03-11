# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import CubicSpline

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

class TimeWarpingLayer(nn.Module):
    """
    Time warping with smooth spline interpolation.
    """
    def __init__(self, warp_factor_range=(0.9, 1.1), p=0.5):
        super().__init__()
        self.warp_factor_range = warp_factor_range
        self.p = p
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return inputs
        
        T, N, bands, C, freq = inputs.shape
        device = inputs.device
        warp_factors = torch.rand(N, device=device) * (self.warp_factor_range[1] - self.warp_factor_range[0]) + self.warp_factor_range[0]
        output = torch.zeros_like(inputs)
        
        for b in range(N):
            factor = warp_factors[b].item()
            src_time = np.arange(T)
            dst_time = np.linspace(0, T - 1, int(T * factor))
            
            for ch in range(bands * C * freq):
                x_reshaped = inputs[:, b].reshape(T, -1)[:, ch].cpu().numpy()
                cs = CubicSpline(src_time, x_reshaped)
                warped_signal = cs(dst_time)
                
                if len(warped_signal) < T:
                    padding = np.zeros(T - len(warped_signal))
                    warped_signal = np.concatenate([warped_signal, padding])
                
                warped_signal_tensor = torch.tensor(warped_signal, device=device)
                if warped_signal_tensor.shape[0] > T:
                    warped_signal_tensor = warped_signal_tensor[:T]  # Truncate if longer
                elif warped_signal_tensor.shape[0] < T:
                    padding = torch.zeros(T - warped_signal_tensor.shape[0], device=device)
                    warped_signal_tensor = torch.cat([warped_signal_tensor, padding])  # Pad if shorter
                output[:, b].reshape(T, -1)[:, ch] = warped_signal_tensor
        
        return output


class AmplitudeTransformLayer(nn.Module):
    """
    Amplitude scaling and noise augmentation.
    """
    def __init__(self, scale_range=(0.8, 1.2), noise_std=0.02, p=0.5):
        super().__init__()
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.p = p
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return inputs
        
        T, N, bands, C, freq = inputs.shape
        device = inputs.device
        
        scales = torch.rand(N, bands, C, 1, device=device) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        output = inputs * scales.unsqueeze(0)
        
        if torch.rand(1).item() > 0.5:
            noise_level = torch.rand(1, device=device) * self.noise_std
            noise = torch.randn_like(output) * noise_level
            output = output + noise
            
        return output


class SoftChannelSelectionLayer(nn.Module):
    """
    Soft attention-based channel selection.
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.channel_weights = nn.Parameter(torch.randn(input_channels, output_channels))
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape
        device = inputs.device
        
        attention_weights = F.softmax(self.channel_weights, dim=0)  # Soft selection
        inputs_reshaped = inputs.view(T, N, bands, C * freq)
        selected = torch.einsum('cf,tbnc->tbnf', attention_weights, inputs_reshaped)
        return selected.view(T, N, bands, self.output_channels, freq)


class SpectralReductionLayer(nn.Module):
    """
    Learnable spectral transformation instead of PCA.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape
        x = inputs.permute(0, 1, 2, 3, 4).reshape(T * N * bands * C, freq)
        x = self.conv(x.unsqueeze(1)).squeeze(1)  # Apply learnable transformation
        return x.view(T, N, bands, C, -1)


class EMGPreprocessingPipeline(nn.Module):
    def __init__(self, bands=2, channels=16, use_time_warping=True, use_amplitude_transform=True,
                 use_channel_selection=True, use_frequency_reduction=True,
                 channels_to_keep=8, freq_components=32):
        super().__init__()
        self.bands = bands
        self.channels = channels
        layers = []
        
        if use_time_warping:
            layers.append(('time_warp', TimeWarpingLayer()))
        if use_amplitude_transform:
            layers.append(('amplitude', AmplitudeTransformLayer()))
        if use_channel_selection:
            layers.append(('channel_select', SoftChannelSelectionLayer(channels, channels_to_keep)))
            self.channels = channels_to_keep
        if use_frequency_reduction:
            layers.append(('freq_reduce', SpectralReductionLayer(freq_components, freq_components // 2)))
            
        self.layers = nn.ModuleDict(dict(layers))
        self.layer_order = [name for name, _ in layers]
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for name in self.layer_order:
            if name == 'freq_reduce':
                continue
            x = self.layers[name](x)
        return x
    
    def process_spectrograms(self, spectrograms: torch.Tensor) -> torch.Tensor:
        if 'freq_reduce' in self.layers:
            return self.layers['freq_reduce'](spectrograms)
        return spectrograms