# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar
import math 
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection
import torch.nn.functional as F

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        use_time_warping: bool = True, # Data Augmentation Methods
        use_amplitude_transform: bool = True, # Data Augmentation Methods
        use_spectrogram: bool = True, # Feature extraction Methods - Preprocessing
        use_wavelet: bool = True, # Feature extraction Methods - Preprocessing
        use_channel_selection: bool = True, # Dimensionality Reduction Methods - Preprocessing
        use_pca: bool = True, # Dimensionality Reduction Methods - Preprocessing
        
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

class CNNLSTMModule(pl.LightningModule):
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        in_features: int,
        cnn_channels: list = [32, 64, 128],
        kernel_size: int = 3,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        optimizer: dict = None,
        lr_scheduler: dict = None,
        decoder: dict = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        
        # Spectrogram normalization
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        
        # CNN Feature Extractor
        first_cnn_in_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS
        cnn_layers = []
        in_channels = first_cnn_in_channels

        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Ensure downsampling in time
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        
        # Adaptive pooling to remove frequency variation (fixed output features)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],  # CNN final channels
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output Projection
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, charset().num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, reduction="mean", zero_infinity=True)

        # Decoder
        self.decoder = instantiate(decoder) if decoder else None
        
        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        x = self.spec_norm(inputs)
        
        # Reshape for CNN: (T*N, bands*C, freq)
        T, N, bands, C, freq = x.shape
        x = x.reshape(T * N, bands * C, freq)

        
        # Apply CNN: (T*N, cnn_out_channels, reduced_freq)
        x = self.cnn(x)
        
        # Adaptive pooling to remove frequency variance: (T*N, cnn_out_channels, 1)
        x = self.adaptive_pool(x)
        
        # Reshape to (T, N, cnn_out_channels) for LSTM
        x = x.squeeze(-1).view(T, N, -1)
        
        # Apply LSTM: (T, N, lstm_hidden_size*2)
        x, _ = self.lstm(x)
        
        # Project to character space: (T, N, num_classes)
        x = self.projection(x)
        
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Since EMG signals may contain input before and after keystrokes,
        # we use full sequence length for emissions
        emission_lengths = torch.tensor([emissions.shape[0]] * N, device=emissions.device)

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        if self.decoder:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )

            # Update metrics
            metrics = self.metrics[f"{phase}_metrics"]
            targets = targets.detach().cpu().numpy()
            target_lengths = target_lengths.detach().cpu().numpy()
            for i in range(N):
                # Unpad targets (T, N) for batch entry
                target = LabelData.from_labels(targets[: target_lengths[i], i])
                metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss
    
    def _epoch_end(self, phase: str) -> None:
            metrics = self.metrics[f"{phase}_metrics"]
            self.log_dict(metrics.compute(), sync_dist=True)
            metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
        

class CNNGRUModule(pl.LightningModule):
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        in_features: int,
        cnn_channels: list = [32, 64, 128],
        kernel_size: int = 3,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        dropout: float = 0.2,
        optimizer: dict = None,
        lr_scheduler: dict = None,
        decoder: dict = None,
        adaptive_pool_size: int = 1
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        
        # Spectrogram normalization
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        
        # CNN Feature Extractor
        first_cnn_in_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS
        cnn_layers = []
        in_channels = first_cnn_in_channels

        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Ensure downsampling in time
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        
        # Adaptive pooling to remove frequency variation (fixed output features)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(adaptive_pool_size)

        self.gru_input_size = cnn_channels[-1] * adaptive_pool_size
        # LSTM for temporal modeling
        self.gru = nn.GRU(
            input_size=self.gru_input_size,  # CNN final channels
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=False,
            dropout=dropout if gru_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output Projection
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, charset().num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, reduction="mean", zero_infinity=True)

        # Decoder
        self.decoder = instantiate(decoder) if decoder else None
        
        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        x = self.spec_norm(inputs)
        
        # Reshape for CNN: (T*N, bands*C, freq)
        T, N, bands, C, freq = x.shape
        x = x.reshape(T * N, bands * C, freq)

        
        # Apply CNN: (T*N, cnn_out_channels, reduced_freq)
        x = self.cnn(x)
        
        # Adaptive pooling to remove frequency variance: (T*N, cnn_out_channels, 1)
        x = self.adaptive_pool(x)
        
        # Reshape to (T, N, cnn_out_channels) for LSTM
        x = x.squeeze(-1).view(T, N, -1)
        
        # Apply RNN: (T, N, rnn_hidden_size*2)
        x, _ = self.gru(x)
        
        # Project to character space: (T, N, num_classes)
        x = self.projection(x)
        
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Since EMG signals may contain input before and after keystrokes,
        # we use full sequence length for emissions
        emission_lengths = torch.tensor([emissions.shape[0]] * N, device=emissions.device)

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        if self.decoder:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )

            # Update metrics
            metrics = self.metrics[f"{phase}_metrics"]
            targets = targets.detach().cpu().numpy()
            target_lengths = target_lengths.detach().cpu().numpy()
            for i in range(N):
                # Unpad targets (T, N) for batch entry
                target = LabelData.from_labels(targets[: target_lengths[i], i])
                metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
        

class CNNTransformerModule(pl.LightningModule):
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        in_features: int,
        cnn_channels: list = [32, 64, 128],
        kernel_size: int = 3,
        transformer_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        optimizer: dict = None,
        lr_scheduler: dict = None,
        decoder: dict = None,
        max_seq_length: int = 10000,  # Increased for long sequences
        chunk_size: int = 512,        # Size of chunks for processing long sequences
        chunk_overlap: int = 128,     # Overlap between chunks to maintain context
        use_time_warping: bool = True, # Data Augmentation Methods
        use_amplitude_transform: bool = True, # Data Augmentation Methods
        use_channel_selection: bool = True, # Dimensionality Reduction Methods - Preprocessing
        use_frequency_reduction: bool = True, # Dimensionality Reduction Methods - Preprocessing
        channels_to_keep: int = 8, # Dimensionality Reduction Methods - Preprocessing
        freq_components: int = 32, # Dimensionality Reduction Methods - Preprocessing
        
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        
        # Spectrogram normalization
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        # CNN Feature Extractor
        first_cnn_in_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS
        cnn_layers = []
        in_channels = first_cnn_in_channels

        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Ensure downsampling in time
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)
        
        # Adaptive pooling to remove frequency variation (fixed output features)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Feature dimension from CNN
        cnn_feature_dim = cnn_channels[-1]
        
        # Input projection to transformer dimension
        self.input_projection = nn.Linear(cnn_feature_dim, transformer_dim)
        
        # Positional Encoding for Transformer
        self.pos_encoder = PositionalEncoding(
            d_model=transformer_dim,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # Transformer Encoder for temporal modeling (replaces LSTM)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=True,  # Pre-norm architecture for better training stability
            activation=F.gelu  # GELU activation often works better for transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Memory-efficient attention when testing with long sequences
        self.use_memory_efficient_attention = True  # Can be toggled via config
        
        # Output Projection
        self.projection = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, charset().num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, reduction="mean", zero_infinity=True)

        # Decoder
        self.decoder = instantiate(decoder) if decoder else None
        
        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        x = self.spec_norm(inputs)
        
        # Reshape for CNN: (T*N, bands*C, freq)
        T, N, bands, C, freq = x.shape
        x = x.reshape(T * N, bands * C, freq)
        
        # Apply CNN: (T*N, cnn_out_channels, reduced_freq)
        x = self.cnn(x)
        
        # Adaptive pooling to remove frequency variance: (T*N, cnn_out_channels, 1)
        x = self.adaptive_pool(x)
        
        # Reshape to (T, N, cnn_out_channels) for Transformer
        x = x.squeeze(-1).view(T, N, -1)
        
        # Project to transformer dimension
        x = self.input_projection(x)
        
        # Process long sequences in chunks if needed
        if self.training or T <= self.hparams.chunk_size:
            # Add positional encoding - regular processing
            x = self.pos_encoder(x)
            
            # Apply Transformer: (T, N, transformer_dim)
            x = self.transformer_encoder(x)
        else:
            # Handle long sequence with chunking approach
            x = self._process_long_sequence(x)
        
        # Project to character space: (T, N, num_classes)
        x = self.projection(x)
        
        return x
        
    def _process_long_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Process long sequences by chunking them into smaller segments with overlap."""
        T, N, C = x.shape
        chunk_size = self.hparams.chunk_size
        overlap = self.hparams.chunk_overlap
        stride = chunk_size - overlap
        
        # Compute how many chunks we need
        n_chunks = max(1, math.ceil((T - overlap) / stride))
        
        # Initialize output tensor
        out = torch.zeros_like(x)
        
        # Keep track of overlap counts for proper averaging
        overlap_counts = torch.zeros((T, N, 1), device=x.device)
        
        for i in range(n_chunks):
            # Determine current chunk boundaries
            start_idx = i * stride
            end_idx = min(start_idx + chunk_size, T)
            
            # Extract chunk
            chunk = x[start_idx:end_idx]
            
            # Process chunk with positional encoding
            # Note: We use modulo to handle positional encoding for very long sequences
            # that exceed max_seq_length
            chunk_pos = self.pos_encoder(chunk)
            
            # Process with transformer
            chunk_out = self.transformer_encoder(chunk_pos)
            
            # Add processed chunk back to the output tensor
            out[start_idx:end_idx] += chunk_out
            overlap_counts[start_idx:end_idx] += 1
        
        # Average the output in overlapped regions
        out = out / overlap_counts
        
        return out

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Since EMG signals may contain input before and after keystrokes,
        # we use full sequence length for emissions
        emission_lengths = torch.tensor([emissions.shape[0]] * N, device=emissions.device)

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        if self.decoder:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )

            # Update metrics
            metrics = self.metrics[f"{phase}_metrics"]
            targets = targets.detach().cpu().numpy()
            target_lengths = target_lengths.detach().cpu().numpy()
            for i in range(N):
                # Unpad targets (T, N) for batch entry
                target = LabelData.from_labels(targets[: target_lengths[i], i])
                metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss
    
    def _epoch_end(self, phase: str) -> None:
            metrics = self.metrics[f"{phase}_metrics"]
            self.log_dict(metrics.compute(), sync_dist=True)
            metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# Positional Encoding class needed for Transformer
class PositionalEncoding(nn.Module):
    """
    Inject information about the position of tokens in the sequence
    Implementation based on "Attention Is All You Need" paper
    with modifications for handling longer sequences
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize positional encoding
        pe = torch.zeros(max_len, 1, d_model)
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter but part of the module)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            offset: Optional position offset for chunked processing
        """
        seq_len = x.size(0)
        
        # Handle case where sequence length exceeds max_len
        if offset + seq_len > self.max_len:
            # Use modulo arithmetic for position encoding
            positions = torch.arange(offset, offset + seq_len) % self.max_len
            x = x + self.pe[positions]
        else:
            # Standard case - directly use pre-computed positions
            x = x + self.pe[offset:offset + seq_len]
            
        return self.dropout(x)

class RNNModule(pl.LightningModule):
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(
        self,
        in_features: int,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 3,
        dropout: float = 0.2,
        optimizer: dict = None,
        lr_scheduler: dict = None,
        decoder: dict = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Spectrogram normalization
        self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        
        # RNN for temporal processing
        rnn_input_size = self.NUM_BANDS * self.ELECTRODE_CHANNELS * in_features
        self.rnn = nn.RNN(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=False,  # (T, N, features)
            dropout=dropout if rnn_num_layers > 1 else 0,
            bidirectional=True,
            nonlinearity='tanh'
        )
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, charset().num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        
        # Decoder
        self.decoder = instantiate(decoder) if decoder else None
        
        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Normalize spectrogram
        x = self.spec_norm(inputs)
        
        # Reshape for RNN: (T, N, features)
        T, N, bands, C, freq = x.shape
        x = x.view(T, N, -1)
        
        # Apply RNN
        x, _ = self.rnn(x)
        
        # Project to character space
        x = self.projection(x)
        
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        emission_lengths = torch.tensor([emissions.shape[0]] * N, device=emissions.device)

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        # Decode emissions
        if self.decoder:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )
            
            metrics = self.metrics[f"{phase}_metrics"]
            targets = targets.detach().cpu().numpy()
            target_lengths = target_lengths.detach().cpu().numpy()
            for i in range(N):
                target = LabelData.from_labels(targets[: target_lengths[i], i])
                metrics.update(prediction=predictions[i], target=target)
        
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        return self._step("train", batch, *args, **kwargs)

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        return self._step("val", batch, *args, **kwargs)

    def test_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        return self._step("test", batch, *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
