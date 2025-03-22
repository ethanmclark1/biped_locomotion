import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from humanoid_loco.scripts.utils.adam_wr.adam_w import AdamW
from humanoid_loco.scripts.utils.adam_wr.cyclic_scheduler import (
    CyclicLRWithRestarts,
)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class PAE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        intermediate_channels: int,
        embedding_channels: int,
        time_range: float,
        window: int,
        batch_size: int,
        config: dict,
        sample_count: int = None,
    ) -> None:

        super(PAE, self).__init__()

        self.name = self.__class__.__name__

        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range

        learning_rate = config[self.name]["learning_rate"]
        weight_decay = config[self.name]["weight_decay"]
        restart_period = config[self.name]["restart_period"]
        t_mult = config[self.name]["t_mult"]

        self.two_pi = nn.Parameter(
            torch.from_numpy(np.array([2.0 * np.pi], dtype=np.float32)),
            requires_grad=False,
        )
        self.args = nn.Parameter(
            torch.from_numpy(
                np.linspace(-window / 2, window / 2, self.time_range, dtype=np.float32)
            ),
            requires_grad=False,
        )
        self.freqs = nn.Parameter(
            torch.fft.rfftfreq(time_range)[1:] * time_range / window,
            requires_grad=False,
        )  # Remove DC frequency

        self.conv1 = nn.Conv1d(
            input_channels,
            intermediate_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.norm1 = LayerNorm(time_range)
        self.conv2 = nn.Conv1d(
            intermediate_channels,
            embedding_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.fc = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))

        self.deconv1 = nn.Conv1d(
            embedding_channels,
            intermediate_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.denorm1 = LayerNorm(time_range)
        self.deconv2 = nn.Conv1d(
            intermediate_channels,
            input_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.loss_fn = nn.MSELoss()

        self.optimizer = AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        if sample_count:
            self.scheduler = CyclicLRWithRestarts(
                self.optimizer,
                batch_size,
                sample_count,
                restart_period,
                t_mult,
                policy="cosine",
                verbose=True,
            )

        self.train()

    # Fast-Fourier Transform to extract frequency, amplitude, and offset
    def FFT(self, function: torch.Tensor, dim: int) -> tuple:
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
        power = spectrum**2

        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        offset = rfft.real[:, :, 0] / self.time_range  # DC component

        return freq, amp, offset

    def forward(self, x: torch.Tensor) -> tuple:
        # Signal Embedding
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.elu(x)

        latent = self.conv2(x)

        freq, amp, offset = self.FFT(latent, dim=2)

        shift = torch.empty(
            (latent.shape[0], self.embedding_channels),
            dtype=torch.float32,
            device=latent.device,
        )
        for i in range(self.embedding_channels):
            v = self.fc[i](latent[:, i, :])
            shift[:, i] = torch.atan2(v[:, 1], v[:, 0]) / self.two_pi

        # Parameters
        shift = shift.unsqueeze(2)
        freq = freq.unsqueeze(2)
        amp = amp.unsqueeze(2)
        offset = offset.unsqueeze(2)
        parameters = [shift, freq, amp, offset]

        # Latent Reconstruction
        signal = amp * torch.sin(self.two_pi * (freq * self.args + shift)) + offset

        # Signal Reconstruction
        reconstructed = self.deconv1(signal)
        reconstructed = self.denorm1(reconstructed)
        reconstructed = F.elu(reconstructed)

        reconstructed = self.deconv2(reconstructed)

        return reconstructed, parameters
