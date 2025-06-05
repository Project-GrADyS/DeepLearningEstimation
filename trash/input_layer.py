
import torch
import torch.nn as nn
import numpy as np

# --- TemporalSummary ---
class TemporalSummary(nn.Module):
    def __init__(self, time_steps: int = 5858, num_outputs: int = 10, mode: str = 'linear', min_ratio: float = 0.1):
        super().__init__()
        self.time_steps = time_steps
        self.num_outputs = num_outputs
        self.mode = mode.lower()
        self.min_ratio = min_ratio

        x = np.arange(1, num_outputs + 1)

        if self.mode == 'short':
            alpha_base = 1 / x
        elif self.mode == 'linear':
            alpha_base = 1 - (x - 1) / num_outputs
        elif self.mode == 'long':
            alpha_base = 1 + 1 / num_outputs - 1 / (num_outputs - x + 1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        alpha_norm = (alpha_base - alpha_base.min()) / (alpha_base.max() - alpha_base.min())
        alpha_scaled = self.min_ratio + (1.0 - self.min_ratio) * alpha_norm

        self.window_sizes = [max(1, int(round(time_steps * a))) for a in alpha_scaled]

    def forward(self, x):
        outputs = []
        for window_size in self.window_sizes:
            slice_ = x[:, :, -window_size:]
            avg = slice_.mean(dim=2)
            outputs.append(avg)
        return torch.cat(outputs, dim=1)

# --- InputLayer ---
class InputLayer(nn.Module):
    def __init__(self, time_steps, num_outputs, mode='linear'):
        super().__init__()
        self.shape_summary = TemporalSummary(time_steps, num_outputs, mode)
        self.presence_summary = TemporalSummary(time_steps, num_outputs, mode)
        self.leader_summary = TemporalSummary(time_steps, num_outputs, mode)

    def forward(self, x_static, x_shape, x_presence, x_leader):
        s1 = self.shape_summary(x_shape)
        s2 = self.presence_summary(x_presence)
        s3 = self.leader_summary(x_leader)
        return torch.cat([x_static, s1, s2, s3], dim=1)
