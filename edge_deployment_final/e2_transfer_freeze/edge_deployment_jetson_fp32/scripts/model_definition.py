"""Standalone definition of the frozen C0 LSTM deployment architecture."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    input_sequence_length: int = 24
    input_dimension: int = 6
    hidden_dimension: int = 32
    num_layers: int = 1
    output_horizon: int = 24
    dropout: float = 0.0


class C0LSTM(nn.Module):
    """Frozen final_v1 C0 architecture: LSTM followed by a linear 24-hour head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_dimension,
            hidden_size=config.hidden_dimension,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
        )
        self.head = nn.Linear(config.hidden_dimension, config.output_horizon)

    def forward(self, x: Tensor) -> Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

