"""
Bidirectional GRU (Bi-GRU) Model for GNSS Spoofing Detection
=============================================================
Architecture:
  Input (seq_len, n_features)
    ↓
  Bidirectional GRU (Layer 1: 64 units per direction -> 128)
    ↓
  Dropout (0.2)
    ↓
  Bidirectional GRU (Layer 2: 32 units per direction -> 64)
    ↓
  Dropout (0.2)
    ↓
  Dense / Linear (32 units, ReLU)
    ↓
  Output Layer (1 unit, Sigmoid)

Hyperparameters configured in app.config.BIGRU_CONFIG.
Implemented with PyTorch for robust, production-grade training & inference.
"""
import torch
import torch.nn as nn
from app.config import BIGRU_CONFIG


class BiGRUNetwork(nn.Module):
    def __init__(self, sequence_length: int, n_features: int, cfg: dict = None):
        super().__init__()
        if cfg is None:
            cfg = BIGRU_CONFIG

        self.sequence_length = sequence_length
        self.n_features = n_features
        gru_units_1 = cfg.get("gru_units_1", 64)
        gru_units_2 = cfg.get("gru_units_2", 32)
        dropout_rate = cfg.get("dropout_rate", 0.2)
        dense_units = cfg.get("dense_units", 32)

        # Bi-GRU Layer 1: returns sequences
        self.bigru_1 = nn.GRU(
            input_size=n_features,
            hidden_size=gru_units_1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Bi-GRU Layer 2: take output of Bi-GRU 1 (2 * gru_units_1)
        self.bigru_2 = nn.GRU(
            input_size=gru_units_1 * 2,
            hidden_size=gru_units_2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_2 = nn.Dropout(dropout_rate)

        # Dense / Classification Head
        self.fc1 = nn.Linear(gru_units_2 * 2, dense_units)
        self.relu = nn.ReLU()
        self.out = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        out1, _ = self.bigru_1(x)
        out1 = self.dropout_1(out1)

        out2, _ = self.bigru_2(out1)
        # Take the last time step representation
        last_timestep = out2[:, -1, :]
        last_timestep = self.dropout_2(last_timestep)

        dense = self.relu(self.fc1(last_timestep))
        probs = self.sigmoid(self.out(dense))
        return probs


def build_bigru_model(sequence_length: int, n_features: int):
    """Factory function returning the instantiated BiGRUNetwork."""
    model = BiGRUNetwork(sequence_length=sequence_length, n_features=n_features)
    return model


def model_summary_str(model: nn.Module) -> str:
    """Return model parameter count and architecture string."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = (
        f"BiGRUNetwork Architecture:\n"
        f"{str(model)}\n"
        f"Total parameters: {total_params:,}\n"
        f"Trainable parameters: {trainable_params:,}"
    )
    return summary
