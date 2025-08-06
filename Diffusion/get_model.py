
import torch.nn as nn
import torch
# ==== Model ====
# The get_model function instantiates and returns a model based on the provided model_name.
# Currently supported models: MLP_Flattened, CNN_Restore, LSTM_Restore, and Transformer_Restore.


def get_model(name: str):
    name = name.lower()
    if name == "mlp":
        return MLP_Flattened(input_dim=30)
    elif name == "cnn":
        return CNN_Restore()
    elif name == "lstm":
        return LSTM_Restore()
    elif name == "transformer":
        return Transformer_Restore()
    else:
        raise ValueError(f"Unknown model type: {name}")
    


class MLP_Flattened(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=256, output_dim=30):
        super().__init__()
        self.name = 'MLP_Flattened'
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.mlp(x)
        return out.view(x.size(0), 10, 3)

import torch.nn as nn

class CNN_Restore(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=64):
        super().__init__()
        self.name = 'CNN_Restore'
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.net(x)
        return out.permute(0, 2, 1)

import torch.nn as nn

class LSTM_Restore(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, output_dim=3):
        super().__init__()
        self.name = "LSTM_Restore"
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

import torch
import torch.nn as nn

class Transformer_Restore(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.name = "Transformer_Restore"
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        out = self.transformer(x)
        return self.output_layer(out)
