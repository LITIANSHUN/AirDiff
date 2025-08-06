import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ====data ====

def create_trajectory_windows(data, window_size=10):
    segments = []
    vehicle_groups = data.groupby("VehicleName")
    for _, group in vehicle_groups:
        group = group.sort_values("TimeStamp")
        for i in range(len(group) - window_size + 1):
            segment = group.iloc[i:i + window_size][['POS_X', 'POS_Y', 'POS_Z']].values
            segments.append(segment)
    return np.array(segments)

def normalize_trajectories(segments):
    mean = np.mean(segments, axis=(0, 1), keepdims=True)
    std = np.std(segments, axis=(0, 1), keepdims=True) + 1e-8
    normalized = (segments - mean) / std
    return normalized, mean, std

def denormalize_trajectories(normalized_segments, mean, std):
    return normalized_segments * std + mean

# ====  Dataset ====

class TrajectoryDataset(Dataset):
    def __init__(self, segments, mask_ratio=0.3, fixed_mask=False):
        self.segments = segments.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.fixed_mask = fixed_mask

        if self.fixed_mask:
            self.masked_segments = []
            self.masks = []
            for seg in self.segments:
                masked, mask = self._mask_segment(seg)
                self.masked_segments.append(masked)
                self.masks.append(mask)
            self.masked_segments = np.array(self.masked_segments)
            self.masks = np.array(self.masks)

    def _mask_segment(self, segment):
        masked = np.copy(segment)
        mask = np.ones_like(segment)

        valid_idx = np.arange(1, segment.shape[0] - 1)
        num_mask = max(1, int(len(valid_idx) * self.mask_ratio))
        mask_idx = np.random.choice(valid_idx, num_mask, replace=False)

        for idx in mask_idx:
            masked[idx] = 0.0
            mask[idx] = 0.0
        return masked, mask

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        if self.fixed_mask:
            masked = self.masked_segments[idx]
            mask = self.masks[idx]
        else:
            masked, mask = self._mask_segment(segment)
        return (
            torch.tensor(masked, dtype=torch.float32),
            torch.tensor(segment, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )

# ==== 模型 ====

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

class CNN_Restore(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=64):
        super().__init__()
        self.name = 'CNN_Restore'
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.net(x)
        return out.permute(0, 2, 1)

# ==== Trainer ====

class Trainer:
    def __init__(self, model, train_loader, test_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=4e-3)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for X, y, mask in self.train_loader:
            X, y, mask = X.to(self.device), y.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)
            # loss = self.criterion(output * (1 - mask), y * (1 - mask))
            loss = self.criterion(output, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y, mask in self.test_loader:
                X, y, mask = X.to(self.device), y.to(self.device), mask.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

    def train(self, epochs=1000):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            test_loss = self.evaluate()
            if epoch % 100 == 0:
                print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        torch.save(self.model.state_dict(), f"model_epoch_{epoch}_{self.model.name}.pth")
        print(f"Model saved at epoch {epoch}")

# ====main ====

folder_path = "Data"
file_paths = glob.glob(os.path.join(folder_path, "*.csv"))
dfs = []
for idx, path in enumerate(file_paths):
    df = pd.read_csv(path)
    file_id = f"F{idx}"
    df["VehicleName"] = df["VehicleName"].apply(lambda x: f"{file_id}_{x}")
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data = all_data.dropna()

# prepare
window_size = 10
trajectory_segments = create_trajectory_windows(all_data, window_size)
trajectory_segments, traj_mean, traj_std = normalize_trajectories(trajectory_segments)

# Dataset & DataLoader
train_dataset = TrajectoryDataset(trajectory_segments, mask_ratio=0.3, fixed_mask=False)
test_dataset = TrajectoryDataset(trajectory_segments, mask_ratio=0.3, fixed_mask=True)

train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=32)

# Model selection
# model = MLP_Flattened(input_dim=window_size * 3, hidden_dim=256, output_dim=window_size*3)
model = CNN_Restore(in_channels=3, out_channels=3, hidden_channels=64)

# Whether to train
# train_flag = True
train_flag = False


if train_flag:
    trainer = Trainer(model, train_loader, test_loader)
    trainer.train(epochs=10000)
else:
    # Load pre-trained weights
    model.load_state_dict(torch.load(f"model_epoch_10000_{model.name}.pth"))
    print("Model loaded.")

    # Visualize a single segment
    idx = 4
    segment = trajectory_segments[idx]
    test_dataset = TrajectoryDataset(np.array([segment]), mask_ratio=0.3, fixed_mask=True)
    masked_input, target, mask = test_dataset[0]

    with torch.no_grad():
        pred = model(masked_input.unsqueeze(0)).squeeze(0).numpy()

    # De-normalize
    pred   = denormalize_trajectories(pred,           traj_mean, traj_std)[0]
    target = denormalize_trajectories(target.numpy(), traj_mean, traj_std)[0]
    masked_input = denormalize_trajectories(masked_input.numpy(), traj_mean, traj_std)[0]

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    known_idx   = mask[:, 0] == 1
    missing_idx = mask[:, 0] == 0

    ax.scatter(masked_input[known_idx, 0],   masked_input[known_idx, 1],   masked_input[known_idx, 2],   c='blue',  label='Known')
    ax.scatter(target[missing_idx, 0],       target[missing_idx, 1],       target[missing_idx, 2],       c='red',   label='Ground Truth', marker='x')
    ax.scatter(pred[missing_idx, 0],         pred[missing_idx, 1],         pred[missing_idx, 2],         c='green', label='Predicted',    marker='^')

    ax.plot(target[:, 0], target[:, 1], target[:, 2], color='black',  alpha=0.4, label='Original Trajectory')
    ax.plot(pred[:, 0],   pred[:, 1],   pred[:, 2],   color='orange', alpha=0.4, label='Predicted Trajectory')

    # Axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Axis limits (optional)
    ax.set_zlim([15, 25])

    ax.set_title("Trajectory Completion – " + model.name)
    ax.legend()
    plt.savefig("Plot/evaluate_trajectory_plot.png", dpi=300)
    plt.show()

    # Compute MSE over the entire test set
    test_dataset = TrajectoryDataset(trajectory_segments, mask_ratio=0.3, fixed_mask=True)
    test_loader  = DataLoader(test_dataset, batch_size=32)

    test_loss = 0.0
    with torch.no_grad():
        for X, y, mask in test_loader:
            X, y, mask = X.cpu(), y.cpu(), mask.cpu()
            output = model(X)
            loss   = nn.MSELoss()(output * (1 - mask), y * (1 - mask))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
