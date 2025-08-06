import numpy as np
import torch
from torch.utils.data import Dataset
from Encoder import Autoencoder

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def generate_synthetic_data(num_samples=1000, seq_length=50):
    """
    Generate synthetic 3-D trajectories.
    Each trajectory is a (seq_length, 3) array containing
    noisy sine, cosine, and linear components.
    """
    data = []
    for _ in range(num_samples):
        t = np.linspace(0, 1, seq_length)
        x = np.sin(2 * np.pi * t) + np.random.normal(0, 0.01, seq_length)
        y = np.cos(2 * np.pi * t) + np.random.normal(0, 0.01, seq_length)
        z = t + np.random.normal(0, 0.01, seq_length)
        trajectory = np.stack((x, y, z), axis=1)
        data.append(trajectory)
    return np.array(data)

# Ground-truth trajectory
# Input: incomplete original trajectory with 10 % random masking
# Output: reconstructed trajectory to compare against ground truth

# Generate data
data = generate_synthetic_data()
dataset = TrajectoryDataset(data)

print(len(dataset))                    # 1000
print(dataset[0].shape)                # torch.Size([50, 3])
print(dataset[0][0])                   # tensor([ 0.0000,  1.0000, -0.0136])
print(dataset[0][0][0])                # tensor(0.0000)
print(dataset[0][0][0].item())         # 0.0
print(dataset[0][0][0].item() == 0.0)  # True

# Plot a sample trajectory
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_3d_trajectories(original, reconstructed):
    """
    Visualize original and reconstructed 3-D trajectories.
    Solid line = original; dashed line = reconstructed.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Plot original trajectory (solid)
    colors_original = cm.viridis(np.linspace(0, 1, len(original)))
    for i in range(len(original) - 1):
        ax.plot([original[i][0], original[i + 1][0]],
                [original[i][1], original[i + 1][1]],
                [original[i][2], original[i + 1][2]],
                color=colors_original[i], linestyle='-', label='Original' if i == 0 else "")

    # Plot reconstructed trajectory (dashed)
    colors_reconstructed = cm.plasma(np.linspace(0, 1, len(reconstructed)))
    for i in range(len(reconstructed) - 1):
        ax.plot([reconstructed[i][0], reconstructed[i + 1][0]],
                [reconstructed[i][1], reconstructed[i + 1][1]],
                [reconstructed[i][2], reconstructed[i + 1][2]],
                color=colors_reconstructed[i], linestyle='--', label='Reconstructed' if i == 0 else "")

    plt.title('Original vs Reconstructed 3D Trajectories')
    ax.legend()
    plt.savefig('original_vs_reconstructed_3d_trajectory.png')
    plt.show()


if __name__ == '__main__':
    # Plot the first trajectory
    # plot_3d_trajectory(dataset[0].numpy())

    # Load pre-trained autoencoder if encoder.py and unet.py exist
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('autoencoder.pth'))
    autoencoder.eval()

    # Encode and decode the first trajectory
    encoded = autoencoder.encode(dataset[0].unsqueeze(0))
    print(encoded.shape)  # (1, 32)

    reconstructed_trajectory = autoencoder.decode(encoded).detach().numpy()[0]
    print(reconstructed_trajectory.shape)  # (50, 3)

    # Plot both trajectories
    original_trajectory = dataset[0].numpy()
    plot_3d_trajectories(original_trajectory, reconstructed_trajectory)
