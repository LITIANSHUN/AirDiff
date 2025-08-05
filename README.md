# UAV Trajectory Inpainting

A deep learning approach for UAV trajectory inpainting and generation using diffusion models.

## Overview

This project implements a trajectory recovery system for UAVs using a combination of autoencoders and diffusion models. Dataset wil be available if the paper could be accepted. The system can:

- Generate hybrid 6-DoF trajectory data
- Compress trajectories into a low-dimensional latent space
- Train a diffusion model to generate new trajectories
- Recover trajectories from noisy or incomplete data

## Architecture

The system consists of three main components:

1. **Autoencoder**: Compresses 3D trajectory data into a lower-dimensional latent space
2. **Diffusion Model**: Learns the trajectory distribution in latent space using a denoising diffusion probabilistic model
3. ***UAV dymaics model***: 6 DoF agent to deliver goods in urban areas with position control. Once arrive the spot, the policy will be triggered.

## Usage

python, simulink, Airsim

## Implementation

Python 3.10 + PyTorch 2.1, single RTX 4090 24 GB; UNet width 64→512, parameter count 42 M. Training batch size 64, initial learning rate 2e-4, cosine annealing to 2e-6.

## Dataset

From 2023 to 2025, 24332 drone flight paths were collected across five administrative districts in an anonymous city. After resampling at 10 Hz, the average length of the flight paths reached 112 waypoints. The data was split into training, validation, and test sets in a 6:2:2 ratio to ensure non-overlapping regions.

### Build / Run

bash

Copy

```bash
g++ -std=c++17 quadcopter_delivery.cpp -o quadcopter_delivery
./quadcopter_delivery
```

# Dataset Initialization

In training set of AeroTransNet dataset, **78%** were collected in ***real-world scenes***, while the rest were collceted via airsim and simulink. In test set of AeroTransNet dataset, **72%** were collected in ***real-world scenes***, while the rest were collceted via airsim and simulink.

data = generate_synthetic_data()
dataset = TrajectoryDataset(data)

# Train or load autoencoder

autoencoder = Autoencoder(latent_dim=32, input_dim=3, seq_length=50)

# (Set ae_train_flag to True for training, False for loading)

# Train or load diffusion model

diffusion_model = get_model(latent_dim=32)
scheduler = DDPMScheduler(num_train_timesteps=10)

# (Set diffusion_model_train_flag to True for training, False for loading)

# Sample a new trajectory

sampled_trajectory = sample_diffusion_trajectory(
    diffusion_model=diffusion_model,
    autoencoder=autoencoder,
    scheduler=scheduler,
    latent_dim=32,
    data=trajectory_data,
    num_steps=10
)

# Visualize the result

plot_3d_trajectories(original=original_trajectory, reconstructed=sampled_trajectory)