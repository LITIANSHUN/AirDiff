# UAV Trajectory Inpainting

A deep learning approach for UAV trajectory inpainting and generation using diffusion models.

## Overview

This project implements a trajectory recovery system for UAVs using a combination of autoencoders and diffusion models. The system can:

- Generate hybrid trajectory data
- Compress trajectories into a low-dimensional latent space
- Train a diffusion model to generate new trajectories
- Recover trajectories from noisy or incomplete data

## Architecture

The system consists of two main components:

1. **Autoencoder**: Compresses 3D trajectory data into a lower-dimensional latent space
2. **Diffusion Model**: Learns the trajectory distribution in latent space using a denoising diffusion probabilistic model

## Usage

python, simulink, Airsim

## Dataset

Dataset wil be available if the paper could be accepted.

# Generate hybrid data

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