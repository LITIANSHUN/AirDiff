import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from diffusers import DDPMScheduler, UNet1DModel
from Encoder import Autoencoder
from Demo import TrajectoryDataset, generate_synthetic_data, plot_3d_trajectories
from Unet import get_model


def train_diffusion_model(diffusion_model, autoencoder, dataloader, scheduler, num_epochs=50, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)
    predict_noise = True  # Set to False if you want to predict the original image instead of noise

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            with torch.no_grad():
                z = autoencoder.encode(batch)
                z = z.view(z.size(0), 1, -1)  # shape: [batch, 1, 32]
                # print(batch.shape)
                # print(z.shape)
                # exit()

            noise = torch.randn_like(z)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (z.size(0),), device=z.device).long()
            noisy_latents = scheduler.add_noise(z, noise, timesteps)

            optimizer.zero_grad()
            # noise_pred = diffusion_model(noisy_latents, timesteps, class_labels=z).sample
            noise_pred = diffusion_model(noisy_latents, timesteps, encoder_hidden_states=z).sample
            #  model(noisy_latents, timesteps, class_labels=z).sample
            # noise_pred = noise_pred.view(noise_pred.size(0), 1, -1)  # shape: [batch, 1, 32]
            # print(noise_pred.shape)
            # print(noise.shape)
            # exit()
            # if predict_noise:
            #     # Predict the noise
            #     noise_pred = diffusion_model(noisy_latents, timesteps).sample
            # else:
            #     # Predict the original image
            #     noise_pred = diffusion_model(noisy_latents, timesteps).prev_sample

            loss = criterion(noise_pred, noise)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Diffusion Model Loss: {total_loss/len(dataloader):.4f}')


def train_autoencoder(autoencoder, dataloader, num_epochs=5000, learning_rate=2e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs, _ = autoencoder(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Autoencoder Loss: {total_loss/len(dataloader):.4f}')


def sample_diffusion_trajectory(diffusion_model, autoencoder, scheduler, latent_dim, data, num_steps=100):
    """
    Generate a trajectory sample using the diffusion model.

    Args:
        diffusion_model: The trained diffusion model.
        autoencoder: The trained autoencoder for encoding/decoding trajectories.
        scheduler: The diffusion scheduler (e.g., DDPMScheduler).
        latent_dim: The dimensionality of the latent space.
        data: The input trajectory data.
        num_steps: The number of diffusion steps.

    Returns:
        A reconstructed trajectory in the original space.
    """
    device = next(diffusion_model.parameters()).device
    
    # Encode the input data first to match the expected format
    with torch.no_grad():
        encoded_data = autoencoder.encode(data)
        encoded_data = encoded_data.view(encoded_data.size(0), 1, -1)  # Shape: [batch, 1, latent_dim]
    
    # Start with random noise in the latent space
    z = torch.randn(1, 1, latent_dim).to(device)  # Shape: [1, 1, latent_dim]

    # Iteratively denoise the latent representation
    for t in reversed(range(num_steps)):
        t_tensor = torch.tensor([t], device=device).long()
        with torch.no_grad():
            # Predict the noise at the current timestep using encoded data
            noise_pred = diffusion_model(z, t_tensor, encoder_hidden_states=encoded_data).sample

        # Remove noise using the scheduler
        z = scheduler.step(noise_pred, t_tensor, z).prev_sample

    # Decode the denoised latent representation back to the original trajectory space
    with torch.no_grad():
        reconstructed_trajectory = autoencoder.decode(z.view(1, latent_dim)).cpu().numpy()

    return reconstructed_trajectory[0]  # Return the trajectory (shape: [seq_length, 3])

if __name__ == '__main__':
    ae_train_flag = False  # True for training, False for loading pre-trained model
    diffusion_model_train_flag = False  # True for training, False for loading pre-trained model
    diffusion_steps = 10
    data = generate_synthetic_data()
    dataset = TrajectoryDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # train
    if ae_train_flag:
        autoencoder = Autoencoder(latent_dim = 32, input_dim=3, seq_length=50)
        autoencoder.train()
        train_autoencoder(autoencoder, dataloader)
        # save
        torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    else:
        # load VAE
        autoencoder = Autoencoder(latent_dim = 32, input_dim=3, seq_length=50)
        autoencoder.load_state_dict(torch.load('autoencoder.pth'))
        autoencoder.eval()
    # output
    print(autoencoder)
    # 训练扩散模型
    diffusion_model = get_model(latent_dim = 32)
    scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps)

    if diffusion_model_train_flag:
        diffusion_model.train()

        train_diffusion_model(diffusion_model, autoencoder, dataloader, scheduler,
                              num_epochs=100, learning_rate= 4e-4)
        # save
        torch.save(diffusion_model.state_dict(), 'diffusion_model.pth')
    else:
        # upload
        diffusion_model.load_state_dict(torch.load('diffusion_model.pth'))
        diffusion_model.eval()
        # Generate a sample trajectory
        sampled_trajectory = sample_diffusion_trajectory(
            diffusion_model = diffusion_model,
            autoencoder = autoencoder,
            scheduler = scheduler,
            latent_dim = 32,
            data = dataset[0].unsqueeze(0).to(next(diffusion_model.parameters()).device),
            num_steps = diffusion_steps
        )

        # Plot the sampled trajectory
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D


        plot_3d_trajectories(
            original=dataset[0].numpy(),
            reconstructed=sampled_trajectory
        )
        # Plot the original vs reconstructed trajectory
        # plot_3d_trajectories(dataset[0].numpy(), sampled_trajectory)