import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from diffusers import DDPMScheduler, UNet1DModel
from Encoder import Autoencoder
from Demo import TrajectoryDataset, generate_synthetic_data, plot_3d_trajectories
from Unet import get_model
from trainer import sample_diffusion_trajectory

class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_dim=3, d_model=64, nhead=8, num_layers=4, output_dim=3):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = self.input_embedding(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x


class AttentionModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super(AttentionModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        # Encode sequence
        encoder_outputs, _ = self.encoder(x)
        
        # Apply attention
        attn_output, _ = self.attention(encoder_outputs, encoder_outputs, encoder_outputs)
        
        # Decode with attention context
        decoder_output, _ = self.decoder(attn_output)
        
        # Project to output dimension
        output = self.fc(decoder_output)
        return output
    
    # ...existing code...

def train_lstm_model(model, dataloader, num_epochs=100, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], LSTM Model Loss: {total_loss/len(dataloader):.4f}')
    
    return model


def train_transformer_model(model, dataloader, num_epochs=100, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Transformer Model Loss: {total_loss/len(dataloader):.4f}')
    
    return model


def train_attention_model(model, dataloader, num_epochs=100, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Attention Model Loss: {total_loss/len(dataloader):.4f}')
    
    return model


def sample_trajectory(model, data, model_type='diffusion'):
    """
    Generate a trajectory sample using the selected model.
    
    Args:
        model: The trained model.
        data: The input trajectory data.
        model_type: The type of model ('diffusion', 'lstm', 'transformer', 'attention')
        
    Returns:
        A reconstructed trajectory in the original space.
    """
    device = next(model.parameters()).device
    
    if model_type == 'diffusion':
        # This function should call the original sample_diffusion_trajectory
        # with appropriate parameters
        raise NotImplementedError("Please use sample_diffusion_trajectory for diffusion models")
    else:
        with torch.no_grad():
            input_data = data.to(device)
            output = model(input_data)
            return output[0].cpu().numpy()
        

# Update the if __name__ == '__main__': section

if __name__ == '__main__':
    # Model selection - Choose one of: 'diffusion', 'lstm', 'transformer', 'attention'
    model_type = 'attention'  # Change this to select different model types
    
    ae_train_flag = False  # True for training, False for loading pre-trained model
    model_train_flag = True  # True for training, False for loading pre-trained model
    diffusion_steps = 10
    data = generate_synthetic_data()
    dataset = TrajectoryDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 先训练自编码器 (First train the autoencoder)
    if ae_train_flag:
        autoencoder = Autoencoder(latent_dim=32, input_dim=3, seq_length=50)
        autoencoder.train()
        train_autoencoder(autoencoder, dataloader)
        # 保存自编码器模型 (Save the autoencoder model)
        torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    else:
        # 加载自编码器模型 (Load the autoencoder model)
        autoencoder = Autoencoder(latent_dim=32, input_dim=3, seq_length=50)
        autoencoder.load_state_dict(torch.load('autoencoder.pth'))
        autoencoder.eval()
    
    # 输出自编码器的模型结构 (Output the autoencoder model structure)
    print(autoencoder)
    
    # 训练选择的模型 (Train the selected model)
    if model_type == 'diffusion':
        from Unet import get_model
        model = get_model(latent_dim=32)
        scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps)
        model_filename = 'diffusion_model.pth'
        
        if model_train_flag:
            model.train()
            train_diffusion_model(model, autoencoder, dataloader, scheduler, num_epochs=100, learning_rate=4e-4)
            torch.save(model.state_dict(), model_filename)
        else:
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            
        # Generate a sample trajectory
        sampled_trajectory = sample_diffusion_trajectory(
            diffusion_model=model,
            autoencoder=autoencoder,
            scheduler=scheduler,
            latent_dim=32,
            data=dataset[0].unsqueeze(0).to(next(model.parameters()).device),
            num_steps=diffusion_steps
        )
        
    elif model_type == 'lstm':

        model = LSTMModel(input_dim=3, hidden_dim=64, num_layers=2, output_dim=3)
        model_filename = 'lstm_model.pth'
        
        if model_train_flag:
            model.train()
            train_lstm_model(model, dataloader, num_epochs=100, learning_rate=1e-3)
            torch.save(model.state_dict(), model_filename)
        else:
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            
        # Generate a sample trajectory
        sampled_trajectory = sample_trajectory(
            model=model,
            data=dataset[0].unsqueeze(0),
            model_type='lstm'
        )
        
    elif model_type == 'transformer':
        # from models import TransformerModel
        model = TransformerModel(input_dim=3, d_model=64, nhead=8, num_layers=4, output_dim=3)
        model_filename = 'transformer_model.pth'
        
        if model_train_flag:
            model.train()
            train_transformer_model(model, dataloader, num_epochs=100, learning_rate=1e-3)
            torch.save(model.state_dict(), model_filename)
        else:
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            
        # Generate a sample trajectory
        sampled_trajectory = sample_trajectory(
            model=model,
            data=dataset[0].unsqueeze(0),
            model_type='transformer'
        )
        
    elif model_type == 'attention':
        # from models import AttentionModel
        model = AttentionModel(input_dim=3, hidden_dim=64, output_dim=3)
        model_filename = 'attention_model.pth'
        
        if model_train_flag:
            model.train()
            train_attention_model(model, dataloader, num_epochs=100, learning_rate=1e-3)
            torch.save(model.state_dict(), model_filename)
        else:
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            
        # Generate a sample trajectory
        sampled_trajectory = sample_trajectory(
            model=model,
            data=dataset[0].unsqueeze(0),
            model_type='attention'
        )
    
    # Plot the sampled trajectory
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    plot_3d_trajectories(
        original=dataset[0].numpy(),
        reconstructed=sampled_trajectory
    )