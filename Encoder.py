import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=32, seq_length=50):
        super(Autoencoder, self).__init__()
        self.seq_length = seq_length
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * seq_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * seq_length)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), self.seq_length, -1), encoded

    def encode(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded.view(z.size(0), self.seq_length, -1)
    
    def get_latent_dim(self):
        return self.encoder[-1].out_features
    
    def get_input_dim(self):
        return self.encoder[0].in_features // self.seq_length
    
    def get_hidden_dim(self):
        return self.encoder[0].out_features
    

