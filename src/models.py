import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Simple VAE with MLP encoder and decoder.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # Encoder
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)

        # Make optimization easier by predicting logarithm of variance
        # instead of variance directly (this way, we always get positive 
        # values for variance during training)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_to_hidden(x), inplace=True)
        mu, logvar = self.hidden_to_mu(x), self.hidden_to_logvar(x)

        return mu, logvar
    
    def decode(self, x: torch.Tensor, apply_sigmoid: bool = False) -> torch.Tensor:
        x = F.relu(self.latent_to_hidden(x), inplace=True)
        x = self.hidden_to_output(x)
        if apply_sigmoid:
            x = F.sigmoid(x)

        return x
    
    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        eps = torch.randn_like(logvar)

        # Here we really sample, needed for evidence
        latent_sampled = mu + torch.sqrt(torch.exp(logvar))*eps
        x_reconstr = self.decode(latent_sampled)

        return x_reconstr, latent_sampled, mu, logvar
    

if __name__ == '__main__':
    INPUT_DIM = 28*28
    HIDDEN_DIM = 256
    LATENT_DIM = 128
    BATCH_SIZE = 8

    model = VariationalAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    )
    x = torch.randn((BATCH_SIZE, INPUT_DIM))
    x_reconstr, latent_sampled, mu, logvar = model(x)

    print((x_reconstr.shape, latent_sampled.shape, mu.shape, logvar.shape))