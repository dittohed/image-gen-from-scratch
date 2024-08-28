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
    

class UNet(nn.Module):
    """
    Simple U-Net with nearest-neighbors upsampling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # All the convs keep the spatial dims ('same' convs)
        self.convs_down = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2)
        ])
        self.convs_up = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(128, 32, kernel_size=5, padding=2),
            nn.Conv2d(64, out_channels, kernel_size=5, padding=2)
        ])

        self.downscale = nn.MaxPool2d(kernel_size=2)
        self.upscale = nn.Upsample(scale_factor=2)

        self.activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = []

        for i in range(2):
            x = self.convs_down[i](x)
            x = self.activation_fn(x)
            feature_maps.append(x)
            x = self.downscale(x)
        x = self.convs_down[2](x)

        for i in range(2):
            x = self.convs_up[i](x)
            x = self.activation_fn(x)
            x = self.upscale(x)
            x = torch.cat([x, feature_maps.pop()], dim=1)
        x = self.convs_up[2](x)

        return x


if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn((8, 3, 28, 28))
    print(model(x).shape)