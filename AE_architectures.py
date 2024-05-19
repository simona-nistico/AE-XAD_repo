import torch
import torch.nn as nn
import numpy as np

class Shallow_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, latent_dim):
        super(Shallow_Autoencoder, self).__init__()
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


class Deep_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, intermediate_dim, latent_dim):
        super(Deep_Autoencoder, self).__init__()
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, flat_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


class Conv_Deep_Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Conv_Deep_Autoencoder, self).__init__()
        self.dim = np.array(dim)
        print(dim)

        layers = []
        diffs = []

        layers.append(nn.Conv2d(dim[0], 16, (5, 5), stride=1, padding=2))
        mods = np.remainder(-self.dim[1:], 4)
        diff = np.array([mods[-2] // 2, mods[-2] - mods[-2] // 2, mods[-1] // 2, mods[-1] - mods[-1] // 2])
        if np.sum(diff) > 0:
            layers.append(nn.ZeroPad2d(diff))
        diffs.append(diff)
        layers.append(nn.MaxPool2d((4, 4), stride=4))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 32, (5, 5), stride=1, padding=2))
        mods = np.remainder(-np.floor_divide(self.dim[1:], 4), 4)
        diff = np.array([mods[-2] // 2, mods[-2] - mods[-2] // 2, mods[-1] // 2, mods[-1] - mods[-1] // 2])
        if np.sum(diff) > 0:
            layers.append(nn.ZeroPad2d(diff))
        diffs.append(diff)
        layers.append(nn.MaxPool2d((4, 4), stride=4))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        layers = []

        if np.sum(diffs[0]) > 0:
            layers.append(nn.ZeroPad2d(-diffs[0]))
        layers.append(nn.Upsample(scale_factor=4))
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.ReLU())
        if np.sum(diffs[1]) > 0:
            layers.append(nn.ZeroPad2d(-diffs[1]))
        layers.append(nn.Upsample(scale_factor=4))
        layers.append(nn.Conv2d(32, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, dim[0], (5, 5), padding=2))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Conv_Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Conv_Autoencoder, self).__init__()
        self.dim = dim
        print(dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(dim[0], 16, (3, 3), stride=1, padding=1), #In torch non c'è padding='same' come tensorflow per strides > 2
            nn.MaxPool2d((2,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(8, 8, (3, 3), padding=0),
            #nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, dim[0], (3, 3), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Conv_Autoencoder_f2(nn.Module):
    def __init__(self, dim):
        super(Conv_Autoencoder_f2, self).__init__()
        self.dim = dim
        print(dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(dim[0], 16, (3, 3), stride=1, padding=1), #In torch non c'è padding='same' come tensorflow per strides > 2
            nn.MaxPool2d((2,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(8, 8, (3, 3), padding=0),
            #nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, dim[0], (3, 3), padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class PCA_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, latent_dim):
        super(PCA_Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_dim, bias=False),
        )

    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


