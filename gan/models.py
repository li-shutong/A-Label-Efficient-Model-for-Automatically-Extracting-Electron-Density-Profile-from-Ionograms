import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, n_features: int = 64,
                 n_residual: int = 9):
        super().__init__()

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_features, 7),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, n_features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(n_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features * 2, n_features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(n_features * 4),
            nn.ReLU(inplace=True),
        ]

        bottleneck = [ResidualBlock(n_features * 4) for _ in range(n_residual)]

        decoder = [
            nn.ConvTranspose2d(n_features * 4, n_features * 2, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(n_features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_features * 2, n_features, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_features, in_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *bottleneck, *decoder)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, n_features: int = 64):
        super().__init__()

        def discriminator_block(in_ch, out_ch, normalize=True, stride=2):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, n_features, normalize=False),
            *discriminator_block(n_features, n_features * 2),
            *discriminator_block(n_features * 2, n_features * 4),
            *discriminator_block(n_features * 4, n_features * 8, stride=1),
            nn.Conv2d(n_features * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(model: nn.Module, mean: float = 0.0, std: float = 0.02):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean, std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
