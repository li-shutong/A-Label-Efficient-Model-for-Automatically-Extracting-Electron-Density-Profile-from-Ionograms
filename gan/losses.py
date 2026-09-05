import torch
import torch.nn as nn
import torch.autograd as autograd


def gradient_penalty(discriminator, real: torch.Tensor,
                     fake: torch.Tensor, device: str) -> torch.Tensor:
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(B, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp


class WGANGPLoss:
    def __init__(self, lambda_gp: float = 10.0):
        self.lambda_gp = lambda_gp

    def discriminator_loss(self, discriminator, real: torch.Tensor,
                           fake: torch.Tensor, device: str) -> dict:
        d_real = discriminator(real).mean()
        d_fake = discriminator(fake.detach()).mean()
        gp = gradient_penalty(discriminator, real, fake.detach(), device)
        loss = d_fake - d_real + self.lambda_gp * gp
        return {
            "total": loss,
            "wasserstein_dist": (d_real - d_fake).item(),
            "gp": gp.item(),
        }

    def generator_loss(self, discriminator, fake: torch.Tensor) -> torch.Tensor:
        return -discriminator(fake).mean()


class CycleLoss:
    def __init__(self, lambda_cycle: float = 10.0):
        self.lambda_cycle = lambda_cycle
        self.l1 = nn.L1Loss()

    def __call__(self, real: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        return self.lambda_cycle * self.l1(reconstructed, real)


class IdentityLoss:
    def __init__(self, lambda_identity: float = 5.0):
        self.lambda_identity = lambda_identity
        self.l1 = nn.L1Loss()

    def __call__(self, real: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.lambda_identity * self.l1(identity, real)
