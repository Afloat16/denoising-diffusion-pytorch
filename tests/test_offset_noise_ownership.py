import copy

import pytest
import torch

from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from denoising_diffusion_pytorch.classifier_free_guidance import (
    GaussianDiffusion as ClassConditionalDiffusion,
)
from denoising_diffusion_pytorch.classifier_free_guidance import (
    Unet as ClassConditionalUnet,
)
from denoising_diffusion_pytorch.repaint import GaussianDiffusion as RepaintDiffusion


def make_diffusion(diffusion_type, objective, strength):
    model_kwargs = {"dim": 8, "dim_mults": (1,), "channels": 1, "attn_heads": 1, "attn_dim_head": 4}
    if diffusion_type is ClassConditionalDiffusion:
        model = ClassConditionalUnet(num_classes=2, cond_drop_prob=0, **model_kwargs)
        loss_kwargs = {"classes": torch.tensor([0, 1]), "cond_keep_mask": torch.ones(2, dtype=torch.bool)}
    else:
        model = Unet(**model_kwargs)
        loss_kwargs = {}
    diffusion = diffusion_type(
        model, image_size=4, timesteps=8, objective=objective, offset_noise_strength=strength,
    )
    return diffusion, loss_kwargs


@pytest.mark.parametrize("diffusion_type", [GaussianDiffusion, RepaintDiffusion, ClassConditionalDiffusion])
@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
@pytest.mark.parametrize("storage", ["dense", "expanded", "requires_grad"])
@pytest.mark.parametrize("strength", [0.0, 0.2])
def test_training_preserves_reusable_noise(diffusion_type, objective, storage, strength):
    torch.manual_seed(0)
    diffusion, kwargs = make_diffusion(diffusion_type, objective, strength)
    image = torch.randn(2, 1, 4, 4).clamp(-1, 1)
    times = torch.tensor([2, 5])
    if storage == "expanded":
        noise = torch.randn(1, 1, 4, 4).expand_as(image)
    else:
        noise = torch.randn_like(image).requires_grad_(storage == "requires_grad")
    original = noise.detach().clone()

    torch.manual_seed(21)
    first = diffusion.p_losses(image, times, noise=noise, **kwargs)
    torch.testing.assert_close(noise, original, atol=0, rtol=0)
    torch.manual_seed(21)
    repeated = diffusion.p_losses(image, times, noise=noise, **kwargs)
    torch.testing.assert_close(repeated, first, atol=0, rtol=0)
    first.backward()
    assert torch.isfinite(first)
    gradients = [parameter.grad for parameter in diffusion.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    if noise.requires_grad:
        assert noise.grad is not None and torch.isfinite(noise.grad).all()


@pytest.mark.parametrize("diffusion_type", [GaussianDiffusion, RepaintDiffusion, ClassConditionalDiffusion])
@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_offset_training_matches_explicit_augmented_noise(diffusion_type, objective):
    torch.manual_seed(0)
    diffusion, kwargs = make_diffusion(diffusion_type, objective, 0.2)
    reference = copy.deepcopy(diffusion)
    reference.offset_noise_strength = 0.0
    image = torch.randn(2, 1, 4, 4).clamp(-1, 1)
    times = torch.tensor([2, 5])
    noise = torch.randn_like(image)
    torch.manual_seed(21)
    offset = torch.randn(2, 1)[:, :, None, None]
    expected = reference.p_losses(image, times, noise=noise + 0.2 * offset, **kwargs)
    torch.manual_seed(21)
    actual = diffusion.p_losses(image, times, noise=noise, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
