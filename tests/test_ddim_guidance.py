import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.guided_diffusion import GaussianDiffusion


class ZeroPrediction(nn.Module):
    channels = out_dim = 1
    self_condition = False
    random_or_learned_sinusoidal_cond = False

    def forward(self, image, time, self_cond=None):
        return torch.zeros_like(image)


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_one_step_guidance_matches_conditional_score(objective):
    diffusion = GaussianDiffusion(
        ZeroPrediction(), image_size=4, timesteps=8, sampling_timesteps=1,
        objective=objective, auto_normalize=False,
    )
    torch.manual_seed(12)
    initial = torch.randn(2, 1, 4, 4)
    alpha = diffusion.alphas_cumprod[-1]
    if objective == "pred_noise":
        denoised = (initial / alpha.sqrt()).clamp(-1, 1)
    elif objective == "pred_x0":
        denoised = torch.zeros_like(initial)
    else:
        denoised = (alpha.sqrt() * initial).clamp(-1, 1)

    gradient = torch.full_like(initial, 0.01)
    expected = denoised + (1 - alpha) / alpha.sqrt() * gradient
    torch.manual_seed(12)
    result = diffusion.sample(batch_size=2, cond_fn=lambda image, time: gradient)
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
@pytest.mark.parametrize("sampling_steps", [1, 4])
@pytest.mark.parametrize("eta", [0.0, 0.5])
def test_native_unet_ddim_uses_classifier_gradients(objective, sampling_steps, eta):
    torch.manual_seed(0)
    model = Unet(
        dim=8, dim_mults=(1,), channels=1, attn_heads=1, attn_dim_head=4,
        self_condition=True,
    )
    diffusion = GaussianDiffusion(
        model, image_size=4, timesteps=8, sampling_timesteps=sampling_steps,
        objective=objective, ddim_sampling_eta=eta,
    ).eval()
    calls = []

    def classifier_gradient(image, time, *, target, strength):
        calls.append(time.clone())
        with torch.enable_grad():
            image = image.detach().requires_grad_(True)
            log_probability = -strength * (image - target).square().sum() / 2
            return torch.autograd.grad(log_probability, image)[0]

    torch.manual_seed(12)
    unconditioned = diffusion.sample(batch_size=2)
    torch.manual_seed(12)
    conditioned = diffusion.sample(
        batch_size=2, cond_fn=classifier_gradient,
        guidance_kwargs={"target": 0.3, "strength": 0.01},
    )

    expected_times = [7] if sampling_steps == 1 else [7, 5, 3, 1]
    assert [time[0].item() for time in calls] == expected_times
    assert all(time.shape == (2,) and torch.equal(time, time[:1].expand(2)) for time in calls)
    assert conditioned.shape == (2, 1, 4, 4)
    assert torch.isfinite(conditioned).all()
    assert not torch.allclose(conditioned, unconditioned)
    assert not conditioned.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_zero_guidance_preserves_single_step_prediction(objective):
    torch.manual_seed(0)
    model = Unet(dim=8, dim_mults=(1,), channels=1, attn_heads=1, attn_dim_head=4)
    diffusion = GaussianDiffusion(
        model, image_size=4, timesteps=8, sampling_timesteps=1, objective=objective,
    ).eval()
    torch.manual_seed(12)
    expected = diffusion.sample(batch_size=2)
    torch.manual_seed(12)
    actual = diffusion.sample(batch_size=2, cond_fn=lambda image, time: torch.zeros_like(image))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
