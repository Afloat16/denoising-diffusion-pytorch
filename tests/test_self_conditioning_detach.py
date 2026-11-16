"""Self-conditioning must detach a tensor without mutating the model's output."""

from copy import deepcopy
from importlib import import_module

import pytest
import torch
from torch import nn

VARIANTS = (
    "denoising_diffusion_pytorch",
    "denoising_diffusion_pytorch_1d",
    "guided_diffusion",
    "repaint",
)


class ViewDenoiser(nn.Module):
    channels = out_dim = 2
    self_condition = True
    random_or_learned_sinusoidal_cond = False

    def __init__(self, dimensions):
        super().__init__()
        conv = nn.Conv1d if dimensions == 1 else nn.Conv2d
        self.conv = conv(4, 4, kernel_size=3, padding=1)
        self.conditions = []

    def forward(self, x, time, x_self_cond=None):
        self.conditions.append(x_self_cond)
        condition = torch.zeros_like(x) if x_self_cond is None else x_self_cond
        return self.conv(torch.cat((x, condition), dim=1))[:, : self.channels]


class AliasingDenoiser(nn.Module):
    channels = out_dim = 2
    self_condition = True
    random_or_learned_sinusoidal_cond = False

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.condition_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, time, x_self_cond=None):
        if x_self_cond is None:
            return x
        return self.scale * x + self.condition_scale * x_self_cond


def make_diffusion(variant, model, objective, dtype):
    module = import_module(f"denoising_diffusion_pytorch.{variant}")
    if variant.endswith("_1d"):
        diffusion = module.GaussianDiffusion1D(
            model, seq_length=8, timesteps=10, objective=objective
        )
        shape = (2, 2, 8)
    else:
        diffusion = module.GaussianDiffusion(
            model,
            image_size=8,
            timesteps=10,
            objective=objective,
            beta_schedule="cosine",
        )
        shape = (2, 2, 8, 8)
    return module, diffusion.to(dtype=dtype), shape


def manual_loss(diffusion, model, x_start, noise, times, use_condition):
    """Compute the two forwards and weighted objective independently of p_losses."""
    expand = (-1,) + (1,) * (x_start.ndim - 1)
    alpha = diffusion.sqrt_alphas_cumprod[times].reshape(expand)
    sigma = diffusion.sqrt_one_minus_alphas_cumprod[times].reshape(expand)
    noisy = alpha * x_start + sigma * noise
    condition = None
    if use_condition:
        with torch.no_grad():
            prediction = model(noisy, times)
            if diffusion.objective == "pred_noise":
                condition = (noisy - sigma * prediction) / alpha
            elif diffusion.objective == "pred_v":
                condition = alpha * noisy - sigma * prediction
            else:
                condition = prediction
            condition = condition.detach()
    prediction = model(noisy, times, condition)
    if diffusion.objective == "pred_noise":
        target = noise
    elif diffusion.objective == "pred_v":
        target = alpha * noise - sigma * x_start
    else:
        target = x_start
    per_sample = (prediction - target).square().flatten(1).mean(1)
    return (per_sample * diffusion.loss_weight[times]).mean()


def assert_matches_manual(diffusion, x, noise, times, use_condition):
    expected_model = deepcopy(diffusion.model)
    expected_x = x.detach().clone().requires_grad_()
    expected = manual_loss(
        diffusion, expected_model, expected_x, noise, times, use_condition
    )
    expected.backward()

    actual = diffusion.p_losses(x, times, noise=noise)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=2e-5, atol=1e-7)
    for actual_parameter, expected_parameter in zip(
        diffusion.model.parameters(), expected_model.parameters(), strict=True
    ):
        assert actual_parameter.grad is not None
        assert torch.isfinite(actual_parameter.grad).all()
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, rtol=2e-5, atol=1e-7
        )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("objective", ("pred_noise", "pred_x0", "pred_v"))
def test_view_output_matches_loss_and_gradient_oracle(
    monkeypatch, variant, dtype, objective
):
    torch.manual_seed(11)
    model = ViewDenoiser(1 if variant.endswith("_1d") else 2)
    module, diffusion, shape = make_diffusion(variant, model, objective, dtype)
    monkeypatch.setattr(module, "random", lambda: 0.0)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    noise = torch.randn_like(x)
    assert_matches_manual(diffusion, x, noise, torch.tensor([1, 6]), True)
    assert len(model.conditions) == 2
    assert model.conditions[0] is None
    assert not model.conditions[1].requires_grad


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_alias_output_preserves_training_input_gradients(monkeypatch, variant, dtype):
    torch.manual_seed(21)
    module, diffusion, shape = make_diffusion(
        variant, AliasingDenoiser(), "pred_x0", dtype
    )
    monkeypatch.setattr(module, "random", lambda: 0.0)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    assert_matches_manual(diffusion, x, torch.randn_like(x), torch.tensor([2, 7]), True)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_unselected_self_conditioning_keeps_one_forward(monkeypatch, variant, dtype):
    torch.manual_seed(31)
    model = ViewDenoiser(1 if variant.endswith("_1d") else 2)
    module, diffusion, shape = make_diffusion(variant, model, "pred_x0", dtype)
    monkeypatch.setattr(module, "random", lambda: 0.9)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    assert_matches_manual(
        diffusion, x, torch.randn_like(x), torch.tensor([1, 4]), False
    )
    assert model.conditions == [None]
