import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch.learned_gaussian_diffusion import (
    LearnedGaussianDiffusion,
)


class LearnedModel(nn.Module):
    channels = 1
    out_dim = 2
    self_condition = False

    def __init__(self):
        super().__init__()
        self.prediction = nn.Parameter(torch.tensor(0.25))
        self.variance = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t):
        return torch.cat(
            (self.prediction.expand_as(x), self.variance.expand_as(x)), dim=1
        )


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
@pytest.mark.parametrize("clip", [False, True])
def test_learned_prediction_decodes_the_selected_parameterization(objective, clip):
    diffusion = LearnedGaussianDiffusion(
        LearnedModel(), image_size=2, timesteps=8, objective=objective
    )
    x = torch.full((2, 1, 2, 2), 0.7)
    t = torch.tensor([2, 5])
    alpha = diffusion.alphas_cumprod[t, None, None, None]
    output = diffusion.model.prediction.expand_as(x)
    if objective == "pred_noise":
        expected_start = (x - (1 - alpha).sqrt() * output) / alpha.sqrt()
    elif objective == "pred_x0":
        expected_start = output
    else:
        expected_start = alpha.sqrt() * x - (1 - alpha).sqrt() * output
    if clip:
        expected_start = expected_start.clamp(-1, 1)
    expected_noise = (x - alpha.sqrt() * expected_start) / (1 - alpha).sqrt()
    predictions = diffusion.model_predictions(
        x, t, clip_x_start=clip, rederive_pred_noise=True
    )
    torch.testing.assert_close(predictions.pred_x_start, expected_start)
    torch.testing.assert_close(predictions.pred_noise, expected_noise)
    torch.testing.assert_close(
        predictions.pred_variance, diffusion.model.variance.expand_as(x)
    )
    mean, _, _, start = diffusion.p_mean_variance(x=x, t=t, clip_denoised=clip)
    torch.testing.assert_close(start, expected_start)
    torch.testing.assert_close(mean, diffusion.q_posterior(expected_start, x, t)[0])


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_simple_loss_uses_the_corresponding_target(objective):
    diffusion = LearnedGaussianDiffusion(
        LearnedModel(), image_size=2, timesteps=8, objective=objective, vb_loss_weight=0
    )
    clean = torch.full((2, 1, 2, 2), -0.4)
    noise = torch.full_like(clean, 0.6)
    t = torch.tensor([2, 5])
    alpha = diffusion.alphas_cumprod[t, None, None, None]
    if objective == "pred_noise":
        target = noise
    elif objective == "pred_x0":
        target = clean
    else:
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
    expected = (diffusion.model.prediction - target).square().flatten(1).mean(1)
    actual = diffusion.p_losses(clean, t, noise=noise, loss_reduction="none")
    torch.testing.assert_close(actual, expected)
    expected_gradient = 2 * (diffusion.model.prediction - target).mean()
    gradient = torch.autograd.grad(actual.mean(), diffusion.model.prediction)[0]
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_learned_ddim_sampling_is_finite(objective):
    diffusion = LearnedGaussianDiffusion(
        LearnedModel(),
        image_size=2,
        timesteps=8,
        sampling_timesteps=3,
        objective=objective,
    )
    samples = diffusion.sample(batch_size=2)
    assert samples.shape == (2, 1, 2, 2)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
def test_variational_loss_keeps_the_mean_head_detached(objective):
    diffusion = LearnedGaussianDiffusion(
        LearnedModel(), image_size=2, timesteps=8, objective=objective
    )
    clean = torch.full((2, 1, 2, 2), -0.4)
    noise = torch.full_like(clean, 0.6)
    t = torch.tensor([2, 5])
    gradients = []
    for weight in (0.0, 1.0):
        diffusion.vb_loss_weight = weight
        loss = diffusion.p_losses(clean, t, noise=noise)
        gradients.append(
            torch.autograd.grad(
                loss, (diffusion.model.prediction, diffusion.model.variance)
            )
        )
    torch.testing.assert_close(gradients[0][0], gradients[1][0])
    torch.testing.assert_close(gradients[0][1], torch.zeros_like(gradients[0][1]))
    assert torch.isfinite(gradients[1][1])
    assert gradients[1][1].abs() > 0
