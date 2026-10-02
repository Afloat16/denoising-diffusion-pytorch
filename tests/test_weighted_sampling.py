import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch.weighted_objective_gaussian_diffusion import (
    WeightedObjectiveGaussianDiffusion,
)


class WeightedModel(nn.Module):
    channels = 1
    out_dim = 4
    self_condition = False

    def __init__(self):
        super().__init__()
        self.output = nn.Parameter(torch.tensor([0.1, 0.3, 0.5, -0.5]))

    def forward(self, x, t):
        return self.output[None, :, None, None].expand(x.shape[0], -1, *x.shape[2:])


def make_diffusion():
    return WeightedObjectiveGaussianDiffusion(
        WeightedModel(), image_size=2, timesteps=4
    )


@pytest.mark.parametrize("timestep", [0, 2])
def test_weighted_p_sample_obeys_parent_sampler_contract(timestep):
    diffusion = make_diffusion()
    x = torch.zeros(2, 1, 2, 2)
    sample, prediction = diffusion.p_sample(x, timestep)
    assert sample.shape == prediction.shape == x.shape
    assert torch.isfinite(sample).all()
    if timestep == 0:
        torch.testing.assert_close(sample, prediction)


@pytest.mark.parametrize("clip", [False, True])
def test_weighted_posterior_returns_the_mixture_estimate(clip):
    diffusion = make_diffusion()
    x = torch.zeros(2, 1, 2, 2)
    t = torch.tensor([1, 2])
    mean, variance, log_variance, prediction = diffusion.p_mean_variance(
        x=x, t=t, clip_denoised=clip
    )
    alpha = diffusion.alphas_cumprod[t, None, None, None]
    from_noise = -((1 - alpha) / alpha).sqrt() * 0.1
    weights = torch.tensor([0.5, -0.5]).softmax(0)
    expected = weights[0] * from_noise + weights[1] * 0.3
    expected = expected.expand_as(x)
    if clip:
        expected = expected.clamp(-1, 1)
    torch.testing.assert_close(prediction, expected)
    reference = diffusion.q_posterior(expected, x, t)
    for actual, expected_value in zip((mean, variance, log_variance), reference):
        torch.testing.assert_close(actual, expected_value)


def test_weighted_full_reverse_chain():
    diffusion = make_diffusion()
    samples = diffusion.sample(batch_size=2, return_all_timesteps=True)
    assert samples.shape == (2, 5, 1, 2, 2)
    assert torch.isfinite(samples).all()
