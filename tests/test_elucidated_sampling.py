import pytest
import torch

from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.elucidated_diffusion import ElucidatedDiffusion


def make_diffusion():
    net = Unet(
        dim=8,
        dim_mults=(1,),
        channels=1,
        random_fourier_features=True,
        attn_heads=1,
        attn_dim_head=4,
    )
    return ElucidatedDiffusion(net, image_size=4, channels=1, S_churn=0).eval()


@pytest.mark.parametrize("steps", [1, 2, 8])
def test_sampling_schedule_reaches_zero(steps):
    diffusion = make_diffusion()
    schedule = diffusion.sample_schedule(steps)
    fractions = torch.linspace(0, 1, steps, dtype=torch.float64)
    expected = (
        diffusion.sigma_max ** (1 / diffusion.rho)
        + fractions * (
            diffusion.sigma_min ** (1 / diffusion.rho)
            - diffusion.sigma_max ** (1 / diffusion.rho)
        )
    ) ** diffusion.rho
    expected = torch.cat((expected, torch.zeros(1))).float()
    torch.testing.assert_close(schedule, expected)
    assert (schedule[:-1] > schedule[1:]).all()


@pytest.mark.parametrize("method", ["sample", "sample_using_dpmpp"])
@pytest.mark.parametrize("steps", [1, 4])
def test_native_sampling_with_one_or_multiple_steps(method, steps):
    diffusion = make_diffusion()
    torch.manual_seed(12)
    initial = diffusion.sigma_max * torch.randn(2, 1, 4, 4)
    with torch.no_grad():
        denoised = diffusion.preconditioned_network_forward(initial, float(diffusion.sigma_max))
    torch.manual_seed(12)
    result = getattr(diffusion, method)(batch_size=2, num_sample_steps=steps)
    assert result.shape == initial.shape
    assert torch.isfinite(result).all()
    assert ((0 <= result) & (result <= 1)).all()
    if steps == 1:
        torch.testing.assert_close(result, (denoised.clamp(-1, 1) + 1) / 2, atol=1e-5, rtol=1e-5)
