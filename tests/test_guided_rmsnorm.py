import pytest
import torch

from denoising_diffusion_pytorch.guided_diffusion import RMSNorm, Unet


@pytest.mark.parametrize('shape', [(2, 4, 3, 4), (2, 4, 3, 7), (1, 8, 3, 16)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rmsnorm_matches_channel_rms_values_and_gradients(shape, dtype):
    x = torch.linspace(-2., 3., torch.tensor(shape).prod().item(), dtype = dtype)
    x = x.reshape(shape).requires_grad_(True)
    norm = RMSNorm(shape[1]).to(dtype = dtype)
    with torch.no_grad():
        norm.g.copy_(torch.linspace(0.5, 1.5, shape[1], dtype = dtype).reshape_as(norm.g))

    actual = norm(x)
    expected = x / x.square().mean(dim = 1, keepdim = True).sqrt() * norm.g
    torch.testing.assert_close(actual, expected)

    weights = torch.linspace(-1., 1., x.numel(), dtype = dtype).reshape_as(x)
    actual_grads = torch.autograd.grad((actual * weights).sum(), (x, norm.g))
    expected_grads = torch.autograd.grad((expected * weights).sum(), (x, norm.g))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_rmsnorm_is_invariant_to_spatial_tiling():
    norm = RMSNorm(4)
    x = torch.arange(1., 41.).reshape(1, 4, 2, 5)
    expected = norm(x).repeat(1, 1, 1, 3)
    actual = norm(x.repeat(1, 1, 1, 3))
    torch.testing.assert_close(actual, expected)


def test_guided_unet_normalizes_features_at_each_resolution():
    torch.manual_seed(7)
    model = Unet(dim = 8, dim_mults = (1, 2), channels = 1)
    rms_values = []

    def record_rms(module, inputs, output):
        rms_values.append(output.detach().square().mean(dim = 1).sqrt())

    handles = [
        module.register_forward_hook(record_rms)
        for module in model.modules() if isinstance(module, RMSNorm)
    ]
    output = model(torch.randn(2, 1, 8, 8), torch.tensor([1, 3]))
    for handle in handles:
        handle.remove()

    assert rms_values
    for rms in rms_values:
        torch.testing.assert_close(rms, torch.ones_like(rms))
    output.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
