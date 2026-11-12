import math

import pytest
import torch

from denoising_diffusion_pytorch.karras_unet import Conv2d
from denoising_diffusion_pytorch.karras_unet_1d import Conv1d
from denoising_diffusion_pytorch.karras_unet_3d import Conv3d


@pytest.mark.parametrize("convolution,dimensions", [(Conv1d, 1), (Conv2d, 2), (Conv3d, 3)])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("concat_ones", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_magnitude_scaling_counts_every_input_channel(
    convolution, dimensions, channels, kernel_size, concat_ones, training, dtype,
):
    layer = convolution(channels, 2, kernel_size, concat_ones_to_input=concat_ones).to(dtype)
    layer.train(training)
    with torch.no_grad():
        layer.weight.fill_(1)

    image = torch.ones((1, channels, *((5,) * dimensions)), dtype=dtype, requires_grad=True)
    output = layer(image)
    center = output[(slice(None), slice(None), *((2,) * dimensions))]
    fan_in = (channels + int(concat_ones)) * kernel_size ** dimensions
    expected = torch.full_like(center, math.sqrt(fan_in))
    torch.testing.assert_close(center, expected)

    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
