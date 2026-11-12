import math

import pytest
import torch

from denoising_diffusion_pytorch import karras_unet, karras_unet_1d, karras_unet_3d

BLOCKS = [
    (module, dimensions, block)
    for module, dimensions in [(karras_unet_1d, 1), (karras_unet, 2), (karras_unet_3d, 3)]
    for block in ["Encoder", "Decoder", "Attention", "MPFeedForward"]
    if not (dimensions == 1 and block == "MPFeedForward")
]


@pytest.mark.parametrize("module,dimensions,block", BLOCKS)
@pytest.mark.parametrize("balance", [0.0, 0.3, 0.5, 1.0])
def test_balance_weights_the_update_branch(module, dimensions, block, balance):
    kwargs = {"dim": 4, "mp_add_t": balance}
    if block in ("Encoder", "Decoder"):
        kwargs["dropout"] = 0.0
    elif block == "Attention":
        kwargs.update(heads=1, dim_head=4)
    layer = getattr(module, block)(**kwargs)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()

    image = torch.ones((2, 4, *((3,) * dimensions)), requires_grad=True)
    result = layer(image)
    expected_scale = (1 - balance) / math.sqrt((1 - balance) ** 2 + balance ** 2)
    torch.testing.assert_close(result, image * expected_scale)
    result.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
