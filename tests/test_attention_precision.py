from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from denoising_diffusion_pytorch.attend import Attend


@pytest.mark.parametrize(
    "dtype,autocast",
    [
        (torch.float16, False), (torch.bfloat16, False),
        (torch.float32, False), (torch.float64, False),
        (torch.float16, True), (torch.bfloat16, True),
    ],
)
@pytest.mark.parametrize("scale", [None, 0.25])
def test_large_finite_attention_scores(dtype, autocast, scale):
    input_dtype = torch.float32 if autocast else dtype
    query = torch.full((2, 2, 3, 8), 100.0, dtype=input_dtype, requires_grad=True)
    key = torch.full((2, 2, 3, 8), 100.0, dtype=input_dtype, requires_grad=True)
    value = (torch.arange(48, dtype=input_dtype).reshape(2, 2, 3, 4) / 10).requires_grad_(True)
    reference = F.scaled_dot_product_attention(
        query.detach().double(), key.detach().double(), value.detach().double(), scale=scale,
    ).to(dtype)

    context = torch.autocast("cpu", dtype=dtype) if autocast else nullcontext()
    with context:
        result = Attend(scale=scale)(query, key, value)

    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference)
    result.square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
