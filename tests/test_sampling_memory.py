import weakref

import pytest
import torch

from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from denoising_diffusion_pytorch.guided_diffusion import (
    GaussianDiffusion as GuidedDiffusion,
)
from denoising_diffusion_pytorch.repaint import GaussianDiffusion as RepaintDiffusion


@pytest.mark.parametrize("diffusion_type", [GaussianDiffusion, GuidedDiffusion, RepaintDiffusion])
@pytest.mark.parametrize("sampler", ["p_sample_loop", "ddim_sample"])
@pytest.mark.parametrize("keep_history", [False, True])
def test_sampling_releases_unrequested_image_history(diffusion_type, sampler, keep_history):
    torch.manual_seed(0)
    model = Unet(dim=8, dim_mults=(1,), channels=1, attn_heads=1, attn_dim_head=4)
    diffusion = diffusion_type(model, image_size=4, timesteps=8, sampling_timesteps=4).eval()
    image_refs = []
    live_counts = []

    def observe_image(module, inputs):
        image_refs.append(weakref.ref(inputs[0]))
        live_counts.append(sum(reference() is not None for reference in image_refs))

    handle = model.register_forward_pre_hook(observe_image)
    try:
        result = getattr(diffusion, sampler)((2, 1, 4, 4), return_all_timesteps=keep_history)
    finally:
        handle.remove()

    steps = diffusion.num_timesteps if sampler == "p_sample_loop" else diffusion.sampling_timesteps
    assert len(live_counts) == steps
    assert torch.isfinite(result).all()
    if keep_history:
        assert result.shape == (2, steps + 1, 1, 4, 4)
        assert max(live_counts) == steps
    else:
        assert result.shape == (2, 1, 4, 4)
        assert max(live_counts) <= 2
