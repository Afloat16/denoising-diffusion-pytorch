import pytest
import torch

from denoising_diffusion_pytorch.karras_unet_3d import (
    Attention,
    Decoder,
    Encoder,
    KarrasUnet3D,
)


def make_model(**kwargs):
    config = dict(
        image_size=8,
        frames=4,
        channels=1,
        dim=8,
        dim_max=16,
        num_downsamples=1,
        num_blocks_per_stage=1,
        attn_res=(8, 4, 2),
        attn_dim_head=4,
        dropout=0.0,
    )
    config.update(kwargs)
    model = KarrasUnet3D(**config)
    # Exercise gradients through the full network on the first iteration.
    with torch.no_grad():
        model.output_block[-1].gain.fill_(0.5)
    return model


def trace_attention(model):
    records = {}
    handles = []
    for name, block in model.named_modules():
        if not isinstance(block, (Encoder, Decoder)) or block.attn is None:
            continue
        records[name] = []
        for attention in block.attn.modules():
            if not isinstance(attention, Attention):
                continue
            state = {}

            def remember_shape(module, args, state=state):
                state["shape"] = args[0].shape

            def record_sequence(module, args, state=state, name=name):
                q, k, _ = args
                records[name].append((state["shape"], q.shape, k.shape))

            handles.append(attention.register_forward_pre_hook(remember_shape))
            handles.append(attention.attend.register_forward_pre_hook(record_sequence))
    return records, handles


def check_sequences(records, factorized):
    assert records
    for name, calls in records.items():
        batch, _, frames, height, width = calls[0][0]
        expected = (
            [(batch * frames, height * width), (batch * height * width, frames)]
            if factorized
            else [(batch, frames * height * width)]
        )
        actual = [(q[0], q[-2]) for _, q, _ in calls]
        assert actual == expected, (name, actual, expected)
        for _, query, key in calls:
            assert key[0] == query[0]
            # Each independent attention sequence keeps four learned memory tokens.
            assert key[-2] == query[-2] + 4


@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("downsample_type", ["all", "image", "frame"])
@pytest.mark.parametrize("blocks", [1, 2])
def test_all_stage_attention_uses_requested_sequences(
    factorized, downsample_type, blocks
):
    torch.manual_seed(17)
    model = make_model(
        factorize_space_time_attn=factorized,
        downsample_types=(downsample_type,),
        num_blocks_per_stage=blocks,
    )
    records, handles = trace_attention(model)
    image = torch.randn(2, 1, 4, 8, 8, requires_grad=True)
    output = model(image, torch.tensor([0.2, 0.7]))
    for handle in handles:
        handle.remove()
    assert output.shape == image.shape
    check_sequences(records, factorized)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert model.input_block.weight.grad.abs().sum() > 0
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize("downsamples", [0, 2])
@pytest.mark.parametrize("self_condition", [False, True])
@pytest.mark.parametrize("flash", [False, True])
def test_middle_blocks_and_multiple_stages(downsamples, self_condition, flash):
    torch.manual_seed(29)
    model = make_model(
        factorize_space_time_attn=True,
        num_downsamples=downsamples,
        num_blocks_per_stage=(1,) * (downsamples + 1),
        self_condition=self_condition,
        attn_flash=flash,
    )
    image = torch.randn(1, 1, 4, 8, 8)
    condition = torch.randn_like(image) if self_condition else None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_weight = model.input_block.weight.detach().clone()
    for _ in range(2):
        records, handles = trace_attention(model)
        output = model(image, torch.tensor([0.4]), self_cond=condition)
        for handle in handles:
            handle.remove()
        check_sequences(records, True)
        assert output.shape == image.shape
        loss = (output - image).square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert not torch.equal(initial_weight, model.input_block.weight)


@pytest.mark.parametrize("flash", [False, True])
def test_default_joint_attention_matches_explicit_false(flash):
    torch.manual_seed(41)
    default_model = make_model(attn_flash=flash)
    torch.manual_seed(41)
    explicit_model = make_model(attn_flash=flash, factorize_space_time_attn=False)
    image = torch.randn(1, 1, 4, 8, 8)
    time = torch.tensor([0.3])
    for name, tensor in default_model.state_dict().items():
        torch.testing.assert_close(
            tensor, explicit_model.state_dict()[name], rtol=0, atol=0
        )
    torch.testing.assert_close(
        default_model(image, time), explicit_model(image, time), rtol=0, atol=0
    )


@pytest.mark.parametrize("self_condition", [False, True])
def test_absent_attention_keeps_identical_training(self_condition):
    models = []
    for factorized in (False, True):
        torch.manual_seed(53)
        models.append(
            make_model(
                factorize_space_time_attn=factorized,
                attn_res=(),
                self_condition=self_condition,
            )
        )
    image = torch.randn(1, 1, 4, 8, 8)
    condition = torch.randn_like(image) if self_condition else None
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
    for _ in range(2):
        outputs = []
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            output = model(image, torch.tensor([0.3]), self_cond=condition)
            outputs.append(output.detach())
            (output - image).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for name, tensor in models[0].state_dict().items():
            torch.testing.assert_close(
                tensor, models[1].state_dict()[name], rtol=0, atol=0
            )
