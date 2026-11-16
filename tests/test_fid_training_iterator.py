import importlib
import json
import os
import socket
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from PIL import Image
from torch import nn

from denoising_diffusion_pytorch import fid_evaluation


class DeterministicInception(nn.Module):
    """Exercise real FID statistics without downloading pretrained weights."""

    BLOCK_INDEX_BY_DIM = {2048: 0}

    def __init__(self, blocks):
        super().__init__()

    def forward(self, images):
        mean = images.mean(dim=(1, 2, 3))
        square_mean = images.square().mean(dim=(1, 2, 3))
        return [torch.stack((mean, square_mean), dim=1)[:, :, None, None]]


class CountedIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.next_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.iterator)
        self.next_calls += 1
        return value


def prepare_module(variant):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{variant}")
    original_cycle = module.cycle
    # Count real iterator advancement; keep DataLoaderShard and its synchronization.
    module.cycle = lambda loader: CountedIterator(original_cycle(loader))
    module.cpu_count = lambda: 0
    fid_evaluation.InceptionV3 = DeterministicInception
    return module


def make_trainer(module, folder, results_folder, calculate_fid):
    torch.manual_seed(83)
    model = module.Unet(
        dim=8,
        dim_mults=(1,),
        channels=1,
        attn_dim_head=4,
        attn_heads=1,
    )
    diffusion = module.GaussianDiffusion(
        model,
        image_size=8,
        timesteps=2,
        sampling_timesteps=1,
        beta_schedule="cosine",
        objective="pred_noise",
    )
    return module.Trainer(
        diffusion,
        folder,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_num_steps=1,
        save_and_sample_every=100,
        num_samples=1,
        augment_horizontal_flip=False,
        results_folder=results_folder,
        calculate_fid=calculate_fid,
        num_fid_samples=128,
    )


def create_images(folder):
    folder.mkdir()
    for index in range(100):
        pixels = np.full((8, 8), index + 1, dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"{index:03d}.png")


@pytest.mark.parametrize("variant", ["denoising_diffusion_pytorch", "repaint"])
@pytest.mark.parametrize("calculate_fid", [False, True])
def test_real_statistics_do_not_advance_training(
    tmp_path, monkeypatch, variant, calculate_fid
):
    module = importlib.import_module(f"denoising_diffusion_pytorch.{variant}")
    original_cycle = module.cycle
    monkeypatch.setattr(
        module, "cycle", lambda loader: CountedIterator(original_cycle(loader))
    )
    monkeypatch.setattr(module, "cpu_count", lambda: 0)
    monkeypatch.setattr(fid_evaluation, "InceptionV3", DeterministicInception)
    folder = tmp_path / "images"
    create_images(folder)
    trainer = make_trainer(module, folder, tmp_path / "results", calculate_fid)
    trainer.train()
    assert trainer.step == 1
    assert trainer.dl.next_calls == 1
    before = next(trainer.model.parameters()).detach().clone()

    if calculate_fid:
        trainer.fid_scorer.load_or_precalc_dataset_stats()
        assert trainer.fid_scorer.dataset_stats_loaded
        assert np.isfinite(trainer.fid_scorer.m2).all()
        assert np.isfinite(trainer.fid_scorer.s2).all()
        assert trainer.dl.next_calls == 1, "FID consumed the training iterator"

    trainer.train_num_steps = 2
    trainer.train()
    assert trainer.step == 2
    assert trainer.dl.next_calls == 2
    assert not torch.equal(before, next(trainer.model.parameters()))


def distributed_worker(rank, port, folder, results_folder, variant):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        ACCELERATE_USE_CPU="true",
    )
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    phase_path = results_folder.parent / f"rank-{rank}.json"

    def record_phase(phase):
        phase_path.write_text(json.dumps({"rank": rank, "phase": phase}))

    record_phase("initialized")
    try:
        module = prepare_module(variant)
        trainer = make_trainer(module, folder, results_folder, calculate_fid=True)
        trainer.train()
        assert trainer.step == 1
        assert trainer.dl.next_calls == 1
        record_phase("first_training_step_complete")

        # The first training epoch has fewer than eight batches on each rank.
        # Rank zero must compute its 128-image statistics without a collective.
        if rank == 0:
            record_phase("collecting_real_statistics")
            trainer.fid_scorer.load_or_precalc_dataset_stats()
            assert trainer.dl.next_calls == 1, "FID consumed the training iterator"
            assert np.isfinite(trainer.fid_scorer.m2).all()
            assert np.isfinite(trainer.fid_scorer.s2).all()
        record_phase("waiting_for_real_statistics")
        distributed.barrier()

        before = next(trainer.model.parameters()).detach().clone()
        trainer.train_num_steps = 2
        trainer.train()
        assert trainer.step == 2
        assert trainer.dl.next_calls == 2
        assert not torch.equal(before, next(trainer.model.parameters()))
        weights = torch.cat(
            [parameter.detach().reshape(-1) for parameter in trainer.model.parameters()]
        )
        assert torch.isfinite(weights).all()
        replicas = [torch.empty_like(weights) for _ in range(2)]
        distributed.all_gather(replicas, weights)
        torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
        record_phase("second_training_step_and_gradient_sync_complete")
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("variant", ["denoising_diffusion_pytorch", "repaint"])
def test_rank_zero_statistics_leave_distributed_training_synchronized(
    tmp_path, variant
):
    folder = tmp_path / "images"
    create_images(folder)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    multiprocessing.spawn(
        distributed_worker,
        args=(port, folder, tmp_path / "results", variant),
        nprocs=2,
        join=True,
    )
