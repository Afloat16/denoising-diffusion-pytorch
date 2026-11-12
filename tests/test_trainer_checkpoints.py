import pytest
import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from ema_pytorch import EMA

from denoising_diffusion_pytorch import Trainer, Trainer1D
from denoising_diffusion_pytorch.guided_diffusion import Trainer as GuidedTrainer


def make_trainer(trainer_type, rank, folder, monkeypatch):
    accelerator = Accelerator(cpu=True)
    state = PartialState()
    monkeypatch.setattr(state, "process_index", rank)
    monkeypatch.setattr(state, "local_process_index", rank % 2)
    trainer = trainer_type.__new__(trainer_type)
    trainer.accelerator = accelerator
    trainer.model = torch.nn.Linear(2, 2)
    trainer.opt = torch.optim.Adam(trainer.model.parameters())
    trainer.model(torch.ones(1, 2)).sum().backward()
    trainer.opt.step()
    trainer.step = 13
    trainer.results_folder = folder
    if accelerator.is_main_process:
        trainer.ema = EMA(trainer.model)
    return trainer


@pytest.mark.parametrize("trainer_type", [Trainer, Trainer1D, GuidedTrainer])
@pytest.mark.parametrize("rank", [1, 2])
def test_nonowner_does_not_write_checkpoint(trainer_type, rank, tmp_path, monkeypatch):
    trainer = make_trainer(trainer_type, rank, tmp_path, monkeypatch)
    path = tmp_path / "model-3.pt"
    path.write_bytes(b"existing checkpoint")
    trainer.save(3)
    assert path.read_bytes() == b"existing checkpoint"


@pytest.mark.parametrize("trainer_type", [Trainer, Trainer1D])
def test_owner_saves_model_optimizer_and_ema(trainer_type, tmp_path, monkeypatch):
    trainer = make_trainer(trainer_type, 0, tmp_path, monkeypatch)
    trainer.save(3)
    checkpoint = torch.load(tmp_path / "model-3.pt", weights_only=True)
    assert checkpoint["step"] == trainer.step
    assert checkpoint["opt"]["state"]
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(checkpoint["model"][key], value)
    for key, value in trainer.ema.state_dict().items():
        torch.testing.assert_close(checkpoint["ema"][key], value)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_guided_checkpoint_restores_training_state_on_every_rank(rank, tmp_path, monkeypatch):
    source = make_trainer(Trainer, 0, tmp_path, monkeypatch)
    source.save(3)
    checkpoint = torch.load(tmp_path / "model-3.pt", weights_only=True)
    restored = make_trainer(GuidedTrainer, rank, tmp_path, monkeypatch)
    restored.step = 0
    restored.load(3)
    assert restored.step == checkpoint["step"]
    for key, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][key])
    for actual, expected in zip(restored.opt.state_dict()["state"].values(), checkpoint["opt"]["state"].values()):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])
    if rank == 0:
        for key, value in restored.ema.state_dict().items():
            torch.testing.assert_close(value, checkpoint["ema"][key])
