from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import nn

from denoising_diffusion_pytorch import fid_evaluation


class SmallFeatureExtractor(nn.Module):
    BLOCK_INDEX_BY_DIM: ClassVar[dict] = {3: 0}

    def __init__(self, blocks):
        super().__init__()

    def forward(self, samples):
        return [samples.mean(dim=(-1, -2), keepdim=True)]


class Sampler(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout(0.5)
        self.fail = fail
        self.calls = []

    def sample(self, batch_size):
        self.calls.append([module.training for module in self.modules()])
        if self.fail and len(self.calls) == 2:
            raise RuntimeError("sampling failed")
        return self.dropout(self.norm(torch.randn(batch_size, 3, 4, 4)))


def make_evaluator(monkeypatch, tmp_path, sampler):
    monkeypatch.setattr(fid_evaluation, "InceptionV3", SmallFeatureExtractor)
    real = iter([torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)])
    return fid_evaluation.FIDEvaluation(
        batch_size=2, dl=real, sampler=sampler, channels=3, stats_dir=tmp_path,
        device="cpu", num_fid_samples=4, inception_block_idx=3,
    )


@pytest.mark.parametrize("mode", ["train", "eval", "frozen_norm"])
@pytest.mark.parametrize("fail", [False, True])
def test_fid_restores_every_sampler_module_mode(monkeypatch, tmp_path, mode, fail):
    torch.manual_seed(0)
    sampler = Sampler(fail=fail)
    sampler.train(mode != "eval")
    if mode == "frozen_norm":
        sampler.norm.eval()
    before = [module.training for module in sampler.modules()]
    evaluator = make_evaluator(monkeypatch, tmp_path, sampler)

    if fail:
        with pytest.raises(RuntimeError, match="sampling failed"):
            evaluator.fid_score()
    else:
        assert np.isfinite(evaluator.fid_score())

    assert len(sampler.calls) == 2
    assert all(not any(flags) for flags in sampler.calls)
    assert [module.training for module in sampler.modules()] == before


def test_fid_accepts_non_module_samplers(monkeypatch, tmp_path):
    class CustomSampler:
        def eval(self):
            return self

        def sample(self, batch_size):
            return torch.randn(batch_size, 3, 4, 4)

    torch.manual_seed(0)
    evaluator = make_evaluator(monkeypatch, tmp_path, CustomSampler())
    assert np.isfinite(evaluator.fid_score())
