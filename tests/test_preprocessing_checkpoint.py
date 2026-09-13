"""Checkpoint preprocessing must agree with its declared model configuration."""
import pytest
import torch

from transformer_rl.checkpoint import load_checkpoint, save_checkpoint
from transformer_rl.config import ModelConfig, PPOConfig
from transformer_rl.model import ActorCritic
from transformer_rl.ppo import PPOTrainer


@pytest.mark.parametrize("key", ["actor.frame_scale", "actor.time_frequencies"])
def test_finite_but_inconsistent_preprocessing_rejected(tmp_path, key):
    model = ActorCritic(ModelConfig())
    trainer = PPOTrainer(model, PPOConfig())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, trainer, 0, {})
    state = torch.load(path, weights_only=True)
    state["model_state"][key][0] += 1
    modified = tmp_path / "modified.pt"
    torch.save(state, modified)
    with pytest.raises(ValueError, match="configured preprocessing"):
        load_checkpoint(modified)
