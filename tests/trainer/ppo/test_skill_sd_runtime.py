from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.model import compute_position_id_with_mask


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, chat, **kwargs):
        return chat[0]["content"]

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def encode(self, text, **kwargs):
        return [1, 4, 5]

    def batch_decode(self, responses, **kwargs):
        return ["<action>look</action>"] * len(responses)


class RolloutWorker:
    world_size = 1

    def generate_sequences(self, batch):
        prompts = batch.batch["input_ids"]
        responses = torch.tensor([[7, 2]]).repeat(len(batch), 1)
        mask = torch.cat([batch.batch["attention_mask"], torch.ones_like(responses)], dim=1)
        return DataProto.from_dict(tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": mask,
            "position_ids": compute_position_id_with_mask(mask),
        })


class Environments:
    def __init__(self, count):
        self.count = count
        self.observation = {"text": ["Your task is to: put an apple on the table."] * count, "image": None, "anchor": ["observation"] * count}
        self.infos = [{"extra.gamefile": "/train/pick_and_place-Apple-None-Table-1/trial/game.tw-pddl", "is_action_valid": True} for _ in range(count)]

    def reset(self, kwargs):
        self.steps = 0
        return self.observation, self.infos

    def step(self, actions):
        assert actions == ["<action>look</action>"] * self.count
        self.steps += 1
        done = self.steps == 2
        rewards = np.array([10.0 if done and i % 2 == 0 else 0.0 for i in range(self.count)])
        return self.observation, rewards, np.full(self.count, done), self.infos

    def success_evaluator(self, **kwargs):
        return {"success_rate": (kwargs["episode_rewards"] > 0).astype(float)}


@pytest.mark.parametrize("is_train,env_count", [(True, 4), (False, 2)])
def test_config_collects_trajectories_and_adjusts_actor_batches(is_train, env_count):
    config = OmegaConf.load(Path(__file__).parents[3] / "verl/trainer/config/skill_sd.yaml")
    config.data.train_batch_size = config.data.val_batch_size = 2
    config.data.max_prompt_length = 8
    config.data.return_raw_chat = True
    config.env.rollout.n = 2
    config.trainer.n_gpus_per_node = 1
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 4
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 2
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 2
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.use_reference_policy = False
    trainer._validate_config()

    batch = DataProto.from_dict(
        tensors={"input_ids": torch.ones(2, 3, dtype=torch.long)},
        non_tensors={"raw_prompt": np.array([[{"role": "user", "content": ""}]] * 2, dtype=object), "data_source": np.array(["text", "text"], dtype=object)},
    )
    result = TrajectoryCollector(config, Tokenizer()).multi_turn_loop(batch, RolloutWorker(), Environments(env_count), is_train=is_train)

    assert len(result) == env_count * 2
    assert len(set(result.non_tensor_batch["traj_uid"])) == env_count
    assert result.non_tensor_batch["episode_lengths"].tolist() == [2.0] * len(result)
    assert result.non_tensor_batch["episode_rewards"].tolist() == [10.0, 10.0, 0.0, 0.0] * (env_count // 2)
    assert result.non_tensor_batch["turn_step"].tolist() == [0, 1] * env_count
    shortened = result.select_idxs(np.arange(len(result) - 1))
    assert len(adjust_batch(config, shortened)) == len(result)
