import math
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from agent_system.multi_turn_rollout import adjust_batch
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, apply_invalid_action_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.skill_sd_bank import SkillRecord
from verl.trainer.ppo.skill_sd_writer import SkillEvolution, SummaryRequest
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import masked_mean

ALFWORLD_TASK_TYPES = (
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
)

MAX_TRANSCRIPT_CHARS = 200000


def alfworld_task_type(gamefile: str) -> str:
    for task_type in ALFWORLD_TASK_TYPES:
        if task_type in gamefile:
            return task_type
    raise ValueError(f"cannot infer an ALFWorld task type from {gamefile!r}")


def build_teacher_batch(batch: DataProto, selected: dict[str, SkillRecord | None], tokenizer, max_prompt_length: int) -> DataProto:
    response_length = batch.batch["responses"].shape[1]
    prompt_ids = batch.batch["input_ids"][:, :-response_length]
    prompt_mask = batch.batch["attention_mask"][:, :-response_length].bool()
    prompts = []
    for row, trajectory_id in enumerate(batch.non_tensor_batch["traj_uid"]):
        prompt = prompt_ids[row][prompt_mask[row]]
        record = selected.get(str(trajectory_id))
        if record is not None:
            prefix = tokenizer.encode(f"[Privileged Skill Information]\n{record.privileged_text()}\n", add_special_tokens=False)
            prompt = torch.cat([torch.as_tensor(prefix, dtype=prompt.dtype, device=prompt.device), prompt])
        prompts.append(prompt)
    width = max(max_prompt_length, max(len(prompt) for prompt in prompts))
    response_mask = batch.batch["attention_mask"][:, -response_length:]
    teacher_prompt_ids = torch.stack([F.pad(prompt, (width - len(prompt), 0), value=tokenizer.pad_token_id) for prompt in prompts])
    teacher_prompt_mask = torch.stack([F.pad(torch.ones(len(prompt), dtype=response_mask.dtype, device=response_mask.device), (width - len(prompt), 0)) for prompt in prompts])
    attention_mask = torch.cat([teacher_prompt_mask, response_mask], dim=-1)
    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.cat([teacher_prompt_ids, batch.batch["responses"]], dim=-1),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "responses": batch.batch["responses"],
        }
    )
    teacher_batch.meta_info = dict(batch.meta_info)
    return teacher_batch


class SkillSDRayTrainer(RayPPOTrainer):
    def __init__(self, *args, skill_evolution: SkillEvolution, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_evolution = skill_evolution
        skill_sd_cfg = self.config.algorithm.get("skill_sd", {})
        self.reward_scale = skill_sd_cfg.get("reward_scale", 10.0)
        self.privileged_context = skill_sd_cfg.get("privileged_context", "skill")
        self.trajectory_context_tokens = skill_sd_cfg.get("trajectory_context_tokens", 2048)

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        if self.global_steps >= self.total_training_steps:
            return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                new_skills = self.skill_evolution.start_iteration(self.global_steps)

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"] + [key for key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs") if key in batch.non_tensor_batch]
                gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        batch = self.traj_collector.multi_turn_loop(gen_batch=gen_batch, actor_rollout_wg=self.actor_rollout_wg, envs=self.envs, is_train=True)

                    requests = self._summary_requests(batch)
                    selections = self.skill_evolution.select([(request.trajectory_id, request.task_id) for request in requests])

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch.pop("entropys")
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=batch.batch["response_mask"], loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode)
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        batch = batch.union(old_log_prob)

                    with _timer("teacher_forward", timing_raw):
                        batch.batch["teacher_log_probs"] = self._teacher_log_probs(batch, dict(selections))

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch) if self.ref_in_actor else self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with _timer("adv", timing_raw):
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({key: np.array(value) for key, value in reward_extra_infos_dict.items()})
                        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch, invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef)
                            metrics.update(invalid_metrics)
                        batch.batch["token_level_rewards"] = reward_tensor
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                            multi_turn=False,
                        )
                        rewards = {str(trajectory_id): float(reward) / self.reward_scale for trajectory_id, reward in zip(batch.non_tensor_batch["traj_uid"], batch.non_tensor_batch["episode_rewards"])}
                        self.skill_evolution.record_rewards(selections, rewards)

                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    if self.privileged_context == "trajectory":
                        new_skills += self.skill_evolution.add_trajectories(requests, self._trajectory_texts(requests))
                    elif self.skill_evolution.writer is not None:
                        self.skill_evolution.submit(requests)
                    metrics.update(self._skill_metrics(batch, requests, selections, new_skills))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            self._dump_generations(
                                inputs=self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True),
                                outputs=self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True),
                                scores=batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=self.resource_pool_manager.get_n_gpus()))
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _save_checkpoint(self):
        self.skill_evolution.drain()
        self.skill_evolution.save(Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "skill_sd_state.json")
        super()._save_checkpoint()

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps == 0:
            return
        if self.config.trainer.resume_mode == "resume_path":
            checkpoint_dir = Path(self.config.trainer.resume_from_path)
        else:
            checkpoint_dir = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        self.skill_evolution.load(checkpoint_dir / "skill_sd_state.json")

    def _summary_requests(self, batch: DataProto) -> list[SummaryRequest]:
        response_mask = compute_response_mask(batch)
        rows_by_trajectory = defaultdict(list)
        for row, trajectory_id in enumerate(batch.non_tensor_batch["traj_uid"]):
            rows_by_trajectory[str(trajectory_id)].append(row)
        requests = []
        for trajectory_id, rows in rows_by_trajectory.items():
            rows.sort(key=lambda row: int(batch.non_tensor_batch["turn_step"][row]))
            transcript = "\n".join(f"USER: {batch.non_tensor_batch['anchor_obs'][row]}\nASSISTANT: {self.tokenizer.decode(batch.batch['responses'][row][response_mask[row].bool()], skip_special_tokens=True)}" for row in rows)
            requests.append(
                SummaryRequest(
                    task_id=alfworld_task_type(batch.non_tensor_batch["gamefile"][rows[0]]),
                    trajectory_id=trajectory_id,
                    iteration=self.global_steps,
                    transcript=transcript[-MAX_TRANSCRIPT_CHARS:],
                    reward=float(batch.non_tensor_batch["episode_rewards"][rows[0]]) / self.reward_scale,
                    step_count=len(rows),
                )
            )
        return requests

    def _teacher_log_probs(self, batch: DataProto, selected: dict[str, SkillRecord | None]) -> torch.Tensor:
        if all(record is None for record in selected.values()):
            return batch.batch["old_log_probs"].detach().clone()
        teacher_batch = build_teacher_batch(batch, selected, self.tokenizer, self.config.data.max_prompt_length)
        teacher_log_probs = self.actor_rollout_wg.compute_log_prob(teacher_batch).batch["old_log_probs"]
        no_skill = torch.tensor([selected.get(str(trajectory_id)) is None for trajectory_id in batch.non_tensor_batch["traj_uid"]], device=teacher_log_probs.device)
        teacher_log_probs[no_skill] = batch.batch["old_log_probs"][no_skill].to(teacher_log_probs.device)
        return teacher_log_probs

    def _trajectory_texts(self, requests: list[SummaryRequest]) -> list[str]:
        texts = []
        for request in requests:
            token_ids = self.tokenizer.encode(request.transcript, add_special_tokens=False)[-self.trajectory_context_tokens :]
            texts.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        return texts

    def _skill_metrics(self, batch: DataProto, requests: list[SummaryRequest], selections: list[tuple[str, SkillRecord | None]], new_skills: int) -> dict[str, float]:
        mask = batch.batch["response_mask"]
        gap = (batch.batch["teacher_log_probs"] - batch.batch["old_log_probs"]) * mask
        retrieved = [(request.task_id, record.task_id) for request, (_, record) in zip(requests, selections) if record is not None]
        metrics = {
            "skill_sd/teacher_student_gap_mean": masked_mean(gap, mask).item(),
            "skill_sd/teacher_student_gap_rms": math.sqrt(masked_mean(gap.square(), mask).item()),
            "skill_sd/skill_coverage": len(retrieved) / max(len(selections), 1),
            "skill_sd/same_task_retrieval_rate": sum(query == source for query, source in retrieved) / len(retrieved) if retrieved else 0.0,
            "skill_sd/new_skills": new_skills,
        }
        metrics.update({f"skill_sd/bank_{key}": value for key, value in self.skill_evolution.stats().items()})
        return metrics
