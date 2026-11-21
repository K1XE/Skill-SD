import math

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="skill_sd", version_base=None)
def main(config):
    run_skill_sd(config)


def run_skill_sd(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = SkillSDTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class SkillSDTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        skill_sd_cfg = config.algorithm.get("skill_sd", {})
        with open_dict(config):
            config.actor_rollout_ref.actor.use_sdl_loss = True
            config.actor_rollout_ref.actor.sdl_loss_coef = skill_sd_cfg.get("sdl_lambda", 0.001)

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id}
        if config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        from agent_system.reward_manager import EpisodeRewardManager

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from verl.trainer.ppo.skill_sd_bank import SkillBank
        from verl.trainer.ppo.skill_sd_writer import SkillEmbedder, SkillEvolution, SkillWriter

        writer_cfg = skill_sd_cfg.get("writer", {})
        embedder_cfg = skill_sd_cfg.get("embedder", {})
        writer = None
        if writer_cfg.get("base_url"):
            writer = SkillWriter(
                base_url=writer_cfg.base_url,
                model=writer_cfg.model,
                temperature=writer_cfg.get("temperature", 0.6),
                max_tokens=writer_cfg.get("max_tokens", 4096),
                timeout=writer_cfg.get("timeout", 300),
            )
        embedder = None
        if embedder_cfg.get("base_url"):
            embedder = SkillEmbedder(
                base_url=embedder_cfg.base_url,
                model=embedder_cfg.model,
                timeout=embedder_cfg.get("timeout", 60),
                max_batch=embedder_cfg.get("max_batch", 32),
            )
        bank = SkillBank(
            ucb_constant=skill_sd_cfg.get("ucb_constant", math.sqrt(2)),
            max_per_task=skill_sd_cfg.get("max_skills_per_task", 10000),
            usage_pruning_threshold=skill_sd_cfg.get("usage_pruning_threshold", 5),
            cross_task_only=skill_sd_cfg.get("cross_task_only", False),
        )
        skill_evolution = SkillEvolution(
            bank=bank,
            writer=writer,
            embedder=embedder,
            selection_seed=config.data.get("seed", 1),
            max_workers=writer_cfg.get("max_workers", 16),
            max_attempts=writer_cfg.get("max_attempts", 3),
        )

        student_skill_injection = skill_sd_cfg.get("student_skill_injection", "none")
        if student_skill_injection == "none":
            from agent_system.multi_turn_rollout import TrajectoryCollector

            traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        else:
            from agent_system.multi_turn_rollout.skill_sd_collector import SkillPromptTrajectoryCollector

            traj_collector = SkillPromptTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor, skill_evolution=skill_evolution, inject_val=student_skill_injection == "train_and_val")

        from verl.trainer.data import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.skill_sd_ray_trainer import SkillSDRayTrainer

        trainer = SkillSDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_evolution=skill_evolution,
        )
        try:
            trainer.init_workers()
            trainer.fit()
        finally:
            skill_evolution.close()


if __name__ == "__main__":
    main()
