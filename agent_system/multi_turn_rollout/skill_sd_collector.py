from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from verl.trainer.ppo.skill_sd_ray_trainer import alfworld_task_type


class SkillPromptTrajectoryCollector(TrajectoryCollector):
    def __init__(self, config, tokenizer, processor, skill_evolution, inject_val: bool):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor)
        self.skill_evolution = skill_evolution
        self.inject_val = inject_val
        self.inject = False
        self.episode_skills = {}

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        self.inject = is_train or self.inject_val
        self.episode_skills = {}
        return super().multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train)

    def preprocess_batch(self, gen_batch, obs):
        if self.inject:
            obs = {**obs, "text": [self._with_skill(text, gamefile) for text, gamefile in zip(obs["text"], obs["gamefile"])]}
        return super().preprocess_batch(gen_batch=gen_batch, obs=obs)

    def _with_skill(self, text: str, gamefile: str) -> str:
        task_id = alfworld_task_type(gamefile)
        if task_id not in self.episode_skills:
            record = self.skill_evolution.bank.peek(task_id)
            if record is None:
                return text
            self.episode_skills[task_id] = record.privileged_text()
        return f"[Privileged Skill Information]\n{self.episode_skills[task_id]}\n\n{text}"
