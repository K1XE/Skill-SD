import math

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.skill_sd_bank import SkillBank
from verl.trainer.ppo.skill_sd_ray_trainer import alfworld_task_type, build_teacher_batch
from verl.trainer.ppo.skill_sd_writer import SkillEvolution, SummaryRequest, parse_skill_summary

SUMMARY = {"success_analysis": "s", "mistake_analysis": "m", "golden_workflow": "g"}


def make_bank(cross_task_only=False):
    return SkillBank(ucb_constant=math.sqrt(2), max_per_task=10000, usage_pruning_threshold=5, cross_task_only=cross_task_only)


def test_unseen_skills_are_drawn_before_ucb():
    bank = make_bank()
    first = bank.add("pick_and_place", 1, "a", 1.0, summary=SUMMARY)
    second = bank.add("pick_and_place", 1, "b", 0.0, summary=SUMMARY)
    assert bank.select("pick_and_place", [0, 1, 2]) == [first, second, first]
    assert (first.selection_count, second.selection_count) == (2, 1)
    assert bank.select("look_at_obj_in_light", [0]) == [None]
    assert bank.peek("pick_and_place") is bank._ucb_winner([first, second])
    assert (first.selection_count, second.selection_count) == (2, 1)


def test_cross_task_bank_excludes_own_family():
    bank = make_bank(cross_task_only=True)
    own = bank.add("pick_and_place", 1, "a", 1.0, summary=SUMMARY)
    other = bank.add("look_at_obj_in_light", 1, "b", 1.0, summary=SUMMARY)
    bank.set_embedding(own.id, np.array([1.0, 0.0]))
    bank.set_embedding(other.id, np.array([0.0, 1.0]))
    assert bank.select_similar("pick_and_place", bank.similarity(np.array([1.0, 0.0]))) is other


def test_below_average_skill_is_pruned():
    bank = make_bank()
    good = bank.add("pick_and_place", 1, "a", 1.0, summary=SUMMARY)
    bad = bank.add("pick_and_place", 1, "b", 1.0, summary=SUMMARY)
    for _ in range(5):
        good.selection_count += 1
        assert bank.record_reward(good.id, 1.0) is None
    for _ in range(5):
        bad.selection_count += 1
        pruned = bank.record_reward(bad.id, 0.0)
    assert pruned == bad.id
    assert bank.records() == [good]


def test_rewards_are_binarized_and_pruned_skills_skipped():
    bank = make_bank()
    evolution = SkillEvolution(bank=bank, writer=None, embedder=None, selection_seed=43, max_workers=1, max_attempts=3)
    evolution.start_iteration(1)
    assert evolution.select([("t0", "pick_and_place")]) == [("t0", None)]
    record = bank.add("pick_and_place", 1, "a", 1.0, summary=SUMMARY)
    evolution.start_iteration(2)
    selections = evolution.select([("t0", "pick_and_place"), ("t1", "pick_and_place")])
    evolution.record_rewards(selections, {"t0": 1.0, "t1": 0.5})
    assert (record.reward_count, record.reward_sum) == (2, 1.0)
    evolution.close()


def test_parse_skill_summary():
    assert parse_skill_summary('```json\n{"success_analysis": " s ", "mistake_analysis": "m", "golden_workflow": "g"}\n```') == SUMMARY
    with pytest.raises(ValueError):
        parse_skill_summary('{"success_analysis": "s", "mistake_analysis": "m"}')
    with pytest.raises(ValueError):
        parse_skill_summary("no json here")


def test_prompt_reports_outcome():
    request = SummaryRequest(task_id="pick_and_place", trajectory_id="t", iteration=1, transcript="USER: a\nASSISTANT: b", reward=1.0, step_count=3)
    assert "TASK: pick_and_place\nUSER: a" in request.prompt()
    assert "**Result**: SUCCESS" in request.prompt()
    assert "**Completion Rate**: 100.00%" in request.prompt()


def test_alfworld_task_type():
    assert alfworld_task_type("/data/json_2.1.1/train/pick_two_obj_and_place-Apple-None-Fridge-1/trial/game.tw-pddl") == "pick_two_obj_and_place"
    with pytest.raises(ValueError):
        alfworld_task_type("/data/unknown")


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [7] * 3


def test_teacher_batch_prepends_skill_and_keeps_responses():
    input_ids = torch.tensor([[0, 0, 5, 6, 9, 9], [0, 4, 5, 6, 9, 0]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]])
    batch = DataProto.from_dict(tensors={"input_ids": input_ids, "attention_mask": attention_mask, "responses": input_ids[:, 4:].clone()})
    batch.non_tensor_batch["traj_uid"] = np.array(["a", "b"], dtype=object)
    bank = make_bank()
    record = bank.add("pick_and_place", 1, "x", 1.0, summary=SUMMARY)
    teacher = build_teacher_batch(batch, {"a": record, "b": None}, Tokenizer(), max_prompt_length=4)
    assert teacher.batch["input_ids"].tolist() == [[7, 7, 7, 5, 6, 9, 9], [0, 0, 4, 5, 6, 9, 0]]
    assert teacher.batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0]]
    assert teacher.batch["position_ids"][0].tolist() == [0, 1, 2, 3, 4, 5, 6]
