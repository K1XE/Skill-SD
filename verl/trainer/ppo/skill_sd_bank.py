import math
import uuid
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SkillRecord:
    id: str
    task_id: str
    created_iteration: int
    source_trajectory_id: str
    source_reward: float
    success_analysis: str = ""
    mistake_analysis: str = ""
    golden_workflow: str = ""
    trajectory_text: str = ""
    selection_count: int = 0
    reward_count: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.reward_count if self.reward_count else 0.0

    def privileged_text(self) -> str:
        if self.trajectory_text:
            return f"Successful Trajectory:\n{self.trajectory_text}\n"
        return f"Success Analysis: {self.success_analysis}\n\nMistake Analysis: {self.mistake_analysis}\n\nGolden Workflow: {self.golden_workflow}\n"


class SkillBank:
    def __init__(self, ucb_constant: float, max_per_task: int, usage_pruning_threshold: int, cross_task_only: bool):
        self.ucb_constant = ucb_constant
        self.max_per_task = max_per_task
        self.usage_pruning_threshold = usage_pruning_threshold
        self.cross_task_only = cross_task_only
        self.pruned_count = 0
        self._records: dict[str, list[SkillRecord]] = {}
        self._index: dict[str, SkillRecord] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    def add(self, task_id: str, created_iteration: int, source_trajectory_id: str, source_reward: float, summary: dict[str, str] | None = None, trajectory_text: str = "") -> SkillRecord:
        record = SkillRecord(
            id=str(uuid.uuid4()),
            task_id=task_id,
            created_iteration=created_iteration,
            source_trajectory_id=source_trajectory_id,
            source_reward=source_reward,
            trajectory_text=trajectory_text,
            **(summary or {}),
        )
        records = self._records.setdefault(task_id, [])
        records.append(record)
        self._index[record.id] = record
        if len(records) > self.max_per_task:
            self._remove(records[0])
        return record

    def _remove(self, record: SkillRecord) -> None:
        self._records[record.task_id].remove(record)
        self._index.pop(record.id)
        self._embeddings.pop(record.id, None)

    def _candidates(self, task_id: str) -> list[SkillRecord]:
        if not self.cross_task_only:
            return self._records.get(task_id, [])
        return [record for other_task, records in self._records.items() if other_task != task_id for record in records]

    def _ucb_bonus(self, record: SkillRecord, total_retrievals: int) -> float:
        return self.ucb_constant * math.sqrt(math.log(max(total_retrievals, 1)) / record.selection_count)

    def _ucb_winner(self, records: list[SkillRecord]) -> SkillRecord:
        total = sum(record.selection_count for record in records)
        return max(records, key=lambda record: record.mean_reward + self._ucb_bonus(record, total))

    def select(self, task_id: str, draw_keys: list[int]) -> list[SkillRecord | None]:
        records = self._candidates(task_id)
        if not records:
            return [None] * len(draw_keys)
        unseen = [record for record in records if record.selection_count == 0]
        if unseen:
            selected = [unseen[key % len(unseen)] for key in draw_keys]
        else:
            selected = [self._ucb_winner(records)] * len(draw_keys)
        for record in selected:
            record.selection_count += 1
        return selected

    def peek(self, task_id: str) -> SkillRecord | None:
        records = self._candidates(task_id)
        if not records:
            return None
        unseen = [record for record in records if record.selection_count == 0]
        return unseen[0] if unseen else self._ucb_winner(records)

    def similarity(self, query: np.ndarray) -> dict[str, float]:
        if not self._embeddings:
            return {}
        values = np.stack(list(self._embeddings.values())) @ (query / np.linalg.norm(query))
        return dict(zip(self._embeddings, values.tolist()))

    def select_similar(self, task_id: str, similarity: dict[str, float]) -> SkillRecord | None:
        records = [record for record in self._candidates(task_id) if record.id in similarity]
        if not records:
            return None
        unseen = [record for record in records if record.selection_count == 0]
        if unseen:
            selected = max(unseen, key=lambda record: similarity[record.id])
        else:
            total = sum(record.selection_count for record in records)
            selected = max(records, key=lambda record: similarity[record.id] + self._ucb_bonus(record, total))
        selected.selection_count += 1
        return selected

    def set_embedding(self, skill_id: str, vector: np.ndarray) -> None:
        if skill_id in self._index:
            self._embeddings[skill_id] = vector / np.linalg.norm(vector)

    def record_reward(self, skill_id: str, reward: float) -> str | None:
        record = self._index[skill_id]
        record.reward_count += 1
        record.reward_sum += reward
        threshold = self.usage_pruning_threshold
        if record.reward_count < threshold:
            return None
        qualified = [other for other in self._records[record.task_id] if other.reward_count >= threshold]
        if len(qualified) < 2:
            return None
        average = sum(other.reward_sum for other in qualified) / sum(other.reward_count for other in qualified)
        if record.mean_reward >= average:
            return None
        self._remove(record)
        self.pruned_count += 1
        return record.id

    def records(self) -> list[SkillRecord]:
        return list(self._index.values())

    def stats(self) -> dict[str, int]:
        return {"tasks": len(self._records), "skills": len(self._index), "pruned": self.pruned_count}

    def to_state(self) -> dict:
        return {"pruned_count": self.pruned_count, "records": [asdict(record) for records in self._records.values() for record in records]}

    def load_state(self, state: dict) -> None:
        self.pruned_count = state["pruned_count"]
        self._records, self._index, self._embeddings = {}, {}, {}
        for raw in state["records"]:
            record = SkillRecord(**raw)
            self._records.setdefault(record.task_id, []).append(record)
            self._index[record.id] = record
