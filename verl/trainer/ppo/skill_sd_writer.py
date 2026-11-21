import hashlib
import json
import logging
import os
import re
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from verl.trainer.ppo.skill_sd_bank import SkillBank, SkillRecord

logger = logging.getLogger(__name__)

SKILL_FIELDS = ("success_analysis", "mistake_analysis", "golden_workflow")

SUMMARY_PROMPT = """# Trajectory Analysis and Golden Experience Generation

You are an expert trajectory analyst. Your task is to analyze the given trajectory and generate a structured golden experience that captures key learnings.

## Conversation History
{conversation_history}

## Trajectory Outcome
- **Result**: {result_status}
- **Total Steps**: {step_count}
- **Completion Rate**: {completion_rate:.2%}
- **Task Goal Completion (TGC)**: {tgc:.2%}

---

## Output Requirements (JSON)

You need to deeply analyze the Trajectory Outcome and Conversation History I provided, thinking step by step, and then, based on the above information, summarize the experience (to guide the model when it encounters this task again in the future).

Output a unique JSON object containing exactly these 3 fields:

1.  `success_analysis`: A comprehensive paragraph summarizing the **strategic intent** that led (or would lead) to success.
2.  `mistake_analysis`: A comprehensive paragraph analyzing the **root causes** of failures or inefficiencies observed in the trajectory. What mindsets or logical fallacies should be avoided?
3.  `golden_workflow`: A concise description of the **ideal overall process verification** to complete this task. This describes the standard flowchart from start to finish.

## REMEMBER

The analysis must be concise. As an expert, the experience you summarize will guide the model's subsequent operations. Therefore, it's crucial to adopt a guiding approach, focusing primarily on the steps that confuse the model.
Your success-to-failure output ratio should follow the Completion Rate ({completion_rate:.2%}). If the Completion Rate is high, you should focus on summarizing the success analysis; otherwise, you should focus on summarizing the mistake analysis.

## JSON Output Template

```json
{{
    "success_analysis": "Your key to success is ...",
    "mistake_analysis": "Avoid ..., pay attention ...",
    "golden_workflow": "To complete this task, You should: step 1. ..., step 2. ..., step n. ..."
}}
"""

ALFWORLD_TASK_QUERIES = {
    "pick_and_place": "Goal: put an object in or on a receptacle. Workflow: search the likely locations for the target object, take it, go to the destination receptacle, and place the object there.",
    "pick_two_obj_and_place": "Goal: put two instances of the same object type in or on a receptacle. Workflow: find and take the first object, carry it to the destination and place it, then search the remaining locations for a second object of the same type and place it in the same receptacle.",
    "look_at_obj_in_light": "Goal: examine an object under a light source. Workflow: find and take the object, go to a floor lamp or desk lamp, turn the lamp on, and look at the object while holding it in the light.",
    "pick_heat_then_place_in_recep": "Goal: heat an object and put it in a receptacle. Workflow: find and take the object, go to the microwave, open it if needed, heat the object with the microwave, then carry it to the destination receptacle and place it there.",
    "pick_cool_then_place_in_recep": "Goal: cool an object and put it in a receptacle. Workflow: find and take the object, go to the fridge, open it if needed, cool the object with the fridge, then carry it to the destination receptacle and place it there.",
    "pick_clean_then_place_in_recep": "Goal: clean an object and put it in a receptacle. Workflow: find and take the object, go to the sink basin, clean the object with the sink, then carry it to the destination receptacle and place it there.",
}


@dataclass(frozen=True)
class SummaryRequest:
    task_id: str
    trajectory_id: str
    iteration: int
    transcript: str
    reward: float
    step_count: int
    id: str = ""
    attempts: int = 0

    def prompt(self) -> str:
        result_status = "SUCCESS" if self.reward >= 1.0 else "PARTIAL_COMPLETION" if self.reward > 0.0 else "FAILED"
        conversation_history = "\n".join(part for part in (f"TASK: {self.task_id}", self.transcript) if part)
        return SUMMARY_PROMPT.format(
            conversation_history=conversation_history,
            result_status=result_status,
            step_count=self.step_count,
            completion_rate=self.reward,
            tgc=self.reward,
        )


def parse_skill_summary(content: str) -> dict[str, str]:
    text = content.strip()
    candidates = [text]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    balanced = _first_balanced_object(text)
    if balanced is not None and len(balanced) >= len(text) * 0.5:
        candidates.append(balanced)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if set(parsed) != set(SKILL_FIELDS) or not all(isinstance(parsed[field], str) and parsed[field].strip() for field in SKILL_FIELDS):
            raise ValueError(f"skill summary must contain exactly {SKILL_FIELDS} as non-empty strings")
        return {field: parsed[field].strip() for field in SKILL_FIELDS}
    raise ValueError("response is not a valid three-field skill JSON object")


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string and char == "{":
            depth += 1
        elif not in_string and char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _post_json(url: str, body: dict, timeout: float, api_key_env: str) -> dict:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class SkillWriter:
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, timeout: float):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def write(self, request: SummaryRequest) -> dict[str, str]:
        body = {"model": self.model, "messages": [{"role": "user", "content": request.prompt()}], "temperature": self.temperature, "max_tokens": self.max_tokens}
        choice = _post_json(self.url, body, self.timeout, "SKILL_SD_WRITER_API_KEY")["choices"][0]
        if choice.get("finish_reason") in ("length", "max_tokens"):
            raise ValueError("skill writer exhausted max_tokens")
        return parse_skill_summary(choice["message"]["content"])


class SkillEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float, max_batch: int):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.timeout = timeout
        self.max_batch = max_batch
        self.failed = 0

    def embed(self, texts: list[str]) -> list[np.ndarray | None]:
        vectors = []
        for start in range(0, len(texts), self.max_batch):
            chunk = texts[start : start + self.max_batch]
            try:
                data = _post_json(self.url, {"model": self.model, "input": chunk}, self.timeout, "SKILL_SD_EMBEDDER_API_KEY")["data"]
                vectors.extend(np.asarray(item["embedding"], dtype=np.float64) for item in sorted(data, key=lambda item: item["index"]))
            except Exception as exc:
                logger.warning("skill embedding failed: %s", exc)
                self.failed += len(chunk)
                vectors.extend([None] * len(chunk))
        return vectors


class SkillEvolution:
    def __init__(self, bank: SkillBank, writer: SkillWriter | None, embedder: SkillEmbedder | None, selection_seed: int, max_workers: int, max_attempts: int):
        self.bank = bank
        self.writer = writer
        self.embedder = embedder
        self.selection_seed = selection_seed
        self.max_attempts = max_attempts
        self.iteration = 0
        self.failures = 0
        self.pending: dict[str, SummaryRequest] = {}
        self._futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="skill-sd-writer")
        self.task_queries = {}
        if embedder is not None:
            vectors = embedder.embed(list(ALFWORLD_TASK_QUERIES.values()))
            if any(vector is None for vector in vectors):
                raise RuntimeError("failed to embed the task queries")
            self.task_queries = dict(zip(ALFWORLD_TASK_QUERIES, vectors))

    def start_iteration(self, iteration: int) -> int:
        self.iteration = iteration
        added = self._harvest()
        for request_id, request in list(self.pending.items()):
            if request_id in self._futures:
                continue
            if request.attempts >= self.max_attempts:
                self.pending.pop(request_id)
                self.failures += 1
            else:
                self._futures[request_id] = self._executor.submit(self.writer.write, request)
        return added

    def select(self, trajectories: list[tuple[str, str]]) -> list[tuple[str, SkillRecord | None]]:
        if self.embedder is not None:
            similarity = {task_id: self.bank.similarity(self.task_queries[task_id]) for task_id in {task_id for _, task_id in trajectories}}
            return [(trajectory_id, self.bank.select_similar(task_id, similarity[task_id])) for trajectory_id, task_id in trajectories]
        indices_by_task: dict[str, list[int]] = {}
        for index, (_, task_id) in enumerate(trajectories):
            indices_by_task.setdefault(task_id, []).append(index)
        records = [None] * len(trajectories)
        for task_id, indices in indices_by_task.items():
            draw_keys = [self._draw_key(task_id, ordinal) for ordinal in range(len(indices))]
            for index, record in zip(indices, self.bank.select(task_id, draw_keys)):
                records[index] = record
        return [(trajectory_id, record) for (trajectory_id, _), record in zip(trajectories, records)]

    def _draw_key(self, task_id: str, ordinal: int) -> int:
        payload = json.dumps(["sha256-v1", self.selection_seed, self.iteration, task_id, ordinal], ensure_ascii=False, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), byteorder="big")

    def record_rewards(self, selections: list[tuple[str, SkillRecord | None]], rewards: dict[str, float]) -> None:
        pruned = set()
        for trajectory_id, record in selections:
            if record is None or record.id in pruned:
                continue
            pruned_id = self.bank.record_reward(record.id, float(rewards[trajectory_id] >= 1.0))
            if pruned_id is not None:
                pruned.add(pruned_id)

    def submit(self, requests: list[SummaryRequest]) -> None:
        for request in requests:
            request = replace(request, id=str(uuid.uuid4()))
            self.pending[request.id] = request
            self._futures[request.id] = self._executor.submit(self.writer.write, request)

    def add_trajectories(self, requests: list[SummaryRequest], texts: list[str]) -> int:
        added = 0
        for request, text in zip(requests, texts):
            if request.reward > 0.0 and text.strip():
                self.bank.add(request.task_id, request.iteration, request.trajectory_id, request.reward, trajectory_text=text)
                added += 1
        return added

    def drain(self) -> None:
        wait(list(self._futures.values()))
        self._harvest()

    def _harvest(self) -> int:
        committed = []
        for request_id, future in list(self._futures.items()):
            if not future.done():
                continue
            self._futures.pop(request_id)
            request = self.pending[request_id]
            try:
                summary = future.result()
            except Exception as exc:
                logger.warning("skill writer failed for trajectory %s: %s", request.trajectory_id, exc)
                retry = replace(request, attempts=request.attempts + 1)
                if retry.attempts >= self.max_attempts:
                    self.pending.pop(request_id)
                    self.failures += 1
                else:
                    self.pending[request_id] = retry
                continue
            self.pending.pop(request_id)
            committed.append(self.bank.add(request.task_id, request.iteration, request.trajectory_id, request.reward, summary=summary))
        if committed and self.embedder is not None:
            self._embed(committed)
        return len(committed)

    def _embed(self, records: list[SkillRecord]) -> None:
        for record, vector in zip(records, self.embedder.embed([record.privileged_text() for record in records])):
            if vector is not None:
                self.bank.set_embedding(record.id, vector)

    def stats(self) -> dict[str, float]:
        stats = {**self.bank.stats(), "pending": len(self.pending), "writer_failures": self.failures}
        if self.embedder is not None:
            stats["embedding_failures"] = self.embedder.failed
        return stats

    def save(self, path: Path) -> None:
        state = {"iteration": self.iteration, "failures": self.failures, "bank": self.bank.to_state(), "pending": [asdict(request) for request in self.pending.values()]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        path.with_suffix(".tmp").replace(path)

    def load(self, path: Path) -> None:
        state = json.loads(path.read_text(encoding="utf-8"))
        self.iteration = state["iteration"]
        self.failures = state["failures"]
        self.bank.load_state(state["bank"])
        self.pending = {raw["id"]: SummaryRequest(**raw) for raw in state["pending"]}
        if self.embedder is not None:
            self._embed(self.bank.records())

    def close(self) -> None:
        self.drain()
        self._executor.shutdown()
