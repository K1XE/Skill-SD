<div align="center">

# Skill-SD

**Skill-Conditioned Self-Distillation for Multi-turn LLM Agents**

[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13.svg)](https://openreview.net/pdf?id=ZeThM3k7x0)
[![Project](https://img.shields.io/badge/Project-Website-2F80ED.svg)](https://skill-sd.github.io/)
[![Logs](https://img.shields.io/badge/Logs-W%26B-FFBE00.svg)](https://wandb.ai/kize-null/skill-sd)
[![License](https://img.shields.io/badge/License-Apache--2.0-4C1.svg)](./LICENSE)

</div>

Skill-SD turns past trajectories into reusable skills and distills a skill-conditioned teacher into the student. The student uses the plain task prompt at both training and inference.

![Skill-SD overview](https://skill-sd.github.io/assets/skill-sd-overview.png)

## Setup

Python 3.12 · PyTorch 2.8.0 · vLLM 0.11.0 · ALFWorld 0.4.2

```bash
git clone https://github.com/K1XE/Skill-SD.git
cd Skill-SD
conda create -n skill-sd python=3.12 -y
conda activate skill-sd
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation

export ALFWORLD_DATA=$HOME/data/alfworld
alfworld-download -f
```

## Reproduce

Training uses **8 GPUs** (reference: H200) and a frozen skill writer served separately on **4 GPUs**.

```bash
# Writer server (separate terminal / GPUs)
bash examples/skill_sd_trainer/serve_writer.sh Qwen/Qwen3-4B-Instruct-2507

# Training
export WRITER_URL="http://<writer-host>:18080/v1"
wandb login
bash examples/skill_sd_trainer/run_alfworld.sh skill_sd 43
bash examples/skill_sd_trainer/run_alfworld.sh grpo 43
```

Default student: **Qwen3-4B-Instruct-2507**; seeds: **43, 44**. Training uses 3,553 games; evaluation samples 128 episodes from `valid_seen`. Report `val/success_rate` at **step 150**.

<details>
<summary><b>Baselines, ablations, and other settings</b></summary>

Use `run_alfworld.sh <arm> <seed>` with any of these arms:

| Arm | Setting |
| --- | --- |
| `skill_sd` | Skill-SD (skills condition the teacher only) |
| `grpo` | Vanilla GRPO |
| `skill_grpo` | Skills in the policy prompt during training |
| `skill_grpo_matched` | Skills in the policy prompt during training and evaluation |
| `trajectory` | Retrieved successful trajectory as teacher context |
| `cross_task` | Skills retrieved from other task families |

For `cross_task`, run `bash examples/skill_sd_trainer/serve_embedder.sh Qwen/Qwen3-Embedding-8B` and set `EMBEDDER_URL="http://<embedder-host>:18081/v1"`.

</details>

## Citation

```bibtex
@inproceedings{wang2026skillsd,
  title={Skill-{SD}: Skill-Conditioned Self-Distillation for Multi-turn {LLM} Agents},
  author={Hao Wang and Guozhi Wang and Han Xiao and Yufeng Zhou and Yue Pan and Jichao Wang and Ke Xu and Yafei Wen and Xiaohu Ruan and Xiaoxin Chen and Honggang Qi},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=ZeThM3k7x0}
}
```

## Acknowledgements

Built on [SDAR](https://github.com/ZJU-REAL/SDAR), [verl-agent](https://github.com/langfengQ/verl-agent), [verl](https://github.com/volcengine/verl), and [ALFWorld](https://github.com/alfworld/alfworld). See [LICENSE](./LICENSE) and [Notice.txt](./Notice.txt).
