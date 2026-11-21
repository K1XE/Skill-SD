set -x

ARM=${1:?usage: run_alfworld.sh <skill_sd|grpo|skill_grpo|skill_grpo_matched|trajectory|cross_task> <seed>}
SEED=${2:?usage: run_alfworld.sh <arm> <seed>}
shift 2

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
SDL_LAMBDA=${SDL_LAMBDA:-0.001}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-skill_sd_alfworld_${ARM}_seed${SEED}}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}

num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8

case $ARM in
    skill_sd|skill_grpo|skill_grpo_matched|cross_task)
        arm_args=(
            +algorithm.skill_sd.writer.base_url=${WRITER_URL:?set WRITER_URL to the skill writer endpoint, e.g. http://localhost:18080/v1}
            +algorithm.skill_sd.writer.model=${WRITER_MODEL:-skill-writer}
        )
        ;;
    grpo|trajectory)
        arm_args=()
        ;;
    *)
        echo "unknown arm: $ARM" >&2
        exit 1
        ;;
esac
case $ARM in
    grpo)
        SDL_LAMBDA=0
        ;;
    skill_grpo)
        SDL_LAMBDA=0
        arm_args+=(+algorithm.skill_sd.student_skill_injection=train_only)
        ;;
    skill_grpo_matched)
        SDL_LAMBDA=0
        arm_args+=(+algorithm.skill_sd.student_skill_injection=train_and_val)
        ;;
    trajectory)
        arm_args+=(+algorithm.skill_sd.privileged_context=trajectory)
        ;;
    cross_task)
        arm_args+=(
            +algorithm.skill_sd.cross_task_only=True
            +algorithm.skill_sd.embedder.base_url=${EMBEDDER_URL:?set EMBEDDER_URL to the embedding endpoint, e.g. http://localhost:18081/v1}
            +algorithm.skill_sd.embedder.model=${EMBEDDER_MODEL:-skill-embedder}
        )
        ;;
esac

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_skill_sd \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    +algorithm.skill_sd.sdl_lambda=$SDL_LAMBDA \
    "${arm_args[@]}" \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    +data.seed=$SEED \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    +actor_rollout_ref.rollout.seed=$SEED \
    +actor_rollout_ref.rollout.engine_seed_only=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.72 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=$SEED \
    env.max_steps=50 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.alfworld.eval_dataset=eval_in_distribution \
    trainer.logger=['console','wandb'] \
    trainer.project_name='skill_sd_alfworld' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.resume_mode=auto \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq=10 \
    trainer.log_val_generations=16 \
    trainer.total_epochs=150 \
    trainer.total_training_steps=150 \
    trainer.val_before_train=True "$@"
