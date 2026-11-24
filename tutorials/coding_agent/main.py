# ---
# order: 2
# ---
#
# # Training a coding agent on competitive programming with Harbor

# Start by importing the necessary packages.

from modal_dojo import (
    HarborAgent,
    HarborTasks,
    Qwen3_5_4B,
    Qwen3_5_4B_Miles_Recipe,
    TrainConfig,
)

from tutorials.coding_agent import tasks

# We want to penalize completions to stay below our `max_seq_len` (the rollout's max length). We noticed a lot of truncations (rollout hitting context limits) and length penalty helps reduce that.

def length_penalty(args, samples):
    from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
    from miles.utils.types import Sample

    start = 0.75 * args.max_seq_len

    def penalty(s):
        if s.status == Sample.Status.TRUNCATED:
            return 0.5
        return 0.5 * min(1.0, max(0.0, (len(s.tokens) - start) / (args.max_seq_len - start)))

    raw = [float(s.get_reward_value(args)) for s in samples]
    shaped = [r - penalty(s) for r, s in zip(raw, samples, strict=True)]
    return raw, _normalize_rewards_by_rollout(args, samples, shaped, None)


# We can define the agent harness we want to train on. For this example, we turn off summarization.

agent = HarborAgent("terminus-2", enable_summarize=False, record_terminal_session=False)

# Then, define the TrainConfig and drop the train and eval dataset in your config

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=HarborTasks(agent=agent, modal_volume=tasks.TASKS_VOLUME, path="code-contests", task_glob="train-*"),
    eval_dataset=HarborTasks(agent=agent, modal_volume=tasks.TASKS_VOLUME, path="code-contests", task_glob="eval-*"),
    recipe=Qwen3_5_4B_Miles_Recipe(
        name="pi-harbor-code-contests",
        # a Miles build with the session server
        docker_image="radixark/miles:dev-202610140350@sha256:c53f65d346a6b364cf89d7dbaa416595a2d5638670b5d82c683214183dca10bd",
        miles_git_ref="d33406f1139f1fa66305b17b9e91dfa110e3c4ce",
        ref_load="/checkpoints/Qwen3.5-4B_torch_dist_miles_d33406f",
        # three 8xH200 nodes: two inference nodes and one training node, connected by RDMA
        gpu_type="H200",
        cpu=64,
        colocate=False,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=16,
        async_mode=True,
        max_tokens_per_gpu=16384,
        num_rollout=100,
        rollout_batch_size=32,
        n_samples_per_prompt=16,
        global_batch_size=512,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        eval_interval=10,
        save_interval=10,
        kl_loss_coef=0.01,
        # resample and replace problems where all attempts in a group pass or all fail (no signal) and sample new ones
        dynamic_sampling_filter_path="miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter",
        custom_reward_post_process_function=length_penalty,
        extra_config={
            "max_seq_len": 32768,
            "use_session_server": True,
            "tito_model": "qwen35",
            "log_probs_chunk_size": 4096,
            "fully_async": True,
            "session_server_workers": 32
        },
    ),
)

# Let it rip!

if __name__ == "__main__":
    # download and filter the tasks into a Modal Volume and training run mounts it
    with tasks.app.run():
        tasks.write_tasks.remote()
    config.launch()
