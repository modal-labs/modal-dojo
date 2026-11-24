"""Qwen3.8-27B LoRA on Miles, fully async, with a 131k context."""

from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_8_27B
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_8_27B_Miles_LoRA_Recipe(MilesRecipe):
    """Qwen3.8-27B rank-16 bridge LoRA, fully async, 131k context on 16 H200s.

    One node trains at TP2 x DP4 and the other serves two TP4 engines.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_8_27B
    docker_image: str = (
        "modalresearch/miles:dojo-dev-202610140350@sha256:"
        "97438bed2a81d8e114f4e218331e9bfce8a8cb6e0d8eb21361261d8cf888903c"
    )
    miles_git_ref: str | None = "d33406f1139f1fa66305b17b9e91dfa110e3c4ce"
    miles_model_name: str = "qwen3.5-27B"

    gpu_type: str = "H200"
    memory: int | tuple[int, int] | None = (128, 2_097_152)
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )
    colocate: bool = False
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int | None = 8
    rollout_num_gpus_per_engine: int = 4
    tensor_model_parallel_size: int = 2

    async_mode: bool = True
    fully_async: bool = True
    async_max_concurrent_samples: int = 80
    max_weight_staleness: int = 3

    lora_rank: int | None = 16
    lora_alpha: int | None = 32
    lora_dropout: float | None = 0.0
    target_modules: str | None = "all-linear"
    lora_base_cpu_backup: bool = True

    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    rollout_max_context_len: int = 131072
    rollout_max_prompt_len: int = 98304
    rollout_max_response_len: int = 32768
    lr: float = 1e-4
    use_tis: bool = True
    dynamic_sampling_filter_path: str | None = (
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    )

    qkv_format: str = "bshd"
    use_dynamic_batch_size: bool = False
    micro_batch_size: MicroBatchSize = 1
    seq_length: int = 131072
    max_position_embeddings: int = 131072
    max_tokens_per_gpu: MicroBatchSize = 131072
    log_probs_backend: str = "fused"
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    sequence_parallel: bool = True
    distributed_timeout_minutes: int = 30
    environment: dict[str, str] = field(
        default_factory=lambda: {
            **MilesRecipe().environment,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    sglang_context_length: int = 131072
    sglang_mem_fraction_static: float = 0.85
    sglang_max_running_requests: int | None = 48
    sglang_tool_call_parser: str | None = "qwen3_coder"
    sglang_reasoning_parser: str | None = "qwen3"
    sglang_lora_backend: str | None = "triton"


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_8_27B_Miles_LoRA_256k_Recipe(Qwen3_8_27B_Miles_LoRA_Recipe):
    """The same recipe with a 262k context."""

    log_probs_backend: str = "fused"
    rollout_max_context_len: int = 262144
    rollout_max_prompt_len: int = 229376
    seq_length: int = 262144
    max_position_embeddings: int = 262144
    max_tokens_per_gpu: MicroBatchSize = 262144
    sglang_context_length: int = 262144
