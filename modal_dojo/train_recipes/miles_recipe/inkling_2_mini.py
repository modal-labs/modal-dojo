"""Full-parameter Inkling 2 Mini with response-heavy 256k coding trajectories."""

from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models.inkling_2_mini import Inkling_2_Mini
from modal_dojo.common.models import ModelConfig
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "frameworks/miles/modal_helpers/patches"
)
_BF16_PATH = (
    f"/root/.cache/huggingface/dojo/Inkling-2-Mini/{Inkling_2_Mini.revision}/bf16"
)


def _image_commands() -> list[str]:
    return [
        f"echo {encode_patch('patch_inkling2_mini_long_context', _PATCH_DIR)}"
        " | base64 -d | python3"
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_2_Mini_256k_Recipe(MilesRecipe):
    """Full-parameter GRPO, 262,144-token context, 1 node × 8 B300 GPUs.

    Reserves 16k initial prompt tokens and 240k response tokens. Agentic coding
    rollouts supply exact generated tokens, logprobs, observation loss masks and
    executable-test rewards. Defaults to one proof step without a reward scorer.
    The throughput preset and full-length capacity require live validation.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Inkling_2_Mini
    # TorchTitan parses its own backend arguments; these belong to Megatron.
    _SKIP_FIELDS: ClassVar[frozenset[str]] = MilesRecipe._SKIP_FIELDS | {
        "attention_dropout",
        "hidden_dropout",
        "attention_softmax_in_fp32",
        "accumulate_allreduce_grads_in_fp32",
        "expert_tensor_parallel_size",
        "megatron_to_hf_mode",
    }
    # Miles #4981 with the matching Inkling SGLang build. Pin the multi-arch
    # manifest so a future image push cannot change the training implementation.
    docker_image: str = (
        "radixark/miles:inkling-2-mini@sha256:"
        "fd701e76034c497b2c9d14fed51dc54c7f98a75daf4b8ba32a03cfc69990fb17"
    )
    train_backend: str = "torchtitan"
    titan_model_name: str = "inkling2"
    titan_model_flavor: str = "looped"
    model_name: str = "inkling2"
    # TorchTitan loads HF weights directly; both runtimes use the same BF16 copy.
    megatron_to_hf_mode: str = ""
    source_hf_checkpoint: str = Inkling_2_Mini.model_name
    hf_checkpoint: str = _BF16_PATH

    gpu_type: str = "B300"
    actor_num_gpus_per_node: int = 8
    num_gpus_per_node: int = 8
    expert_model_parallel_size: int = 8
    memory: int | tuple[int, int] | None = (256 * 1024, 768 * 1024)
    cpu: float | tuple[float, float] | None = 32.0
    download_cpu: float | None = 16.0
    download_memory: int | None = 128 * 1024
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": 512 * 1024}
    )
    max_retries: int = 0

    rollout_batch_size: int = 1
    n_samples_per_prompt: int = 8
    global_batch_size: int = 8
    use_dynamic_batch_size: bool = False
    micro_batch_size: MicroBatchSize = 1
    max_tokens_per_gpu: MicroBatchSize = 262144
    balance_data: bool = True
    # The native loop checkpoints every layer visit itself. Wrapping its blocks
    # a second time with TorchTitan's generic rematerializer is unnecessary.
    gradient_checkpointing: bool = False
    update_weight_buffer_size: int | None = 512 * 1024 * 1024
    # Reclaim idle blocks before compiled kernels need CUDA driver allocations.
    # Expandable segments cannot be shared by the CUDA IPC weight updater on Modal.
    train_env_vars: dict | str | None = field(
        default_factory=lambda: {
            "PYTORCH_ALLOC_CONF": "garbage_collection_threshold:0.8"
        }
    )

    sglang_attention_backend: str | None = "fa4"
    sglang_bf16_gemm_backend: str = "torch"
    sglang_page_size: int = 128
    sglang_mem_fraction_static: float = 0.7
    sglang_disable_radix_cache: bool = True
    sglang_context_length: int = 262144
    sglang_max_running_requests: int | None = 1
    sglang_server_concurrency: int | None = 8
    sglang_chunked_prefill_size: int = 4096
    max_seq_len: int = 262144
    rollout_max_prompt_len: int = 16384
    rollout_max_response_len: int = 245760
    eval_max_response_len: int = 245760
    seq_length: int = 262144
    capture_trace: bool = True
    tito_model: str = "inkling"
    use_session_server: str = "v1"
    image_run_commands: list[str] = field(default_factory=_image_commands)
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/miles:/root/Megatron-LM",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
        }
    )

    @model_validator(mode="after")
    def _validate_titan_backend(self) -> "Inkling_2_Mini_256k_Recipe":
        overrides = self._escape_hatch_values()

        def value(name, default=None):
            return overrides.get(name, getattr(self, name, default))

        if value("train_backend") != "torchtitan" or value("lora_rank") is not None:
            raise ValueError(
                "Inkling 2 Mini requires full-parameter TorchTitan training"
            )
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        ):
            if value(name) != 1:
                raise ValueError(f"Upstream Inkling 2 Mini requires {name}=1")
        if value("sequence_parallel"):
            raise ValueError("Inkling 2 Mini does not support sequence_parallel")
        if value("loss_type") != "policy_loss" or value("qkv_format") != "thd":
            raise ValueError(
                "Inkling 2 Mini streaming scores require packed thd policy_loss"
            )
        for name in (
            "true_on_policy_mode",
            "use_sampling_support_replay",
            "recompute_loss_function",
        ):
            if value(name, False):
                raise ValueError(
                    f"Inkling 2 Mini streaming scores do not support {name}"
                )
        if value("use_dynamic_batch_size") or value("micro_batch_size") != 1:
            raise ValueError(
                "Inkling 2 Mini 256k requires one trajectory per microbatch"
            )
        required = _image_commands()
        if (self.image_run_commands or [])[: len(required)] != required:
            object.__setattr__(
                self,
                "image_run_commands",
                [
                    *required,
                    *(
                        command
                        for command in self.image_run_commands or []
                        if command not in required
                    ),
                ],
            )
        return self

    def post_process_model(self) -> None:
        from modal_dojo.frameworks.miles.inkling2_mini_long_context import prepare_bf16

        if self.hf_checkpoint == _BF16_PATH:
            prepare_bf16(
                Inkling_2_Mini.model_name, Inkling_2_Mini.revision, self.hf_checkpoint
            )
