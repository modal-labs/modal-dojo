"""Full-parameter Inkling 2 Mini with a 256k context window."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.inkling_2_mini_128k import (
    Inkling_2_Mini_128k_Recipe,
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_2_Mini_256k_Recipe(Inkling_2_Mini_128k_Recipe):
    """Full-parameter GRPO, 262,144-token context, 1 node × 8 B300 GPUs.

    Reserves 16k initial prompt tokens and 240k response tokens. Four prompts
    with four attempts each provide sixteen trajectories per optimizer update;
    eight TP1 replicas run one agent episode each at a time.
    """

    rollout_batch_size: int = 4
    global_batch_size: int = 16
    max_tokens_per_gpu: MicroBatchSize = 262144
    max_seq_len: int = 262144
    seq_length: int = 262144
    sglang_context_length: int = 262144
    rollout_max_response_len: int = 245760
    eval_max_response_len: int = 245760
    # Bound the cache to two full windows for each single-request replica.
    sglang_max_total_tokens: int = 524288
