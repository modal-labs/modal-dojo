"""Full-parameter Inkling 2 Mini with a 128k context window."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.inkling_2_mini import (
    Inkling_2_Mini_64k_Recipe,
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_2_Mini_128k_Recipe(Inkling_2_Mini_64k_Recipe):
    """Full-parameter GRPO, 131,072-token context, 1 node × 8 B300 GPUs.

    Reserves 16k initial prompt tokens and 112k response tokens. Two prompts
    with four attempts each fill the eight TP1 inference replicas; one active
    episode per replica bounds evaluation queues as well as training rollouts.
    """

    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    sglang_server_concurrency: int | None = 1
    max_tokens_per_gpu: MicroBatchSize = 131072
    max_seq_len: int = 131072
    seq_length: int = 131072
    sglang_context_length: int = 131072
    rollout_max_response_len: int = 114688
    eval_max_response_len: int = 114688
