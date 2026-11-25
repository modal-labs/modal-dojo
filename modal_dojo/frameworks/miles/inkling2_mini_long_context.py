"""Bound Inkling 2 Mini intermediates without changing its loop or policy loss.

Torch and Miles are runtime dependencies of the pinned image, not of Dojo.
The native BF16/FP32 head, softcap, attention masks and residual equations are
preserved; chunks only bound temporary tensors and checkpoint recomputation.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _flex_compiled():
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    compiled = torch.compile(
        flex_attention, dynamic=False, fullgraph=True, recompile_limit=1024
    )

    def run(*args, **kwargs):
        with torch._dynamo.config.patch(
            accumulated_recompile_limit=max(
                1024, torch._dynamo.config.accumulated_recompile_limit
            )
        ):
            return compiled(*args, **kwargs)

    return run


def report_train(function):
    """Give TorchTitan the same actor timing lane as Dojo's Megatron runs."""
    from functools import wraps

    from modal_dojo.common.timing_recorder import recording_lane_on_reporting_rank

    @wraps(function)
    def wrapped(self, rollout_id, *args, **kwargs):
        with recording_lane_on_reporting_rank(rollout_id):
            return function(self, rollout_id, *args, **kwargs)

    return wrapped


def prepare_bf16(model_name, revision, destination):
    """Publish a complete, revision-specific BF16 copy for both runtimes."""
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    from tempfile import TemporaryDirectory

    from filelock import FileLock
    from huggingface_hub import snapshot_download

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = {"model": model_name, "revision": revision, "format": "bf16-v1"}
    marker = target / "dojo-conversion.json"
    with FileLock(str(target.parent / ".conversion.lock")):
        if marker.is_file() and json.loads(marker.read_text()) == identity:
            index = json.loads((target / "model.safetensors.index.json").read_text())
            if all(
                (target / name).is_file() for name in set(index["weight_map"].values())
            ):
                return
        source = snapshot_download(model_name, revision=revision, local_files_only=True)
        with TemporaryDirectory(dir=target.parent, prefix="converting-") as staging:
            subprocess.run(
                [
                    sys.executable,
                    "/root/miles/tools/convert_inkling2_to_bf16.py",
                    "--model-dir",
                    source,
                    "--save-dir",
                    staging,
                    "--device",
                    "cpu",
                ],
                check=True,
            )
            index = json.loads(
                (Path(staging) / "model.safetensors.index.json").read_text()
            )
            if not index["weight_map"] or not all(
                (Path(staging) / name).is_file()
                for name in set(index["weight_map"].values())
            ):
                raise RuntimeError(
                    "Inkling 2 Mini BF16 conversion did not write every shard"
                )
            (Path(staging) / marker.name).write_text(json.dumps(identity))
            if target.exists():
                shutil.rmtree(target)
            Path(staging).rename(target)


def token_scores(model, hidden, tokens):
    """Retain the native FSDP head and its FP32 projection/softcap in chunks."""
    from modal_dojo.frameworks.miles.inkling2_long_context import stream_token_scores

    cap = model.final_logit_softcapping

    def project(chunk):
        return model.lm_head(chunk[:, 0]).unsqueeze(1)

    def softcap(logits):
        return cap * (logits / cap).tanh() if cap else logits

    scores = stream_token_scores(
        hidden.unsqueeze(1),
        tokens.unsqueeze(0),
        project,
        softcap,
        temperature=model.score_temperature,
        vocab_size=model.unpadded_vocab_size,
    )[0]
    scores._modal_dojo_inkling2_scores = True
    return scores


def softcapped_swiglu(gate, up, gate_cap, up_cap, *, chunk_size=65536):
    """Bound native FP32 expert-activation temporaries, including recomputation.

    Expert dispatch can concentrate millions of routed tokens on one rank.
    Preserve both softcaps and the sigmoid of the uncapped gate in FP32.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    def activate(gate, up):
        gate_f, up_f = gate.float(), up.float()
        capped_gate = (
            gate_cap * torch.tanh(gate_f / gate_cap) if gate_cap > 0 else gate_f
        )
        capped_up = up_cap * torch.tanh(up_f / up_cap) if up_cap > 0 else up_f
        return (capped_gate * torch.sigmoid(gate_f) * capped_up).to(gate.dtype)

    if gate.shape[0] <= chunk_size:
        return activate(gate, up)
    outputs = []
    for start in range(0, gate.shape[0], chunk_size):
        inputs = (gate[start : start + chunk_size], up[start : start + chunk_size])
        outputs.append(
            checkpoint(activate, *inputs, use_reentrant=False)
            if torch.is_grad_enabled()
            else activate(*inputs)
        )
    return torch.cat(outputs)


def message_controller(module, destination, message, *, chunk_size=65536):
    """Bound cross-pass controller temporaries without changing token mixing."""
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    from miles.backends.torchtitan_utils.models.inkling2.layers import _rms_normalize

    def mix(destination, message):
        normalized = _rms_normalize(
            torch.stack((destination.float(), message.float()), dim=1)
        )
        hidden = F.linear(
            normalized.flatten(1).bfloat16(), module.down_weight.bfloat16()
        )
        gain = F.linear(hidden, module.gain_weight.bfloat16()).float()
        forget = F.linear(hidden, module.forget_weight.bfloat16()).float()
        return (1.0 - torch.tanh(forget)) * destination.float() + torch.tanh(
            gain
        ) * message.float()

    if destination.shape[0] <= chunk_size:
        return mix(destination, message)
    outputs = []
    for start in range(0, destination.shape[0], chunk_size):
        inputs = (
            destination[start : start + chunk_size],
            message[start : start + chunk_size],
        )
        outputs.append(
            checkpoint(mix, *inputs, use_reentrant=False)
            if torch.is_grad_enabled()
            else mix(*inputs)
        )
    return torch.cat(outputs)


def feed_forward(module, hidden, docs, *, chunk_size=65536):
    """Bound routed expert buffers while keeping EP collectives in lockstep.

    Every rank must execute the same number of chunks, including when its
    trajectory is shorter. Cache the count for outer checkpoint recomputation.
    Miles' RL config freezes expert bias and disables auxiliary router losses;
    routing and expert computation are then independent for each token.
    """
    import torch
    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint

    key = ("dojo-moe-chunks", chunk_size)
    if key not in docs._masks:
        length = torch.tensor(hidden.shape[0], device=hidden.device)
        if dist.is_initialized():
            dist.all_reduce(length, op=dist.ReduceOp.MAX)
        docs._masks[key] = max(1, (int(length) + chunk_size - 1) // chunk_size)
    chunks = docs._masks[key]
    if chunks == 1:
        return module(hidden)
    if module.router.aux_loss is not None or not module.freeze_expert_bias:
        raise ValueError(
            "Chunked Inkling experts require frozen bias and no auxiliary loss"
        )
    outputs = []
    for chunk in torch.tensor_split(hidden, chunks):
        empty = chunk.shape[0] == 0
        # An empty local slice must still participate in every EP collective,
        # including backward. A masked row keeps its autograd path present.
        mask = None
        if empty:
            chunk = hidden.new_zeros((1, hidden.shape[1]))
            mask = torch.ones(1, device=hidden.device, dtype=torch.bool)
        out = (
            checkpoint(module, chunk, padding_mask_T=mask, use_reentrant=False)
            if torch.is_grad_enabled()
            else module(chunk, padding_mask_T=mask)
        )
        outputs.append(out[:0] if empty else out)
    return torch.cat(outputs)


def block_attn_res(blocks, partial, query, *, partial_addend=None, chunk_size=1024):
    """Keep residual stacks below the CUDA int32-index limit, including backward.

    Miles #4981 uses a contraction-1 bmm whose torch 2.14 native backward
    overflows above 2**31 stacked elements. Token-wise residual mixing is
    independent across chunks, including the loop's cross-pass message bank.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    from miles.backends.torchtitan_utils.models.inkling2.layers import (
        _dojo_original_block_attn_res as original,
    )

    sources = [*blocks, partial] if partial is not None else list(blocks)
    if not sources or chunk_size <= 0:
        raise ValueError("Residual mixing needs sources and a positive chunk size")
    if sources[0].shape[0] <= chunk_size:
        return original(blocks, partial, query, partial_addend=partial_addend)
    count = len(blocks)

    def mix(*values):
        return original(
            list(values[:count]),
            values[count] if partial is not None else None,
            values[-2],
            partial_addend=values[-1],
        )

    outputs = []
    for start in range(0, sources[0].shape[0], chunk_size):
        end = start + chunk_size
        inputs = (
            *(source[start:end] for source in sources),
            query,
            None if partial_addend is None else partial_addend[start:end],
        )
        outputs.append(
            checkpoint(mix, *inputs, use_reentrant=False)
            if torch.is_grad_enabled()
            else mix(*inputs)
        )
    return torch.cat(outputs)


def relative_attention(q, k, v, r, projection, docs, *, window, scale, chunk_size=4096):
    """Chunk queries and relative logits; retain every causal same-document key."""
    import torch
    from torch.nn.attention.flex_attention import AuxRequest, create_block_mask
    from torch.utils.checkpoint import checkpoint

    total = q.shape[0]

    def attend(query, keys, values, relative, offset, block_mask):
        bias = projection(relative).permute(1, 0, 2).contiguous()
        extent, length = bias.shape[-1], query.shape[0]

        def score_mod(score, b, h, q_idx, kv_idx):
            distance = q_idx + offset - kv_idx
            rel = bias[h, q_idx.clamp(max=length - 1), distance.clamp(0, extent - 1)]
            return torch.where(
                (distance >= 0) & (distance < extent),
                score + rel.to(score.dtype),
                score,
            )

        out, aux = _flex_compiled()(
            query.permute(1, 0, 2).unsqueeze(0),
            keys.permute(1, 0, 2).unsqueeze(0),
            values.permute(1, 0, 2).unsqueeze(0),
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=True,
            return_aux=AuxRequest(lse=True),
        )
        return out[0].permute(1, 0, 2).float(), aux.lse[0].permute(1, 0)

    def make_mask(offset, length):
        def mask_mod(b, h, q_idx, kv_idx):
            absolute = q_idx + offset
            keep = (q_idx < length) & (absolute >= kv_idx)
            keep = keep & (
                docs.doc_ids[absolute.clamp(max=total - 1)]
                == docs.doc_ids[kv_idx.clamp(max=total - 1)]
            )
            if window is not None:
                keep = keep & (absolute - kv_idx <= window)
            return keep

        return mask_mod

    outputs, lses = [], []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        key = ("dojo-query-chunk", window, start, end)
        if key not in docs._masks:
            offset = torch.tensor(start, device=q.device)
            block_mask = create_block_mask(
                make_mask(offset, end - start),
                None,
                None,
                end - start,
                total,
                device=q.device,
                # torch 2.14's compiled mask builder recurses in SymPy for this
                # captured-offset/document mask. Eager construction is cached
                # across layer visits and bounded to one query chunk at a time.
                _compile=False,
            )
            # The mask and score closures must share the same offset tensor
            # on every layer visit, including outer checkpoint recomputation.
            # Changing that alias picks a compiled graph with different saved
            # tensors and breaks nested checkpoint backward.
            docs._masks[key] = (offset, block_mask)
        offset, block_mask = docs._masks[key]
        inputs = (q[start:end], k, v, r[start:end], offset, block_mask)
        out, lse = (
            checkpoint(attend, *inputs, use_reentrant=False)
            if torch.is_grad_enabled()
            else attend(*inputs)
        )
        outputs.append(out)
        lses.append(lse)
    return torch.cat(outputs), torch.cat(lses)
