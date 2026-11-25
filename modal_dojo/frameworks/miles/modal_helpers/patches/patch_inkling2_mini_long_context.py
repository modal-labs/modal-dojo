"""Wire bounded intermediates into the pinned Miles #4981 TorchTitan model.

TODO: drop these substitutions when upstream streams the native FP32 softcapped
head and chunks attention, residual and expert-activation intermediates. The
residual split also avoids torch 2.14's contraction-1 bmm backward overflow
above 2**31 elements:
https://github.com/radixark/miles/pull/4981#issuecomment-7126851959

Image: radixark/miles:inkling-2-mini@sha256:fd701e76034c497b2c9d14fed51dc54c7f98a75daf4b8ba32a03cfc69990fb17
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_INKLING2_MINI_LONG_CONTEXT"
MODULE = "modal_dojo.frameworks.miles.inkling2_mini_long_context"
ROOT = Path("/root/miles/miles/backends")


def substitutions(target: str) -> list[tuple[str, str]]:
    match target:
        case "perf.py":
            return [
                (
                    '            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]\n',
                    "",
                ),
                (
                    '    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):',
                    '    if log_dict.get("perf/actor_train_time", 0) > 0:\n'
                    '        log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]\n\n'
                    '    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):',
                ),
            ]
        case "native_actor.py":
            return [
                (
                    "logger = logging.getLogger(__name__)",
                    f"from {MODULE} import report_train as _dojo_report_train\n"
                    "from modal_dojo.common.timing_recorder import time_phase as _dojo_phase\n"
                    "from modal_dojo.frameworks.miles.phase_reporting import report_step_event as _dojo_step_event\n\n"
                    "logger = logging.getLogger(__name__)",
                ),
                ("    def train(\n", "    @_dojo_report_train\n    def train(\n"),
                (
                    '        with timer(f"{store_prefix}log_probs"):',
                    '        with timer(f"{store_prefix}log_probs"), _dojo_phase("ref_log_prob" if store_prefix else "actor_log_prob"):',
                ),
                (
                    '        with routing_replay.stage(routing_replay.REPLAY_BACKWARD), timer("actor_train"):',
                    "        if dist.get_rank() == 0:\n"
                    '            _dojo_step_event("optimizer_step", self.args, rollout_id)\n'
                    '        with routing_replay.stage(routing_replay.REPLAY_BACKWARD), timer("actor_train"), _dojo_phase("actor_train"):',
                ),
                (
                    "        compute_advantages_and_returns(self.args, rollout_data, whiten_group=whiten_group)",
                    '        with _dojo_phase("advantages"):\n'
                    "            compute_advantages_and_returns(self.args, rollout_data, whiten_group=whiten_group)",
                ),
            ]
        case "layers.py":
            return [
                ("def block_attn_res(\n", "def _dojo_original_block_attn_res(\n"),
                (
                    "\nclass ResidualQuery(Module):",
                    f"\nfrom {MODULE} import block_attn_res\n\n\nclass ResidualQuery(Module):",
                ),
                (
                    "\nclass Inkling2Attention(Module):",
                    f"\nfrom {MODULE} import relative_attention\n\n\nclass Inkling2Attention(Module):",
                ),
                (
                    "        rel = self.rel_logits_proj(r)\n"
                    "        out, lse = relative_attention(q, k, v, rel, docs, window=self.window, scale=self.head_dim**-0.5)",
                    "        out, lse = relative_attention(q, k, v, r, self.rel_logits_proj, docs, "
                    "window=self.window, scale=self.head_dim**-0.5)",
                ),
                (
                    "        gate_f, up_f = gate.float(), up.float()\n"
                    "        capped_gate = self.gate_cap * torch.tanh(gate_f / self.gate_cap) if self.gate_cap > 0 else gate_f\n"
                    "        capped_up = self.up_cap * torch.tanh(up_f / self.up_cap) if self.up_cap > 0 else up_f\n"
                    "        return (capped_gate * torch.sigmoid(gate_f) * capped_up).to(gate.dtype)",
                    f"        from {MODULE} import softcapped_swiglu\n\n"
                    "        return softcapped_swiglu(gate, up, self.gate_cap, self.up_cap)",
                ),
                (
                    "        normalized = _rms_normalize(torch.stack((destination.float(), message.float()), dim=1))\n"
                    "        hidden = F.linear(normalized.flatten(1).bfloat16(), self.down_weight.bfloat16())\n"
                    "        gain = F.linear(hidden, self.gain_weight.bfloat16()).float()\n"
                    "        forget = F.linear(hidden, self.forget_weight.bfloat16()).float()\n"
                    "        return (1.0 - torch.tanh(forget)) * destination.float() + torch.tanh(gain) * message.float()",
                    f"        from {MODULE} import message_controller\n\n"
                    "        return message_controller(self, destination, message)",
                ),
            ]
        case "model.py":
            return [
                (
                    "        mlp = self.mlp_sconv(self.feed_forward(self.mlp_norm(mlp_input)), docs.positions)",
                    f"        from {MODULE} import feed_forward\n\n"
                    "        mlp = self.mlp_sconv(feed_forward(self.feed_forward, self.mlp_norm(mlp_input), docs), docs.positions)",
                ),
                (
                    "        activation_checkpoint: bool = True",
                    "        activation_checkpoint: bool = True\n        score_temperature: float = 1.0",
                ),
                (
                    "        self.activation_checkpoint = config.activation_checkpoint",
                    "        self.activation_checkpoint = config.activation_checkpoint\n        self.score_temperature = config.score_temperature",
                ),
                (
                    "        logits = self.lm_head(hidden)[..., : self.unpadded_vocab_size]\n"
                    "        cap = self.final_logit_softcapping\n"
                    "        return cap * torch.tanh(logits / cap) if cap else logits",
                    f"        from {MODULE} import token_scores\n\n"
                    "        return token_scores(self, hidden, tokens)",
                ),
            ]
        case "config.py":
            old = "    model = build_model_config(args, parallelism=parallelism, hf_assets_path=hf_assets_path)"
            return [
                (
                    old,
                    old
                    + '\n    if args.backend.titan_model_name == "inkling2":\n        model.score_temperature = args.rollout_temperature',
                )
            ]
        case "loss.py":
            return [
                (
                    "        pred = pred.unsqueeze(0)",
                    "        streamed = getattr(pred, '_modal_dojo_inkling2_scores', False)\n"
                    "        pred = pred.unsqueeze(0)\n"
                    "        if streamed:\n"
                    "            pred._modal_dojo_inkling2_scores = True",
                )
            ]
        case "logit_processors.py":
            old = "    assert non_loss_data\n"
            return [
                (
                    old,
                    old
                    + """    if getattr(logits, "_modal_dojo_inkling2_scores", False):
        from modal_dojo.frameworks.miles.inkling2_long_context import select_token_scores

        return select_token_scores(
            logits, args=args, unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths, response_lengths=response_lengths,
            with_entropy=with_entropy, entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=max_seq_lens, rollout_sampling_mask=rollout_sampling_mask,
        )
""",
                )
            ]
        case _:
            raise ValueError(f"Unknown Inkling 2 Mini patch target: {target}")


def patch_source(source: str, target: str) -> str:
    if MARKER in source:
        return source
    for old, new in substitutions(target):
        if source.count(old) != 1:
            raise ValueError(f"Pinned Inkling 2 Mini source changed: {target}: {old!r}")
        source = source.replace(old, new, 1)
    return f"# {MARKER}\n" + source


def main() -> None:
    titan = ROOT / "torchtitan_utils"
    paths = [
        titan / "models/inkling2/layers.py",
        titan / "models/inkling2/model.py",
        titan / "config.py",
        titan / "loss.py",
        ROOT / "training_utils/loss/hub/logit_processors.py",
        ROOT / "training_utils/metrics/perf.py",
    ]
    patched = [(path, patch_source(path.read_text(), path.name)) for path in paths]
    native_actor = ROOT / "training_utils/torch_native/actor.py"
    patched.append(
        (native_actor, patch_source(native_actor.read_text(), "native_actor.py"))
    )
    for path, source in patched:
        compile(source, str(path), "exec")
    for path, source in patched:
        path.write_text(source)
        print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
