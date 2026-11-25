"""Rollout engines as a Stitch run (https://github.com/modal-projects/stitch), outside the training cluster."""

import json
import subprocess
import sys

PACKAGE_SPEC = "stitch @ git+https://github.com/modal-projects/stitch@4c17a7e30e785ba73f8662cc473f77eaec3ebee2"
# Weight deltas are compressed and decompressed on the GPUs with nvCOMP.
NVCOMP_INSTALL = "python3 -m pip install --no-deps nvidia-libnvcomp-cu13==6.0.0.12"
_PER_ENGINE_FLAGS = (
    "--model-path",
    "--host",
    "--port",
    "--nccl-port",
    "--nnodes",
    "--node-rank",
    "--dist-init-addr",
    "--base-gpu-id",
    "--gpu-id-step",
    "--gated-launch-port",
    "--random-seed",
)


def extra_config(url: str, generate_url: str, versions: str) -> dict[str, str]:
    return {
        "rollout_external_url": generate_url,
        "rollout_external_control_url": url,
        "update_weight_disk_dir": versions,
        "custom_update_weight_post_write_path": "stitch.trainer.commit_versions",
    }


def engine_launch(
    argv: list[str], env: dict[str, str]
) -> tuple[dict[str, str], dict[str, str], int]:
    """In a child process: building SGLang's arguments reads the GPU, which the trainer keeps."""
    built = subprocess.run(
        ["python3", "-m", __name__, *argv],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    flags, engine_env, gpus = json.loads(built.stdout.splitlines()[-1])
    return flags, engine_env, gpus


def _launch_from_miles(argv: list[str]) -> tuple[dict[str, str], dict[str, str], int]:
    """How Miles launches each rollout engine: its flags, its environment and its GPUs."""
    from miles.backends.sglang_utils.server_args_utils import server_args_to_argv
    from miles.backends.sglang_utils.sglang_engine import (
        _compute_server_args,
        compute_health_check_timeout_seconds,
    )
    from miles.ray.specs.inference import compute_inference_engine_env_vars
    from miles.utils.arguments import parse_args

    sys.argv = ["train_async.py", *argv]
    args = parse_args()
    # An external rollout plans no engines of its own: these are the ones Miles would launch, on
    # rollout_num_gpus_per_engine GPUs with its server args' TP and PP.
    gpus = args.rollout_num_gpus_per_engine
    tp_size, pp_size = (
        args.sglang.base_args["tp_size"],
        args.sglang.base_args["pp_size"],
    )
    server = _compute_server_args(
        args,
        node_rank=0,
        dist_init_addr=None,
        nccl_port=None,
        host="127.0.0.1",
        port=0,
        disaggregation_bootstrap_port=None,
        base_gpu_id=0,
        engine_info_bootstrap_port=None,
        sglang_overrides=None,
        num_gpus_per_engine=gpus,
        tp_size=tp_size,
        pp_size=pp_size,
        gated_launch_port=0,
        random_seed=0,
    )
    flags: dict[str, str] = {}
    for token in server_args_to_argv(server):
        if token.startswith("--"):
            flag, flags[token] = token, ""
        else:
            flags[flag] = f"{flags[flag]} {token}".strip()
    flags = {k: v for k, v in flags.items() if k not in _PER_ENGINE_FLAGS}
    engine_env = compute_inference_engine_env_vars(
        args,
        tp_size=tp_size,
        pp_size=pp_size,
        health_check_timeout_seconds=compute_health_check_timeout_seconds(
            pp_size=pp_size
        ),
    )
    return flags, engine_env, gpus


if __name__ == "__main__":
    print(json.dumps(_launch_from_miles(sys.argv[1:])))
