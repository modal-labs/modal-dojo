import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import modal

TASKS_VOLUME = "coding-agent-tasks"
volume = modal.Volume.from_name(TASKS_VOLUME, create_if_missing=True)
app = modal.App(TASKS_VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("harbor==0.25.0")
)


def in_range(task: Path, low: int, high: int) -> bool:
    statement = (task / "instruction.md").read_text()
    rating = re.search(r"^- \*\*Rating\*\*: (\d+)", statement, re.M)
    inputs = json.loads((task / "tests" / "test_data.json").read_text())["inputs"]
    hidden_tests = any(i.strip() and i.strip() not in statement for i in inputs)
    return bool(rating) and low <= int(rating.group(1)) <= high and hidden_tests


@app.function(image=image, volumes={"/tasks": volume}, timeout=60 * 60)
def write_tasks():
    if Path("/tasks/code-contests").exists():
        return
    subprocess.run(["harbor", "download", "code-contests@1.0", "-o", "/tmp"], check=True)
    tasks = sorted(Path("/tmp/code-contests").iterdir())
    eval_tasks = random.Random(0).sample([t for t in tasks if in_range(t, 1600, 2400)], 128)
    train_tasks = [t for t in tasks if t not in eval_tasks and in_range(t, 1600, 2000)]

    shutil.rmtree("/tasks/tmp", ignore_errors=True)
    for prefix, split in (("train", train_tasks), ("eval", eval_tasks)):
        for task in split:
            dst = f"/tasks/tmp/{prefix}-{task.name}"
            shutil.copytree(task, dst, ignore=shutil.ignore_patterns("solution"))
    Path("/tasks/tmp").rename("/tasks/code-contests")
    volume.commit()
