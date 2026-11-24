import json
import random
import re
import shutil
import subprocess
from pathlib import Path

DATASET = "code-contests@1.0"
TRAIN_RATINGS = (1600, 2000)  # above 2000 the model rarely solves anything
EVAL_RATINGS = (1600, 2400)
N_EVAL = 128


def rating(task: Path) -> int:
    match = re.search(
        r"^- \*\*Rating\*\*: (\d+)", (task / "instruction.md").read_text(), re.M
    )
    return int(match.group(1)) if match else 0


def has_hidden_tests(task: Path) -> bool:
    """False if the only tests are the examples printed in the statement."""
    tests = json.loads((task / "tests" / "test_data.json").read_text())
    statement = (task / "instruction.md").read_text()
    return any(i.strip() and i.strip() not in statement for i in tests["inputs"])


def in_range(task: Path, ratings: tuple[int, int]) -> bool:
    return ratings[0] <= rating(task) <= ratings[1] and has_hidden_tests(task)


def write_tasks(root: Path, cache: Path = Path.home() / ".cache" / "modal-dojo" / "harbor"):
    src = cache / "code-contests"
    if not src.exists():
        subprocess.run(
            [
                "uvx",
                "--from",
                "harbor==0.25.0",
                "harbor",
                "download",
                DATASET,
                "-o",
                str(cache),
            ],
            check=True,
        )

    tasks = sorted(src.iterdir())
    eval_tasks = random.Random(0).sample(
        [t for t in tasks if in_range(t, EVAL_RATINGS)], N_EVAL
    )
    train_tasks = [
        t for t in tasks if t not in eval_tasks and in_range(t, TRAIN_RATINGS)
    ]

    shutil.rmtree(root, ignore_errors=True)
    for prefix, split in (("train", train_tasks), ("eval", eval_tasks)):
        for task in split:
            dst = root / f"{prefix}-{task.name}"
            shutil.copytree(task, dst, ignore=shutil.ignore_patterns("solution"))
    print(f"{root}: {len(train_tasks)} train, {len(eval_tasks)} eval tasks")
