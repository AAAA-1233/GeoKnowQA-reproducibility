from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_from_project(path: str | Path, project_dir: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_dir / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def user_text(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    extra = str(example.get("input", "")).strip()
    return instruction if not extra else f"{instruction}\n\n补充信息：\n{extra}"


def messages_without_answer(example: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = str(example.get("system", "")).strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text(example)})
    return messages

