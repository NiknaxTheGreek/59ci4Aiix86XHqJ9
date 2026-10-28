from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root without hard-coding a machine-specific path."""
    current = Path(start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "submission").is_dir() and (candidate / "README.md").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    # Notebook executed from submission/notebooks in an exported bundle.
    for candidate in candidates:
        if (candidate / "results").is_dir() and candidate.name == "submission":
            return candidate.parent
    raise FileNotFoundError(
        "Could not locate the repository root. Run the notebook inside a clone "
        "of this repository or set MONREADER_REPO_ROOT."
    )


def repo_root() -> Path:
    override = os.getenv("MONREADER_REPO_ROOT")
    return Path(override).expanduser().resolve() if override else find_repo_root()


def submission_root() -> Path:
    return repo_root() / "submission"


def results_dir() -> Path:
    return submission_root() / "results"


def figures_dir() -> Path:
    return submission_root() / "assets" / "figures"


def data_root() -> Path | None:
    """Optional local dataset location; never assumed to live in the repo."""
    value = os.getenv("MONREADER_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else None


def load_snapshot(name: str) -> pd.DataFrame:
    path = results_dir() / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool
    project: str
    entity: str | None
    mode: str
    group: str | None
    tags: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "WandbSettings":
        enabled = truthy_env("MONREADER_ENABLE_WANDB", False)
        project = os.getenv("MONREADER_WANDB_PROJECT", "monreader-historical-repro")
        entity = os.getenv("MONREADER_WANDB_ENTITY") or None
        mode = os.getenv("WANDB_MODE", "online" if enabled else "disabled")
        group = os.getenv("MONREADER_WANDB_GROUP") or None
        tags = tuple(
            item.strip()
            for item in os.getenv("MONREADER_WANDB_TAGS", "historical,pre-v2").split(",")
            if item.strip()
        )
        return cls(
            enabled=enabled,
            project=project,
            entity=entity,
            mode=mode,
            group=group,
            tags=tags,
        )


def wandb_login(settings: WandbSettings):
    """
    Log in only when explicitly enabled.

    Credentials are read by wandb from WANDB_API_KEY / local auth. No API key,
    username, entity or sweep ID is stored in the notebooks.
    """
    if not settings.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B is enabled but wandb is not installed. "
            "Install submission/requirements.txt."
        ) from exc
    wandb.login()
    return wandb


def init_wandb_run(
    settings: WandbSettings,
    *,
    config: dict,
    name: str | None = None,
    job_type: str | None = None,
):
    wandb = wandb_login(settings)
    if wandb is None:
        return None
    return wandb.init(
        project=settings.project,
        entity=settings.entity,
        mode=settings.mode,
        group=settings.group,
        tags=list(settings.tags),
        config=config,
        name=name,
        job_type=job_type,
    )


def create_sweep(settings: WandbSettings, sweep_config: dict) -> str | None:
    """Create a sweep only when explicitly enabled; otherwise return None."""
    wandb = wandb_login(settings)
    if wandb is None:
        return None
    return wandb.sweep(
        sweep=sweep_config,
        project=settings.project,
        entity=settings.entity,
    )


def run_agent(
    settings: WandbSettings,
    sweep_id: str,
    train_callable,
    *,
    count: int,
):
    wandb = wandb_login(settings)
    if wandb is None:
        raise RuntimeError("W&B agent requested while MONREADER_ENABLE_WANDB is false.")
    return wandb.agent(
        sweep_id,
        function=train_callable,
        count=count,
        project=settings.project,
        entity=settings.entity,
    )
