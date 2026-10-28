from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Repository / data paths
# ---------------------------------------------------------------------------

def find_repo_root(start: str | Path | None = None) -> Path:
    """Locate the repository without any machine-specific path assumptions."""
    override = os.getenv("MONREADER_REPO_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if (root / "submission").is_dir():
            return root
        raise FileNotFoundError(
            f"MONREADER_REPO_ROOT does not look like the MonReader repository: {root}"
        )

    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "submission" / "results").is_dir():
            return candidate
        if candidate.name == "submission" and (candidate / "results").is_dir():
            return candidate.parent
    raise FileNotFoundError(
        "Could not locate the repository root. Run inside a clone of this "
        "repository or set MONREADER_REPO_ROOT."
    )


def repo_root() -> Path:
    return find_repo_root()


def submission_root() -> Path:
    return repo_root() / "submission"


def results_dir() -> Path:
    return submission_root() / "results"


def figures_dir() -> Path:
    return submission_root() / "assets" / "figures"


def data_root() -> Path | None:
    """Optional local dataset path. Report mode does not require the dataset."""
    value = os.getenv("MONREADER_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else None


# ---------------------------------------------------------------------------
# Local environment loading
# ---------------------------------------------------------------------------

def load_local_env(path: str | Path | None = None) -> bool:
    """
    Load an optional local .env file without making it a repository requirement.

    The repository never needs a .env file to render the committed report-mode
    results. This helper exists only for users who want local W&B/data settings.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    env_path = Path(path).expanduser().resolve() if path else repo_root() / ".env"
    if not env_path.is_file():
        return False

    load_dotenv(env_path, override=False)
    return True


# ---------------------------------------------------------------------------
# Committed result / figure snapshots
# ---------------------------------------------------------------------------

def load_snapshot(name: str) -> pd.DataFrame:
    path = results_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Committed result snapshot not found: {path}"
        )
    return pd.read_csv(path)


def snapshot_path(name: str) -> Path:
    path = results_dir() / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def figure_path(name: str, *, group: str = "final") -> Path:
    path = figures_dir() / group / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Committed figure not found: {path}"
        )
    return path


def display_snapshot(
    name: str,
    *,
    columns: Iterable[str] | None = None,
    digits: int = 6,
) -> pd.DataFrame:
    """
    Load and display a committed CSV snapshot.

    Returns the DataFrame as well so downstream cells can reuse it.
    """
    frame = load_snapshot(name)
    if columns is not None:
        frame = frame.loc[:, list(columns)]

    try:
        from IPython.display import display
        display(frame.round(digits))
    except ImportError:
        print(frame.round(digits).to_string(index=False))

    return frame


def display_figure(
    name: str,
    *,
    group: str = "final",
    width: int | None = None,
):
    """Display one committed figure in Jupyter and return its resolved path."""
    path = figure_path(name, group=group)

    try:
        from IPython.display import Image, display
        display(Image(filename=str(path), width=width))
    except ImportError:
        print(path)

    return path


# ---------------------------------------------------------------------------
# Safe W&B integration
# ---------------------------------------------------------------------------

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
    create_new_sweep: bool
    run_agent: bool
    agent_count: int

    @classmethod
    def from_env(cls) -> "WandbSettings":
        enabled = truthy_env("MONREADER_ENABLE_WANDB", False)
        project = os.getenv(
            "MONREADER_WANDB_PROJECT",
            "monreader-historical-repro",
        ).strip()
        entity = os.getenv("MONREADER_WANDB_ENTITY") or None
        mode = os.getenv(
            "WANDB_MODE",
            "online" if enabled else "disabled",
        ).strip()
        group = os.getenv("MONREADER_WANDB_GROUP") or None
        tags = tuple(
            item.strip()
            for item in os.getenv(
                "MONREADER_WANDB_TAGS",
                "historical,pre-v2",
            ).split(",")
            if item.strip()
        )
        create_new_sweep = truthy_env(
            "MONREADER_CREATE_NEW_SWEEP",
            False,
        )
        run_agent = truthy_env(
            "MONREADER_RUN_WANDB_AGENT",
            False,
        )
        agent_count = int(
            os.getenv("MONREADER_WANDB_AGENT_COUNT", "1")
        )

        if agent_count < 1:
            raise ValueError("MONREADER_WANDB_AGENT_COUNT must be >= 1.")

        return cls(
            enabled=enabled,
            project=project,
            entity=entity,
            mode=mode,
            group=group,
            tags=tags,
            create_new_sweep=create_new_sweep,
            run_agent=run_agent,
            agent_count=agent_count,
        )

    def summary(self) -> dict:
        """Credential-free configuration summary safe to print in notebooks."""
        return {
            "enabled": self.enabled,
            "project": self.project,
            "entity": self.entity,
            "mode": self.mode,
            "group": self.group,
            "tags": list(self.tags),
            "create_new_sweep": self.create_new_sweep,
            "run_agent": self.run_agent,
            "agent_count": self.agent_count,
            "api_key_present": bool(os.getenv("WANDB_API_KEY")),
        }


def _require_wandb(settings: WandbSettings):
    if not settings.enabled:
        raise RuntimeError(
            "W&B is disabled. Set MONREADER_ENABLE_WANDB=1 to opt in."
        )

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B is enabled but wandb is not installed. "
            "Install submission/requirements.txt."
        ) from exc

    return wandb


def wandb_login(settings: WandbSettings):
    """
    Authenticate only when explicitly enabled.

    W&B reads credentials from WANDB_API_KEY or the user's local wandb login.
    No API key, username, entity, sweep ID, or account-specific credential is
    stored in the repository.
    """
    wandb = _require_wandb(settings)
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


def create_sweep(
    settings: WandbSettings,
    sweep_config: dict,
) -> str | None:
    """
    Create a sweep only when BOTH W&B and the explicit sweep gate are enabled.

    Returning None in report mode keeps notebooks safe to Run All.
    """
    if not settings.enabled or not settings.create_new_sweep:
        return None

    wandb = wandb_login(settings)
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
    count: int | None = None,
):
    """
    Run a W&B agent only behind an explicit opt-in gate.

    train_callable is deliberately supplied by the user/notebook so this
    helper remains generic and does not hide model-training logic.
    """
    if not settings.enabled or not settings.run_agent:
        return None

    wandb = wandb_login(settings)
    return wandb.agent(
        sweep_id,
        function=train_callable,
        count=count or settings.agent_count,
        project=settings.project,
        entity=settings.entity,
    )
