"""Hugging Face backup helpers for reusable checkpoints and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_REPO_NAME = "govind-llm-mdc-indonesian-javanese-asr"


@dataclass(frozen=True)
class BackupItem:
    local_path: Path
    path_in_repo: str


SOURCE_ITEMS = (
    "README.md",
    "LICENSE",
    "Makefile",
    "requirements-local-eval.txt",
    "requirements-local-qwen.txt",
    "configs",
    "docs",
    "scripts",
    "src",
    "submission_src",
    "tests",
    "vendor/README.md",
    "vendor/qwen3_asr_sft.py",
)

SMALL_RESULT_ITEMS = (
    "experiments/results.csv",
    "experiments/plots",
)


def backup_readme() -> bytes:
    text = """# govind-llm MDC Indonesian-Javanese ASR backup

Private backup for competition code, receipts, reusable model checkpoints,
submission zips, and experiment outputs.

This repository intentionally excludes raw MDC competition audio, extracted
restricted datasets, API keys, and local caches. Open-weight checkpoints and
generated submission archives are stored here so training can resume or be
repackaged without rebuilding from scratch.
"""
    return text.encode("utf-8")


def resolve_repo_id(api, repo_id: str, token: str) -> str:
    if "/" in repo_id:
        return repo_id
    identity = api.whoami(token=token)
    namespace = identity.get("name") or identity.get("fullname")
    if not namespace:
        raise ValueError("Could not resolve Hugging Face username from token")
    return "{}/{}".format(namespace, repo_id)


def default_local_items(
    root: Path,
    include_base_models: bool = False,
    include_artifacts: bool = True,
) -> List[BackupItem]:
    items: List[BackupItem] = []
    for relative in SOURCE_ITEMS:
        path = root / relative
        if path.exists():
            items.append(BackupItem(path, "source/{}".format(Path(relative).as_posix())))
    for relative in SMALL_RESULT_ITEMS:
        path = root / relative
        if path.exists():
            items.append(BackupItem(path, Path(relative).as_posix()))
    artifacts = root / "artifacts"
    if include_artifacts and artifacts.exists():
        items.append(BackupItem(artifacts, "artifacts"))
    if include_base_models:
        models = root / "models"
        if models.exists():
            items.append(BackupItem(models, "models"))
    return items


def default_modal_items(
    volume_root: Path,
    checkpoint_root: str,
    include_base_models: bool = False,
) -> List[BackupItem]:
    items: List[BackupItem] = []
    checkpoint_path = volume_root / checkpoint_root
    if checkpoint_path.exists():
        items.append(BackupItem(checkpoint_path, checkpoint_root))
    for relative in ("artifacts", "results"):
        path = volume_root / relative
        if path.exists():
            items.append(BackupItem(path, relative))
    receipt = volume_root / "models" / "qwen3-asr-1.7b" / "MODEL_RECEIPT.json"
    if receipt.exists():
        items.append(BackupItem(receipt, "models/qwen3-asr-1.7b/MODEL_RECEIPT.json"))
    if include_base_models:
        model_dir = volume_root / "models" / "qwen3-asr-1.7b"
        if model_dir.exists():
            items.append(BackupItem(model_dir, "models/qwen3-asr-1.7b"))
    return items


def upload_items(
    *,
    token: str,
    repo_id: str,
    items: Sequence[BackupItem],
    repo_type: str = "model",
    private: bool = True,
    commit_prefix: str = "backup",
    dry_run: bool = False,
) -> Dict[str, object]:
    if not token:
        raise ValueError("HF token is empty")
    from huggingface_hub import HfApi

    api = HfApi()
    resolved_repo = resolve_repo_id(api, repo_id, token)
    plan = [
        {
            "local_path": str(item.local_path),
            "path_in_repo": item.path_in_repo,
            "kind": "dir" if item.local_path.is_dir() else "file",
            "exists": item.local_path.exists(),
        }
        for item in items
    ]
    if dry_run:
        return {"repo_id": resolved_repo, "dry_run": True, "items": plan}

    api.create_repo(
        repo_id=resolved_repo,
        token=token,
        private=private,
        repo_type=repo_type,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=backup_readme(),
        path_in_repo="README.md",
        repo_id=resolved_repo,
        repo_type=repo_type,
        token=token,
        commit_message="{} README".format(commit_prefix),
    )

    uploaded = []
    skipped = []
    for item in items:
        if not item.local_path.exists():
            skipped.append(str(item.local_path))
            continue
        if item.local_path.is_dir():
            commit = api.upload_folder(
                folder_path=item.local_path,
                path_in_repo=item.path_in_repo,
                repo_id=resolved_repo,
                repo_type=repo_type,
                token=token,
                commit_message="{} {}".format(commit_prefix, item.path_in_repo),
            )
        else:
            commit = api.upload_file(
                path_or_fileobj=item.local_path,
                path_in_repo=item.path_in_repo,
                repo_id=resolved_repo,
                repo_type=repo_type,
                token=token,
                commit_message="{} {}".format(commit_prefix, item.path_in_repo),
            )
        uploaded.append(
            {
                "local_path": str(item.local_path),
                "path_in_repo": item.path_in_repo,
                "commit": getattr(commit, "commit_url", None) or str(commit),
            }
        )
    return {"repo_id": resolved_repo, "uploaded": uploaded, "skipped": skipped}


def parse_item_specs(specs: Iterable[str]) -> List[BackupItem]:
    items: List[BackupItem] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--path must use LOCAL=REMOTE format: {}".format(spec))
        local, remote = spec.split("=", 1)
        if not local or not remote:
            raise ValueError("--path must use LOCAL=REMOTE format: {}".format(spec))
        items.append(BackupItem(Path(local), remote.strip("/")))
    return items
