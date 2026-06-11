from __future__ import annotations

import argparse
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


@dataclass(frozen=True)
class Artifact:
    local_path: Path
    path_in_repo: str
    sha256: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload selected LuSIR artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Target repository, for example jwheo/sr-diffusion.")
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--message", default="Upload LuSIR artifacts")
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="LOCAL=REMOTE",
        help="File or folder to upload. Example: runs/latest.pt=checkpoints/latest.pt",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=["wandb/**", "*.log", "__pycache__/**"],
        help="Ignore pattern used when uploading folders. Can be repeated.",
    )
    parser.add_argument("--update-card", action="store_true", help="Upload a generated Hugging Face model card.")
    parser.add_argument("--title", default="LuSIR")
    parser.add_argument(
        "--github-url",
        default="https://github.com/BitIntx/sr-diffusion",
        help="Public GitHub project URL to include in the generated model card.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_artifact(value: str) -> tuple[Path, str]:
    if "=" not in value:
        raise ValueError(f"Artifact must use LOCAL=REMOTE form: {value}")
    local, remote = value.split("=", 1)
    local_path = Path(local).expanduser().resolve()
    path_in_repo = remote.strip().lstrip("/")
    if not path_in_repo:
        raise ValueError(f"Remote path is empty for artifact: {value}")
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    return local_path, path_in_repo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_card(title: str, github_url: str, artifacts: list[Artifact]) -> str:
    rows = []
    for artifact in artifacts:
        checksum = artifact.sha256[:12] if artifact.sha256 else "folder"
        rows.append(f"| `{artifact.path_in_repo}` | `{artifact.local_path.name}` | `{checksum}` |")
    artifact_table = "\n".join(rows)
    return f"""---
library_name: pytorch
license: cc-by-nc-4.0
tags:
- super-resolution
- latent-diffusion
- pytorch
- rocm
- research
---

# {title}

Research checkpoint storage for the LuSIR project.

GitHub: {github_url}

LuSIR is a public source-available, non-commercial research project. It trains a
vision-only x4 latent diffusion super-resolution pipeline from scratch and does
not use a pretrained text-to-image diffusion model.

Current artifacts are study/research checkpoints. They are not a production SR
model.

## License

- Checkpoints, generated samples, metrics, and other non-code artifacts:
  CC BY-NC 4.0.
- Source code: PolyForm Noncommercial License 1.0.0.

Commercial use is not permitted without separate written permission. This
includes paid hosted inference, resale, and integration into commercial
products.

Training data is not redistributed in this repository. Dataset license
constraints should be reviewed before training or redistributing derived
weights.

## Artifacts

| Path | Source | SHA256 |
| --- | --- | --- |
{artifact_table}

## Stages

- Stage 1: factor-4 VAE / Autoencoder over 512px HR crops.
- Stage 2: deterministic LR-to-HR-latent pretraining with the Stage 1 VAE frozen.
- Stage 3: conditional latent diffusion U-Net.
"""


def main() -> None:
    args = parse_args()
    api = HfApi()
    parsed = [parse_artifact(value) for value in args.artifact]
    artifacts = [
        Artifact(
            local_path=local_path,
            path_in_repo=path_in_repo,
            sha256=sha256_file(local_path) if local_path.is_file() else None,
        )
        for local_path, path_in_repo in parsed
    ]

    print(f"repo_id={args.repo_id} repo_type={args.repo_type} private={args.private}")
    for artifact in artifacts:
        print(f"artifact {artifact.local_path} -> {artifact.path_in_repo}")
    if args.dry_run:
        return

    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    if args.update_card:
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = Path(tmpdir) / "README.md"
            card_path.write_text(build_model_card(args.title, args.github_url, artifacts), encoding="utf-8")
            api.upload_file(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                commit_message=args.message,
            )

    for artifact in artifacts:
        if artifact.local_path.is_dir():
            api.upload_folder(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                folder_path=artifact.local_path,
                path_in_repo=artifact.path_in_repo,
                ignore_patterns=args.ignore,
                commit_message=args.message,
            )
        else:
            api.upload_file(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                path_or_fileobj=artifact.local_path,
                path_in_repo=artifact.path_in_repo,
                commit_message=args.message,
            )


if __name__ == "__main__":
    main()
