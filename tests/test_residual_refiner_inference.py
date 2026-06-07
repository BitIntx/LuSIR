from __future__ import annotations

from pathlib import Path

from infer_residual_refiner import resolve_checkpoint


def test_resolve_checkpoint_uses_config_relative_path(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "hf"
    checkpoint = tmp_path / "checkpoints" / "refiner.pt"
    config_dir.mkdir(parents=True)
    checkpoint.parent.mkdir()
    checkpoint.touch()

    config = {
        "_config_path": str(config_dir / "refiner.yaml"),
        "inference": {"checkpoint": "../../checkpoints/refiner.pt"},
    }

    assert resolve_checkpoint(config, None) == checkpoint
