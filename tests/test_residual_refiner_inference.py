from __future__ import annotations

from pathlib import Path

from tools.eval.eval_residual_refiner import resolve_checkpoint as resolve_eval_checkpoint
from tools.infer.infer_residual_refiner import resolve_checkpoint as resolve_inference_checkpoint


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

    assert resolve_inference_checkpoint(config, None) == checkpoint
    assert resolve_eval_checkpoint(config, None) == checkpoint
