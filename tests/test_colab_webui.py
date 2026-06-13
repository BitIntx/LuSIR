from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.demo.colab_webui import build_command, make_compare_slider


def test_colab_webui_slider_html_contains_image_payloads(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (8, 8), (30, 40, 50)).save(before)
    Image.new("RGB", (8, 8), (80, 90, 100)).save(after)

    html = make_compare_slider(before, after, "Before", "After")

    assert "data:image/png;base64," in html
    assert "type=\"range\"" in html
    assert "Before" in html
    assert "After" in html


def test_colab_webui_residual_command_uses_strength_and_tiling(tmp_path: Path) -> None:
    cmd, result_file, is_refiner = build_command(
        variant="residual_refiner_v2",
        input_mode="Low-resolution image to upscale",
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "out",
        residual_strength=0.75,
        use_tiling=True,
        tile_overlap=32,
        tile_batch_size=4,
        steps=32,
        seed=123,
    )

    assert is_refiner
    assert result_file == "refined.png"
    assert "tools/infer/infer_residual_refiner.py" in cmd
    assert "--residual-strength" in cmd
    assert "0.750" in cmd
    assert "--tile-batch-size" in cmd
    assert "4" in cmd


def test_colab_webui_diffusion_command_uses_steps(tmp_path: Path) -> None:
    cmd, result_file, is_refiner = build_command(
        variant="photo100k_xl_edge_b16",
        input_mode="Low-resolution image to upscale",
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "out",
        residual_strength=1.0,
        use_tiling=True,
        tile_overlap=32,
        tile_batch_size=1,
        steps=16,
        seed=123,
    )

    assert not is_refiner
    assert result_file == "sr_00.png"
    assert "tools/infer/infer_diffusion.py" in cmd
    assert "--steps" in cmd
    assert "16" in cmd


def test_colab_webui_detail_command_uses_strength_and_tiling(tmp_path: Path) -> None:
    cmd, result_file, uses_strength = build_command(
        variant="detail_branch_v1d",
        input_mode="Low-resolution image to upscale",
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "out",
        residual_strength=0.9,
        use_tiling=True,
        tile_overlap=32,
        tile_batch_size=2,
        steps=32,
        seed=123,
    )

    assert uses_strength
    assert result_file == "detail.png"
    assert "tools/infer/infer_detail_branch.py" in cmd
    assert "--detail-strength" in cmd
    assert "0.900" in cmd
    assert "--tile-batch-size" in cmd
    assert "2" in cmd
