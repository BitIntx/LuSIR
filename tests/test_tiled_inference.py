import numpy as np
from PIL import Image

from tools.infer.infer_diffusion import edge_pad_image, tile_blend_mask, tile_positions


def test_tile_positions_cover_edges() -> None:
    assert tile_positions(64, 128, 32) == [0]
    assert tile_positions(256, 128, 32) == [0, 96, 128]
    assert tile_positions(300, 128, 32) == [0, 96, 172]


def test_edge_pad_image_repeats_border() -> None:
    image = Image.fromarray(np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8), mode="RGB")
    padded = edge_pad_image(image, 4, 3)
    array = np.asarray(padded)
    assert padded.size == (4, 3)
    assert array[-1, -1].tolist() == [4, 5, 6]


def test_tile_blend_mask_keeps_outer_edges_opaque() -> None:
    mask = tile_blend_mask(16, 4, left_edge=True, right_edge=False, top_edge=True, bottom_edge=False)
    assert mask.shape == (16, 16, 1)
    assert float(mask[0, 0, 0]) == 1.0
    assert float(mask[-1, -1, 0]) == 0.0
    assert float(mask[8, 8, 0]) == 1.0
