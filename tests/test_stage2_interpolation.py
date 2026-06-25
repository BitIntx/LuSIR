from __future__ import annotations

import pytest
import torch

from tools.analysis.sweep_stage2_interpolation import (
    interpolate_state_dicts,
    validate_compatible_states,
)


def test_interpolate_state_dicts_blends_float_tensors() -> None:
    state_a = {"weight": torch.zeros(2, dtype=torch.float16)}
    state_b = {"weight": torch.ones(2, dtype=torch.float16)}

    mixed = interpolate_state_dicts(state_a, state_b, 0.25)

    assert mixed["weight"].dtype == torch.float16
    assert torch.allclose(mixed["weight"].float(), torch.full((2,), 0.25))


def test_interpolate_state_dicts_takes_nearest_non_float_tensor() -> None:
    state_a = {"counter": torch.tensor([1], dtype=torch.int64)}
    state_b = {"counter": torch.tensor([9], dtype=torch.int64)}

    assert int(interpolate_state_dicts(state_a, state_b, 0.49)["counter"]) == 1
    assert int(interpolate_state_dicts(state_a, state_b, 0.50)["counter"]) == 9


def test_validate_compatible_states_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        validate_compatible_states(
            {"weight": torch.zeros(2)},
            {"weight": torch.zeros(3)},
        )
