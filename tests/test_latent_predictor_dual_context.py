import torch

from sr_diffusion.models import LRToLatentPredictor
from sr_diffusion.utils import load_matching_weights


def test_dual_context_partial_init_preserves_multiscale_output() -> None:
    base_config = {
        "architecture": "multiscale_context",
        "base_channels": 32,
        "num_blocks": 2,
        "norm_groups": 8,
        "context_channels": [40, 48],
        "context_blocks": [1, 1],
    }
    expanded_config = {
        **base_config,
        "architecture": "dual_multiscale_context",
        "extra_context_channels": [48, 64],
        "extra_context_blocks": [2, 2],
    }
    base = LRToLatentPredictor.from_config(base_config).eval()
    expanded = LRToLatentPredictor.from_config(expanded_config).eval()
    load_matching_weights(expanded, base.state_dict())

    lr = torch.randn(2, 3, 32, 32)
    domain_id = torch.tensor([0, 1])
    with torch.no_grad():
        expected = base(lr, domain_id)
        actual = expanded(lr, domain_id)

    torch.testing.assert_close(actual, expected)
