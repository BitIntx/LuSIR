# Command-line tools

- `train/`: model training entry points.
- `eval/`: dataset-level evaluation entry points.
- `infer/`: single-image and tiled inference entry points.
- `analysis/`: experiment comparison, diagnostics, and report generation.

Run tools from the repository root, for example:

```bash
python tools/train/train_latent_pretrain.py --config configs/latent_pretrain_photo100k_multiscale_hqmix_long.yaml
python tools/infer/infer_residual_refiner.py --help
```

Repository-operation utilities such as dataset downloads, manifest generation,
Hugging Face uploads, and W&B organization remain in `scripts/`.
