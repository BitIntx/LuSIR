# LuSIR Datasets

The training pipeline uses HR images only. LR inputs are generated on the fly by
the degradation pipeline, so we do not store duplicated LR/HR pairs.

## Current Photo Seed

DIV2K HR is downloaded to:

```text
/home/jwheojjang/scratch/sr-diffusion/datasets/photo/div2k
```

Manifest:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_div2k_photo.csv
```

DIV2K is useful for early super-resolution development, but its page states it
is for academic research only. This project is currently for study/research, so
DIV2K is acceptable as the first development dataset.

The scratch disk can be wiped after VM restart. Recover the current DF2K setup
with:

```bash
bash scripts/recover_scratch.sh
```

## DF2K Expansion

DIV2K alone has 900 images. The current default photo-domain SR set adds
Flickr2K. Flickr2K is commonly used together with DIV2K in SR research and
contains 2,650 2K-resolution HR images. Recover or download it with:

```bash
bash scripts/recover_scratch.sh
```

This creates:

```text
/home/jwheojjang/scratch/sr-diffusion/data/manifest_flickr2k_photo.csv
/home/jwheojjang/scratch/sr-diffusion/data/manifest_df2k_photo.csv
```

The DF2K photo manifest should contain 3,550 rows total: 800 DIV2K train, 100
DIV2K validation, and 2,650 Flickr2K training images.

## 10k Photo Expansion

The current larger photo training manifest adds a deterministic subset of COCO
train2017 to DF2K:

```bash
python scripts/download_coco2017.py \
  --target-count 6550 \
  --min-size 480
python scripts/merge_manifests.py \
  --inputs \
    /home/jwheojjang/scratch/sr-diffusion/data/manifest_df2k_photo.csv \
    /home/jwheojjang/scratch/sr-diffusion/data/manifest_coco2017_photo.csv \
  --output /home/jwheojjang/scratch/sr-diffusion/data/manifest_photo10k.csv
```

Expected count: 10,000 photo training images plus the 100 DIV2K validation
images that remain in the merged manifest.

COCO images have varied original licenses because the images come from Flickr.
Keep this dataset path for study/research experiments and do not redistribute
the downloaded images from this repository.

## 100k Photo Expansion

The completed photo100k scale-up uses a deterministic 100,000-image subset of
COCO train2017, merged with DF2K. This keeps the existing Stage 1 VAE fixed and
gives Stage 2 and Stage 3 broader photo coverage. COCO has only 45,897 images
with short side `>=480`, so the 100k setup uses `--min-size 320`.

```bash
bash scripts/recover_scratch.sh --coco-count 100000
```

Or, if DF2K already exists and only COCO needs to be expanded:

```bash
python scripts/download_coco2017.py \
  --target-count 100000 \
  --min-size 320 \
  --manifest /home/jwheojjang/scratch/sr-diffusion/data/manifest_coco2017_photo100k.csv \
  --keep-archive
python scripts/merge_manifests.py \
  --inputs \
    /home/jwheojjang/scratch/sr-diffusion/data/manifest_df2k_photo.csv \
    /home/jwheojjang/scratch/sr-diffusion/data/manifest_coco2017_photo100k.csv \
  --output /home/jwheojjang/scratch/sr-diffusion/data/manifest_photo100k.csv
python scripts/dataset_report.py \
  --manifest /home/jwheojjang/scratch/sr-diffusion/data/manifest_photo100k.csv \
  --limit 100
```

Expected count is about 103,550 training images plus the 100 DIV2K validation
images preserved by the DF2K manifest.

For a stricter high-resolution-only COCO subset, use `--coco-min-size 480`.
That currently yields about 49k total photo training images after DF2K is
merged.

## 130k Photo + LSDIR Unique-Data Expansion

The latest completed Stage 2 scale-up adds 30,000 unique LSDIR training images
to the photo100k manifest:

```bash
python scripts/download_lsdir_hf.py \
  --output-dir /home/$USER/scratch/sr-diffusion/datasets/photo/lsdir \
  --manifest /home/$USER/scratch/sr-diffusion/data/manifest_lsdir_photo.csv \
  --target-count 30000

python scripts/merge_manifests.py \
  --inputs /home/$USER/scratch/sr-diffusion/data/manifest_photo100k.csv \
           /home/$USER/scratch/sr-diffusion/data/manifest_lsdir_photo.csv \
  --output /home/$USER/scratch/sr-diffusion/data/manifest_photo130k_lsdir.csv
```

Expected merged count: `133450` unique training images and the unchanged
`100` validation images. Raw LSDIR files are not uploaded to GitHub or
Hugging Face.

## Degradation v2

The `photo_v2` preset targets the current denoise/sharpening weakness more
directly than `mild`. It includes stronger pre-downsample blur, optional LR
blur, signal-dependent sensor noise, heavier Gaussian/Poisson noise, lower
quality JPEG/WebP compression, edge ringing, oversharpen halos, color shift,
and stronger banding.

Use it consistently across the condition encoder and diffusion stages. The
completed sequence was:

```text
Stage 2 photo100k fine-tune with degradation_preset: photo_v2
Stage 3/4 photo100k fine-tune with the v2 condition encoder
sampled eval and A/B image review against the mild baseline
```

## Degradation v3 Noise Mix

The `photo_v3_noise_mix` preset is for longer denoise/color-noise training
without writing a separate LR dataset to disk. LR images are still generated
on the fly, which keeps storage small and gives each epoch fresh degradation
samples.

The mix is:

```text
40% photo_v2
40% photo_v3_noise
20% mild
```

`photo_v3_noise` adds explicit chroma/color noise, stronger sensor read/shot
noise, stronger Gaussian noise, heavier JPEG/WebP compression, and banding. It
keeps ringing/oversharpen probabilities moderate because the v2 Stage 3/4
results already showed cyan/green dot artifacts and contrast overshoot on some
samples.

The completed/attempted sequence was:

```text
Stage 2 photo100k long fine-tune with degradation_preset: photo_v3_noise_mix
Stage 3 photo100k v3 fine-tune with the v3 condition encoder
Stage 4 photo100k v3 condition-start fine-tune
bucketed sampled eval for heavy noise, chroma noise, JPEG/WebP, blur+noise
```

## Strict Bicubic Diagnostic

`benchmark_bicubic` is a deterministic evaluation-only preset. It applies PIL
bicubic x4 downsampling and no blur, noise, compression, sharpening, ringing,
color shift, or banding. It exists because the older `clean` preset still
randomly selected bicubic, bilinear, or Lanczos downsampling.

The current five-image DIV2K center-crop result is an exploratory diagnostic,
not a published-benchmark claim. A formal benchmark still requires full-image
evaluation, Y-channel PSNR, border shave, and benchmark-compatible bicubic
generation.

## Scaling Photo Data

The current unique-data scale-up used LSDIR 30k. Further data expansion should
be justified by visible contact-sheet gains, not only small PSNR movement.
Possible future candidates:

- Unsplash Dataset Lite: 25k images, commercial and non-commercial usage stated
  by Unsplash.
- Open Images: very large and more operationally complex; useful after the VAE
  and degradation pipeline are stable.

## Anime / Illustration Data

Do not blindly pull scraped booru datasets for this project unless the intended
use and licensing constraints are clear. Many images are fan art or copyrighted
work. Prefer:

- User-owned or explicitly licensed illustrations.
- Public-domain / CC illustration collections with tracked attribution.
- Small private validation sets separated from training.

Expected layout for user-provided data:

```text
/home/jwheojjang/scratch/sr-diffusion/datasets/photo/...
/home/jwheojjang/scratch/sr-diffusion/datasets/anime/...
```

Build a combined manifest:

```bash
python scripts/build_manifest.py \
  --photo-dir /home/jwheojjang/scratch/sr-diffusion/datasets/photo \
  --anime-dir /home/jwheojjang/scratch/sr-diffusion/datasets/anime \
  --output /home/jwheojjang/scratch/sr-diffusion/data/manifest.csv
```
