#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SRD_SCRATCH:-/home/jwheojjang/scratch}"
SCRATCH_PROJECT="${SRD_SCRATCH_PROJECT:-${SCRATCH}/sr-diffusion}"
DO_DIV2K=1
DO_FLICKR2K=1
DO_COCO=1
DO_TOY=1
RUN_SMOKE=0
COCO_TARGET_COUNT=6550
COCO_MIN_SIZE=480
COCO_MIN_SIZE_SET=0
COCO_MANIFEST_PATH=""
PHOTO_MANIFEST_PATH=""

usage() {
  cat <<'EOF'
Usage: bash scripts/recover_scratch.sh [options]

Recreate the ephemeral scratch workspace after VM restart.

Options:
  --scratch PATH     Scratch mount point. Default: /home/jwheojjang/scratch
  --skip-div2k      Do not download/extract DIV2K.
  --skip-flickr2k   Do not download/extract Flickr2K or build DF2K manifest.
  --flickr2k        Kept for compatibility; Flickr2K is enabled by default.
  --skip-coco       Do not download/extract COCO or build the photo10k manifest.
  --coco-count N    Number of COCO training images to add. Default: 6550.
  --coco-min-size N Minimum short-side size for COCO images. Default: 480.
  --coco-manifest PATH
                    Output COCO manifest path. Default depends on --coco-count.
  --photo-manifest PATH
                    Output merged photo manifest path. Default depends on --coco-count.
  --skip-toy        Do not recreate the toy dataset.
  --smoke           Run a 1-step scratch tiny training smoke test.
  -h, --help        Show this help.

Environment:
  SRD_SCRATCH          Overrides the scratch mount point.
  SRD_SCRATCH_PROJECT  Overrides the project data root inside scratch.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scratch)
      SCRATCH="$2"
      SCRATCH_PROJECT="${SRD_SCRATCH_PROJECT:-${SCRATCH}/sr-diffusion}"
      shift 2
      ;;
    --skip-div2k)
      DO_DIV2K=0
      shift
      ;;
    --flickr2k)
      DO_FLICKR2K=1
      shift
      ;;
    --skip-flickr2k)
      DO_FLICKR2K=0
      shift
      ;;
    --skip-coco)
      DO_COCO=0
      shift
      ;;
    --coco-count)
      COCO_TARGET_COUNT="$2"
      shift 2
      ;;
    --coco-min-size)
      COCO_MIN_SIZE="$2"
      COCO_MIN_SIZE_SET=1
      shift 2
      ;;
    --coco-manifest)
      COCO_MANIFEST_PATH="$2"
      shift 2
      ;;
    --photo-manifest)
      PHOTO_MANIFEST_PATH="$2"
      shift 2
      ;;
    --skip-toy)
      DO_TOY=0
      shift
      ;;
    --smoke)
      RUN_SMOKE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${ROOT_DIR}"

if [[ "${COCO_TARGET_COUNT}" -eq 100000 && "${COCO_MIN_SIZE_SET}" -eq 0 ]]; then
  COCO_MIN_SIZE=320
fi

if [[ -z "${COCO_MANIFEST_PATH}" ]]; then
  if [[ "${COCO_TARGET_COUNT}" -eq 6550 ]]; then
    COCO_MANIFEST_PATH="${SCRATCH_PROJECT}/data/manifest_coco2017_photo.csv"
  elif [[ "${COCO_TARGET_COUNT}" -eq 100000 ]]; then
    COCO_MANIFEST_PATH="${SCRATCH_PROJECT}/data/manifest_coco2017_photo100k.csv"
  else
    COCO_MANIFEST_PATH="${SCRATCH_PROJECT}/data/manifest_coco2017_photo_coco${COCO_TARGET_COUNT}.csv"
  fi
fi

if [[ -z "${PHOTO_MANIFEST_PATH}" ]]; then
  if [[ "${COCO_TARGET_COUNT}" -eq 6550 ]]; then
    PHOTO_MANIFEST_PATH="${SCRATCH_PROJECT}/data/manifest_photo10k.csv"
  elif [[ "${COCO_TARGET_COUNT}" -eq 100000 ]]; then
    PHOTO_MANIFEST_PATH="${SCRATCH_PROJECT}/data/manifest_photo100k.csv"
  else
    PHOTO_MANIFEST_PATH="${SCRATCH_PROJECT}/data/manifest_photo_coco${COCO_TARGET_COUNT}.csv"
  fi
fi

echo "[1/7] mount/check scratch"
bash scripts/mount_doscratch.sh "${SCRATCH}"

mkdir -p \
  "${SCRATCH_PROJECT}/data" \
  "${SCRATCH_PROJECT}/datasets/photo" \
  "${SCRATCH_PROJECT}/datasets/anime" \
  "${SCRATCH_PROJECT}/checkpoints" \
  "${SCRATCH_PROJECT}/cache" \
  "${SCRATCH_PROJECT}/runs"

touch "${SCRATCH_PROJECT}/.write_test"
rm -f "${SCRATCH_PROJECT}/.write_test"

echo "[2/7] scratch layout"
df -hT "${SCRATCH}"
findmnt "${SCRATCH}" || true

if [[ "${DO_TOY}" -eq 1 ]]; then
  echo "[3/7] recreate toy dataset"
  python scripts/make_toy_dataset.py \
    --output "${SCRATCH_PROJECT}/data/toy" \
    --count 16
else
  echo "[3/7] skip toy dataset"
fi

if [[ "${DO_DIV2K}" -eq 1 ]]; then
  echo "[4/7] recover DIV2K HR dataset"
  python scripts/download_div2k.py \
    --output-dir "${SCRATCH_PROJECT}/datasets/photo/div2k" \
    --manifest "${SCRATCH_PROJECT}/data/manifest_div2k_photo.csv"
  python scripts/dataset_report.py \
    --manifest "${SCRATCH_PROJECT}/data/manifest_div2k_photo.csv" \
    --limit 20
else
  echo "[4/7] skip DIV2K"
fi

if [[ "${DO_FLICKR2K}" -eq 1 ]]; then
  echo "[5/7] recover Flickr2K HR dataset and DF2K manifest"
  python scripts/download_flickr2k.py \
    --output-dir "${SCRATCH_PROJECT}/datasets/photo/flickr2k" \
    --manifest "${SCRATCH_PROJECT}/data/manifest_flickr2k_photo.csv"
  python scripts/merge_manifests.py \
    --inputs \
      "${SCRATCH_PROJECT}/data/manifest_div2k_photo.csv" \
      "${SCRATCH_PROJECT}/data/manifest_flickr2k_photo.csv" \
    --output "${SCRATCH_PROJECT}/data/manifest_df2k_photo.csv"
  python scripts/dataset_report.py \
    --manifest "${SCRATCH_PROJECT}/data/manifest_df2k_photo.csv" \
    --limit 20
else
  echo "[5/7] skip Flickr2K/DF2K"
fi

if [[ "${DO_COCO}" -eq 1 ]]; then
  echo "[6/7] recover COCO train2017 subset and merged photo manifest"
  python scripts/download_coco2017.py \
    --output-dir "${SCRATCH_PROJECT}/datasets/photo/coco2017" \
    --manifest "${COCO_MANIFEST_PATH}" \
    --target-count "${COCO_TARGET_COUNT}" \
    --min-size "${COCO_MIN_SIZE}"
  python scripts/merge_manifests.py \
    --inputs \
      "${SCRATCH_PROJECT}/data/manifest_df2k_photo.csv" \
      "${COCO_MANIFEST_PATH}" \
    --output "${PHOTO_MANIFEST_PATH}"
  python scripts/dataset_report.py \
    --manifest "${PHOTO_MANIFEST_PATH}" \
    --limit 20
else
  echo "[6/7] skip COCO/photo manifest"
fi

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  echo "[7/7] run scratch smoke train"
  python tools/train/train_autoencoder.py \
    --config configs/autoencoder_scratch_tiny.yaml \
    --limit-steps 1
else
  echo "[7/7] skip smoke train"
fi

cat <<EOF

Scratch recovery complete.

Data root:
  ${SCRATCH_PROJECT}

Photo manifests:
  DIV2K: ${SCRATCH_PROJECT}/data/manifest_div2k_photo.csv
  DF2K:  ${SCRATCH_PROJECT}/data/manifest_df2k_photo.csv
  COCO:  ${COCO_MANIFEST_PATH}
  photo: ${PHOTO_MANIFEST_PATH}

Next train command:
  python tools/train/train_autoencoder.py --config configs/autoencoder_photo10k.yaml
EOF
