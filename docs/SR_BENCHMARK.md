# LuSIR x4 정식 SR Benchmark

이 문서는 LuSIR와 외부 baseline을 같은 x4 입력/평가 규칙으로 비교하기 위한
재현 절차를 고정한다. 기존 five-crop 진단과 degradation val100 평가는 계속
유효하지만, 공개 SR 수치와 직접 비교할 때는 이 protocol을 사용한다.

## Protocol

- datasets: DIV2K validation, Set5, Set14, Urban100
- input: 각 dataset의 공개 bicubic x4 LR pair
- target: 대응 HR image
- primary metrics: Y-channel PSNR/SSIM
- color conversion: MATLAB-compatible ITU-R BT.601 Y
- border shave: scale과 같은 4 pixels
- SSIM: 11x11 Gaussian window, sigma 1.5, valid region
- secondary metric: RGB PSNR; RGB SSIM is available with `--include-rgb-ssim`
- full-image tiled inference; center crop을 사용하지 않는다.

DIV2K pair는 공식 ETH Zurich 배포 파일을 사용한다. Set5/Set14/Urban100 x4
pair는 SelfExSR benchmark bundle에서 복구한다. 외부 baseline도 동일 manifest의
LR image를 입력으로 받고 같은 evaluator로 점수를 계산해야 한다.

## Dataset Recovery

```bash
python scripts/download_sr_benchmarks.py
```

기본 manifest:

```text
/home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv
```

예상 image 수:

```text
DIV2K:    100
Set5:       5
Set14:     14
Urban100: 100
total:    219
```

## LuSIR Inference

선택된 detail branch v1d:

```bash
python tools/eval/run_sr_benchmark.py \
  --variant detail_v1d \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/detail_v1d \
  --tile-batch-size 8
```

learned-mask-gated detail branch v2:

```bash
python tools/eval/run_sr_benchmark.py \
  --variant detail_v2_masked \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/detail_v2_masked \
  --tile-batch-size 8
```

V2 config는 learned mask predictor step `3250`, floor `0.05`, branch step
`38000`을 함께 로드한다. Run summary의 `detail_mask_step`과
`detail_mask_floor`로 적용 여부를 확인한다.

conservative Colab option residual refiner v2:

```bash
python tools/eval/run_sr_benchmark.py \
  --variant refiner_v2 \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/refiner_v2 \
  --tile-batch-size 8
```

Stage2/base checkpoint 단독 비교:

```bash
python tools/eval/run_sr_benchmark.py \
  --variant stage2_guarded_detail_v2 \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/stage2_guarded_detail_v2 \
  --tile-batch-size 1

python tools/eval/run_sr_benchmark.py \
  --variant stage2_guarded_detail_v2 \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/stage2_guarded_detail_v2_tta_x8 \
  --tile-batch-size 1 \
  --tta x8

python tools/eval/run_sr_benchmark.py \
  --variant stage2_base \
  --config configs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue.yaml \
  --checkpoint /path/to/stage2_checkpoint.pt \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/stage2_base_candidate \
  --tile-batch-size 8
```

## Metrics

```bash
python tools/eval/eval_sr_benchmark.py \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir outputs/formal_benchmark_lusir \
  --candidate detail_base=/home/ubuntu/scratch/sr-diffusion/benchmark_outputs/detail_v1d/{dataset}/{id}/base.png \
  --candidate stage2_base_candidate=/home/ubuntu/scratch/sr-diffusion/benchmark_outputs/stage2_base_candidate/{dataset}/{id}/base.png \
  --candidate detail_v1d=/home/ubuntu/scratch/sr-diffusion/benchmark_outputs/detail_v1d/{dataset}/{id}/detail.png \
  --candidate refiner_condition=/home/ubuntu/scratch/sr-diffusion/benchmark_outputs/refiner_v2/{dataset}/{id}/condition.png \
  --candidate refiner_v2=/home/ubuntu/scratch/sr-diffusion/benchmark_outputs/refiner_v2/{dataset}/{id}/refined.png
```

Evaluator는 MATLAB-compatible bicubic baseline도 같은 manifest LR에서 생성한다.
출력은 `metrics.csv`, `summary.json`, `contact_sheet.jpg`다.
후보가 많아서 빠른 PSNR sweep만 필요할 때는 `--skip-ssim`을 추가한다. 이 경우
`summary.json`에 `include_ssim: false`가 기록된다.

## Real-ESRGAN Baselines

공식 Real-ESRGAN 구현과 checkpoint를 사용한다. 현재 PyTorch/torchvision과
BasicSR 1.4.x 사이의 legacy import 차이는 runner 내부 compatibility shim으로만
처리하며, 외부 baseline source는 수정하지 않는다.

```bash
pip install -e '.[benchmark-baselines]'

python tools/eval/run_realesrgan_benchmark.py \
  --model-name RealESRNet_x4plus \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/realesrgan

python tools/eval/run_realesrgan_benchmark.py \
  --model-name RealESRGAN_x4plus \
  --manifest /home/ubuntu/scratch/sr-diffusion/benchmarks/x4_benchmark_manifest.csv \
  --output-dir /home/ubuntu/scratch/sr-diffusion/benchmark_outputs/realesrgan
```

`RealESRNet_x4plus`는 distortion/fidelity 비교에, `RealESRGAN_x4plus`는
perceptual/GAN 출력 비교에 사용한다.

## 2026-06-13 Results

아래 수치는 219개 full image 전체를 같은 evaluator로 다시 계산한 결과다.
표는 공개 논문 표와 직접 비교 가능한 dataset별 Y-channel PSNR/SSIM을
기록한다.

| Candidate | DIV2K | Set5 | Set14 | Urban100 |
| --- | ---: | ---: | ---: | ---: |
| Bicubic | 28.1044 / 0.77443 | 28.4318 / 0.81126 | 26.0928 / 0.70491 | 23.1412 / 0.65815 |
| RealESRNet x4plus | 28.8250 / 0.80941 | 29.3828 / 0.86025 | 27.1465 / 0.74917 | 24.4613 / 0.73877 |
| RealESRGAN x4plus | 26.6125 / 0.75766 | 26.6160 / 0.80661 | 25.4216 / 0.69605 | 22.6709 / 0.68553 |
| SwinIR classical x4 | **31.0838 / 0.85228** | not run | not run | not run |
| LuSIR residual refiner v2 | 28.7857 / 0.81931 | 28.1896 / 0.84075 | 27.3704 / 0.76602 | 24.9176 / 0.74489 |
| LuSIR dual-context base | 29.9575 / 0.82887 | 31.6621 / 0.88952 | 28.2441 / 0.77340 | 25.4816 / 0.76473 |
| **LuSIR detail v1d** | **30.1602 / 0.83421** | **31.8892 / 0.89440** | **28.4123 / 0.77998** | **25.8755 / 0.77875** |
| **LuSIR masked detail v2** | **30.1636 / 0.83512** | **31.9495 / 0.89534** | **28.4257 / 0.78102** | **25.8922 / 0.78022** |

V1d는 frozen dual-context base 대비 모든 dataset에서 PSNR과 SSIM을 높였다.
Y PSNR 증가는 DIV2K `+0.2027`, Set5 `+0.2271`, Set14 `+0.1682`,
Urban100 `+0.3939 dB`다. 따라서 branch redesign과 3-epoch 학습은 정식
full-image protocol에서도 유효했다.

Masked v2는 v1d 대비 Y PSNR을 DIV2K `+0.0034`, Set5 `+0.0602`, Set14
`+0.0135`, Urban100 `+0.0167 dB` 개선하고, 전체 평균 Y SSIM을
`+0.00118` 높였다. 네 dataset 모두에서 방향은 일관되지만 전체 평균 Y PSNR
이득은 `+0.0114 dB`에 그친다. 따라서 learned mask는 재현 가능한 작은
correction 개선으로 해석하며, visible texture recovery 돌파나 SOTA 승격
근거로 해석하지 않는다.

공식 SwinIR repository commit `6545850fbf8df298df73d81f3e8cba638787c8bd`의
`001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`를 DIV2K validation에 실행한
결과는 `31.0838 / 0.85228`이다. 같은 evaluator에서 v1d보다 Y PSNR
`+0.9235 dB`, Y SSIM `+0.01807` 높다. 이는 다음 fidelity 병목이 작은 detail
branch의 용량보다 Stage 2/base reconstruction 경로에 있음을 보여준다.

RealESRNet/RealESRGAN은 real-world degradation과 perceptual 출력을 목표로
한 모델이므로 clean-bicubic fidelity 수치가 낮다고 해서 실제 사진 복원
품질이 열등하다는 뜻은 아니다. 반대로 이 결과만으로 LuSIR가 classical SR
SOTA라고 주장할 수도 없다. HAT 등 추가 classical fidelity baseline, LPIPS/
DISTS, 실제 degradation test, blind human review가 별도로 필요하다.

Machine-readable results:

```text
metrics/formal_x4_benchmark_lusir_realesr_summary.json
metrics/formal_x4_benchmark_lusir_realesr_metrics.csv
metrics/formal_x4_benchmark_div2k_swinir_summary.json
metrics/formal_x4_benchmark_div2k_swinir_metrics.csv
metrics/formal_x4_benchmark_detail_v2_masked_summary.json
metrics/formal_x4_benchmark_detail_v2_masked_metrics.csv
```

## 2026-06-15 Guarded Stage2 TTA Sweep

Colab default인 guarded-detail Stage2 v2 step10000에 대해 같은 219개 full image에서
`off`, horizontal flip x2, full x8 self-ensemble을 비교했다. 이 sweep은 빠른
전체 후보 비교를 위해 `--skip-ssim`으로 계산했으므로 Y/RGB PSNR만 기록한다.

| Candidate | Mean Y PSNR | Delta vs bicubic | Mean RGB PSNR | Wins vs bicubic |
| --- | ---: | ---: | ---: | ---: |
| Bicubic | `25.7170` | `+0.0000` | `24.2697` | - |
| RealESRGAN x4plus | `24.7366` | `-0.9804` | `22.9936` | `53/219` |
| LuSIR residual refiner v2 | `26.9154` | `+1.1983` | `25.3777` | `206/219` |
| LuSIR guarded Stage2 off | `27.8539` | `+2.1369` | `26.3263` | `219/219` |
| LuSIR guarded Stage2 hflip x2 | `27.9067` | `+2.1897` | `26.3822` | `219/219` |
| LuSIR guarded Stage2 x8 | `27.9496` | `+2.2325` | `26.4303` | `219/219` |
| LuSIR masked detail v2 | **`28.1429`** | **`+2.4259`** | **`26.6097`** | `219/219` |

Dataset별 Y PSNR:

| Candidate | DIV2K | Set5 | Set14 | Urban100 |
| --- | ---: | ---: | ---: | ---: |
| Guarded off | `29.9483` | `31.6980` | `28.2482` | `25.5122` |
| Guarded hflip x2 | `30.0021` | `31.8478` | `28.2830` | `25.5616` |
| Guarded x8 | `30.0521` | `31.9095` | `28.3244` | `25.5965` |
| Masked detail v2 | **`30.1636`** | **`31.9495`** | **`28.4257`** | **`25.8922`** |

판정:

- TTA는 deterministic Stage2 output을 안정화해 PSNR을 올린다. x8은 off 대비
  평균 Y PSNR `+0.0957 dB`, hflip x2 대비 `+0.0428 dB`다.
- 하지만 contact sheet에서 visible detail 차이는 작고, x8은 219장 평균
  `3.84s/image`로 off의 `0.47s/image`보다 약 8배 느리다. 사용자 기본값은
  여전히 `Off`가 합리적이며, x8은 느린 비교/리뷰 옵션으로 유지한다.
- masked detail v2가 PSNR상으로는 guarded Stage2 x8보다 `+0.1933 dB` 높다.
  다음 연구는 TTA 확대보다 detail branch/generator가 실제 texture를 만들도록
  supervision을 바꾸는 쪽이 우선이다.

Machine-readable results:

```text
metrics/formal_x4_benchmark_stage2_guarded_tta_compare_summary.json
metrics/formal_x4_benchmark_stage2_guarded_tta_compare_metrics.csv
```

SwinIR output은 외부 official repository에서 생성한 뒤 같은 evaluator에
입력했다:

```bash
git clone https://github.com/JingyunLiang/SwinIR
cd SwinIR
python main_test_swinir.py \
  --task classical_sr --scale 4 --training_patch_size 64 \
  --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth \
  --folder_lq /home/ubuntu/scratch/sr-diffusion/benchmarks/div2k/DIV2K_valid_LR_bicubic/X4 \
  --folder_gt /home/ubuntu/scratch/sr-diffusion/benchmarks/div2k/DIV2K_valid_HR
```

## Interpretation

- Y PSNR/SSIM은 공개 classical SR 표와 비교하기 위한 primary fidelity 지표다.
- LPIPS/DISTS와 blind human comparison은 perceptual quality 판단에 별도로
  추가해야 한다.
- Real-ESRGAN 같은 GAN 기반 real-world 모델은 이 clean bicubic protocol에서
  낮은 PSNR을 기록할 수 있다. 따라서 clean fidelity와 실제 사용자 이미지의
  시각적 품질을 같은 결론으로 해석하지 않는다.
- candidate가 HR 크기와 정확히 일치하지 않으면 evaluator는 실패한다. 자동
  resize로 오류를 숨기지 않는다.
