# LuSIR VM Handoff / 대화 기반 인수인계

이 문서는 원문 채팅 로그가 아니라, 현재 대화에서 결정하고 실행한 내용을
다른 VM의 Codex/작업자가 바로 이어받을 수 있게 정리한 공개용 요약입니다.

최신 실패/부분 성공/다음 가설 기록은 `docs/TRIAL_AND_ERROR_KO.md`에 누적합니다.

프로젝트 공개명은 **LuSIR**(**Latent Upscaling via Self-trained Image
Restoration**)입니다. GitHub repo id는 `BitIntx/LuSIR`, Hugging Face repo id는
`jwheo/LuSIR`입니다. W&B 기존 run URL, 로컬 scratch 경로, Python import
namespace에는 아직 `sr-diffusion`/`sr_diffusion` 호환 이름이 남아 있습니다.

## 2026-06-22 현재 상태

### 학습 단계와 실제 추론 경로

Stage 번호는 학습 순서를 뜻하며, 추론 때 Stage 1부터 Stage 4까지 모두
직렬로 실행한다는 뜻이 아니다.

```text
deterministic:
  LR -> Stage 2 condition encoder -> optional residual refiner -> Stage 1 VAE decoder -> SR

generative:
  LR -> Stage 2 condition encoder -> Stage 3 또는 Stage 4 diffusion U-Net -> Stage 1 VAE decoder -> SR
```

- Stage 1은 공통 latent 공간과 최종 decoder를 제공한다.
- Stage 2는 LR에서 condition latent를 만든다.
- Stage 3과 Stage 4는 동시에 통과하지 않는다. Stage 4는 Stage 3에서 이어
  학습한 대체 diffusion checkpoint다.
- Stage 5 few-step distillation도 구현되면 Stage 3/4 sampling을 더 짧은
  sampler로 대체하는 단계이지, 뒤에 추가되는 직렬 모듈이 아니다.
- 현재 사용자용 public deterministic 기본값은 T4-friendly guarded-detail Stage2
  v2 step `10000`이다. residual refiner v2는 conservative deterministic 옵션으로
  남겨둔다.
- 기준 deterministic condition 후보는 multiscale Stage 2 step `46000`이고,
  최신 보존 연구 후보는 dual-context LSDIR Stage 2 step `98000`이다.
- 최신 public detail artifact는 v1d step `99500`이다. 3.02M branch를
  정확히 3 epoch 학습했고, v1c에서 identity-preserving 초기화했다.
- 최신 research detail candidate는 learned mask를 적용한 v2 step `38000`이다.
  같은 val100에서 frozen base 대비 PSNR `+0.18177 dB`, mean PSNR
  `+0.20432 dB`, SSIM `+0.00755`, wins `100/100`이다. v1d보다 수치는
  소폭 좋아졌지만 시각적으로 거의 구분되지 않아 public 기본값으로 승격하지 않았다.
- public Colab 기본값은 guarded-detail Stage2 v2 step `10000`이다. detail branch
  v1d, masked v2, residual refiner v2는 단일 이미지/tiled inference 옵션으로
  선택 가능하다.
- 새 NVIDIA GPU VM 재현을 위한 최소 Docker 구성이 추가됐다. `Dockerfile`,
  `.dockerignore`, `scripts/docker_lusir.sh`, `docs/DOCKER_KO.md`를 사용한다.
  dataset/checkpoint/output은 image에 넣지 않고 호스트 scratch를 mount한다.
- learned detail-mask predictor v1은 photo-detail val100에서 observable proxy
  baseline을 통과했다. best step `3250`은 correlation `0.7456`, top20 missing
  capture `0.3861`, excess capture `0.4304`, selection score `0.7013`이다.
- texture-generator용 gate review를 추가로 했다. 같은 v1 predictor에서 selection
  fraction을 비교한 결과 top 5%는 너무 좁고 top 20%는 excess/edge가 많이 열려,
  다음 texture branch의 1차 gate 후보는 `top_fraction: 0.10`,
  `top_mode: binary`, `floor: 0.0`이다. train/infer detail branch 경로에
  선택적 `detail_mask.top_fraction` policy를 추가했으며, 기존 config는 이 값을
  정의하지 않으므로 soft mask 동작이 그대로 유지된다.
- 같은 top10 gate에 synthetic noise patch를 주입해 비교한 결과, GT-supervised
  missing-detail target은 노이즈 patch를 낮게 보지만 learned predictor는 노이즈
  영역을 많이 열었다(`noisy_top_noise_region_mean 0.4531`). 이 반응은
  denoise/correction에는 쓸 수 있어도 texture generator gate로는 위험하다.
- 이 문제를 줄이기 위해 `configs/detail_mask_predictor_v2_noise_negative_probe.yaml`
  를 추가했고, v1 best3250에서 시작해 낮은 target-score patch에 Gaussian noise를
  주입하는 negative augmentation을 학습했다. W&B는
  <https://wandb.ai/jwheo/LuSIR/runs/g0ac6uvt>이다. best step `1500`은 clean top10
  selection score `0.7219`로 v1 `0.7173`보다 약간 높고, excess capture는
  `0.2496 -> 0.2375`로 낮다. 같은 noise response review에서 injected noise
  patch top10 coverage는 `0.4531 -> 0.0000`, predictor mean은 `0.6430 -> 0.0018`
  로 떨어졌다. 다음 texture branch gate는 v1이 아니라 이 v2 noise-negative
  predictor를 우선 사용한다. HF preset은
  `detail_mask_predictor_v2_noise_negative`이다.
- 이 predictor를 frozen soft gate로 쓰는 masked detail branch v2 장기 run은
  완료됐다. step 38000 이후 step 50000까지 best를 갱신하지 못하고 고정 grid도
  거의 같아 조기 중단했다. 위치 선택은 성공했지만 같은 deterministic objective는
  missing texture를 생성하지 못했다.
- 추론/정식 benchmark에서 learned mask가 누락된 재현성 버그를 수정했다.
  새 HF config와 `detail_branch_v2_masked` preset은 predictor step 3250,
  floor `0.05`, branch step 38000을 함께 로드한다.
- masked detail branch v3 patch probe는 완료됐다. best step `1000`은 val100에서
  frozen base 대비 PSNR `+0.18418 dB`, SSIM `+0.00718`, wins `100/100`,
  `lowpass_drift_l1 0.000189`였다. formal 219 benchmark에서는 v2 대비 Y PSNR
  `+0.00667 dB`, RGB PSNR `+0.00470 dB`, Y SSIM `-0.000234`였다.
- v3는 안정적이지만 눈으로 보이는 texture 생성이 거의 없어 public/default로
  승격하지 않는다. 다음 실험은 v3 best generator에서 시작하는
  `configs/detail_branch_v3b_stronger_patch_gan_probe.yaml`이다. masked VGG와
  PatchGAN weight를 올리고 image/residual anchor를 낮춘 visible-detail probe다.
  설계와 중단 기준은 `docs/DETAIL_BRANCH_V3_PATCH_KO.md`다.
- v3b stronger-detail probe는 완료됐지만 장기적으로 실패했다. step 500은
  v3보다 수치상 아주 조금 좋았으나 시각적 차이는 작았고, step 8000은 frozen
  base 대비 PSNR `-0.18243 dB`, SSIM `-0.00113`, wins `11/100`까지 떨어졌다.
- 다음 run은 `configs/detail_branch_v4_teacher_highpass_realesrgan_probe.yaml`다.
  `RealESRGAN_x4plus` teacher cache first `2048` deterministic train samples를
  사용하지만 teacher output 전체를 모방하지 않는다. GT 대비 teacher highpass가
  locally no worse인 위치의 teacher residual/highpass만 약하게 더한다. cache는
  `/home/ubuntu/scratch/sr-diffusion/teacher_cache/realesrgan_x4plus_photo_detail_mix_2048`에 있다.
- v4 teacher-highpass probe도 완료됐다. best checkpoint는 step `0`으로,
  즉 v3 시작점 그대로가 best였다. final step `3000`은 PSNR delta `+0.17735 dB`,
  SSIM delta `+0.00638`, wins `100/100`이지만 step 0의 `+0.18418 dB`,
  `+0.00718`보다 낮다. grid도 시작점과 거의 구분되지 않는다. teacher loss는
  실제 non-zero로 들어갔으나 visible texture generation으로 이어지지 않았다.
- 다음 probe는 `configs/detail_branch_v5_noise_gate_top10_patch_gan_probe.yaml`다.
  v2 noise-negative mask best1500을 `top_fraction 0.10`, `top_mode binary`,
  `floor 0.0`으로 적용하고, selected deterministic v2 branch step38000에서
  다시 시작한다. 목적은 v1 soft mask가 아니라 노이즈에 닫힌 top10 gate에서만
  masked VGG/PatchGAN texture pressure를 주는 것이다. 설계와 중단 기준은
  `docs/DETAIL_BRANCH_V5_NOISE_GATE_KO.md`에 있다.
- v5는 step 3500 eval 직후 중단했다. W&B는
  <https://wandb.ai/jwheo/LuSIR/runs/u9sbs752>이다. step 500까지는
  PSNR delta `+0.0537`, wins `99/100`이었지만, PatchGAN 활성화 이후
  지속적으로 악화되어 step 3250은 `-0.0165 dB`, wins `29/100`, step 3500은
  `-0.0953 dB`, wins `11/100`까지 붕괴했다. `outside_mask_residual_l1`은 끝까지
  `0`이라 v2 top10 gate 자체는 동작했다. 실패 원인은 gate 밖으로 새는 문제가
  아니라, gate 안쪽에서 adversarial pressure가 GT-aligned correction 대신
  artifact 고주파를 키운 것이다. 같은 PatchGAN continuation이나 더 강한 GAN
  weight는 하지 않는다.
- v6 no-GAN detail probe를 추가했다. config는
  `configs/detail_branch_v6_noise_gate_teacher_perceptual_no_gan_probe.yaml`,
  설계 문서는 `docs/DETAIL_BRANCH_V6_NO_GAN_KO.md`다. v2 noise-negative mask
  best1500을 hard top10 gate로 쓰고, selected masked detail v2 step38000에서
  시작한다. PatchGAN은 제거했고, masked VGG + GT-filtered RealESRGAN
  teacher highpass + 새 `artifact_negative_residual_loss`를 쓴다. 4-step smoke는
  정상이며 val100에서 mean PSNR delta `+0.0696 dB`, SSIM delta `+0.00147`,
  detail mask mean `0.1000`, outside mask `0.000000`, lowpass drift `0.000130`을
  확인했다. 이 수치는 초기 안정성 확인일 뿐이고, 실제 판정은 500-6000 step
  W&B sample grid와 eval 추세로 한다.
- v6 장기 probe는 step `6000`까지 완료됐다. W&B run은
  <https://wandb.ai/jwheo/LuSIR/runs/2clmtt44>이다. best checkpoint가 step `0`으로
  남았고, final step6000은 mean PSNR delta `+0.0534 dB`, SSIM delta
  `+0.00103`, wins `94/100`이다. v5 같은 붕괴는 없었지만 시작점보다 나아지지
  못했다. 결론은 “no-GAN teacher/negative loss는 안전하지만 texture 생성력이
  없다”이다. 같은 v6 continuation은 우선하지 않는다.
- 다음 probe는 Stage2 latent residual adapter v1이다. config는
  `configs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1.yaml`,
  설계 문서는 `docs/LATENT_RESIDUAL_ADAPTER_V1_KO.md`다. 기존 dual-context
  Stage2 best98000은 frozen base로 로드하고, zero-init 3.75M adapter만
  학습한다. 4-step smoke는 정상이며 optimizer가 adapter params
  `3,745,296`개만 잡는 것을 확인했다.
- latent residual adapter v1 장기 run은 실행 중이다. W&B run은
  <https://wandb.ai/jwheo/LuSIR/runs/o7tsc4mo>, tmux session은
  `lusir_latent_adapter_v1`, 로컬 로그는
  `/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1/train.log`다.
  로그 확인:
  `tail -f /home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_latent_residual_adapter_v1/train.log`
  초기 step1 eval은 decoded PSNR `24.62`, mean PSNR `26.49`, highpass ratio
  `0.789`, missing `0.01968`로 frozen base 시작점과 일치한다.
- Stage2/base 경로 실험
  `configs/latent_pretrain_photo130k_lsdir_dual_detail_perceptual_v1.yaml`도
  완료됐다. best detail-score는 step `6000`, decoded PSNR `24.5921`,
  detail ratio `0.3824`; latest step `12000`은 decoded PSNR `24.6144`, detail
  ratio `0.3533`이다. 기준 dual-context step98000은 decoded PSNR `24.6197`,
  detail ratio `0.3123`이므로 detail ratio는 올랐지만 눈으로 보이는 차이는
  아직 작고, public/default Stage2로 바로 승격하지 않는다.
- 같은 후보들을 219장 formal x4 benchmark에도 넣었다. 전체 평균 Y PSNR/SSIM은
  dual step98000 `27.8431 / 0.79742`, detail-perceptual step6000
  `27.7737 / 0.79914`, latest12000 `27.8356 / 0.79827`이다. latest12000은
  dual 대비 Y PSNR `-0.0076 dB`, Y SSIM `+0.00085`, Y SSIM wins `156/219`로
  거의 동률이지만 명확한 시각 개선은 작다. 보존 파일:
  `metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_summary.json`,
  `metrics/formal_x4_benchmark_stage2_detail_perceptual_v1_metrics.csv`,
  `samples/stage2_detail_perceptual_v1_benchmark_delta_crop_sheet.jpg`.
- Stage2 `dual_multiscale_attention` v2 probe는 step `8000`까지 봤지만
  decoded PSNR `24.60-24.63`, detail ratio `0.315-0.343` 범위에서 정체했다.
  코드 점검 결과 v2는 이름과 달리 기존 dual-context의 `extra_context` branch가
  꺼진 상태였으므로 attention 자체의 최종 판단으로 쓰지 않는다.
- Stage2 v3 true-dual shifted-window attention `8x8` probe도 step `6000`까지
  확인했다. decoded PSNR은 `24.60-24.63`, detail ratio는 `0.315-0.343`
  범위였고 step `6000`은 decoded PSNR `24.62`, detail ratio `0.329`,
  psnr_detail_score `24.951`이었다. true-dual 구조 정정은 유효했지만
  `8x8` attention은 baseline을 의미 있게 넘지 못해 중단했다.
- Stage2 v3 true-dual shifted-window attention `12x12` window probe도
  step `4000`까지 확인하고 중단했다. config는
  `configs/latent_pretrain_photo130k_lsdir_dual_attention_v3_w12_probe.yaml`,
  W&B는 <https://wandb.ai/jwheo/LuSIR/runs/xg0358xl>이다.
  partial init은 `119.238M / 124.247M = 95.97%`, 새 attention params는
  `5.009M`이었다. 초기 속도는 약 `1.83` micro-step/s, VRAM은 약
  `31.5 / 46.1 GB`였다.
  eval은 decoded PSNR `24.60-24.63`, detail ratio `0.315-0.341` 범위에
  머물렀고, step `4000`은 decoded PSNR `24.62`, detail ratio `0.319`,
  psnr_detail_score `24.941`이었다.
- 결론: v2, v3 `8x8`, v3 `12x12` 모두 baseline 주변에서 정체했다. 더 큰
  attention window는 무거워지기만 하고 missing texture를 복원하지 못했다.
  shifted-window attention/window scaling은 당분간 중단한다.
- 현재 active 학습은 없다. 다음 고신호 방향은 baseline Stage2 latent를 직접
  다시 예측하는 구조가 아니라, frozen/baseline Stage2 출력 위에
  `target_latent - baseline_latent` 또는 decoded highpass/detail residual을
  보정하는 residual/detail correction branch다. 동시에 Stage1 decoder-side
  detail capacity도 병목 후보로 점검한다.

### 최신 완료 detail v1d와 strict-bicubic 진단

- config: `configs/detail_branch_v1d_deep3m_photo130k_lsdir_3ep.yaml`
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/ctg4r7n9>
- 구조: v1c의 8-block/1.35M branch를 18-block/3.02M으로 확장했다. 기존 block은
  복사하고 추가 block은 identity-init하여 시작 출력이 v1c와 정확히 같다.
- 완료 길이: `100086` micro-steps, train `133450`, batch `4` 기준 정확히
  `3 epoch`. `grad_accum_steps: 4`이므로 optimizer update는 micro-step의 1/4이다.
- 선택 checkpoint: step `99500`, `best_eval_detail.pt`, `eval/detail_score` best.
- HF checkpoint:
  `checkpoints/detail_branch_v1d_deep3m_photo130k_lsdir_best99500.pt`
- ordinary `photo_detail_mix` val100:
  - aggregate PSNR delta `+0.1646 dB`
  - mean PSNR delta `+0.1888 dB`
  - SSIM delta `+0.00647`
  - wins `99/100`, detail wins `100/100`
- strict-bicubic DIV2K five-crop:
  - mean RGB PSNR `31.9513 dB`
  - frozen base 대비 `+0.2102 dB`, v1c 대비 `+0.1358 dB`
  - wins `5/5`
- final step `100086`은 strict-bicubic `31.9516 dB`로 사실상 동일하지만,
  ordinary val aggregate PSNR/SSIM/highpass/detail score는 step `99500`이 더 좋다.
- strict-bicubic 진단은 PIL bicubic x4만 적용한 DIV2K val `0801-0805` center
  crop RGB PSNR이다. 정식 SOTA benchmark가 아니다.
- 같은 진단에서 모델별 주요 결과:
  - Stage2 XL condition-only `30.5677`
  - multiscale `31.6068`
  - dual-context `31.7411`
  - dual + v1d `31.9513`
  - 509.658M Stage4 XL sampled `29.5487`
- 판단: Stage2 구조 확장은 실제 clean reconstruction 향상을 만들었다. 반면
  Stage4 XL은 strong-cleanup 역할 때문에 clean input을 과수정한다. v1d 장기
  학습은 안정적 수치 개선을 만들었지만 시각적 detail은 여전히 보수적이다.
  동일 objective continuation이나 단순 capacity 증가는 우선하지 않는다.
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset detail_branch_v1d`

### 정식 full-image x4 benchmark

- dataset: DIV2K validation 100, Set5 5, Set14 14, Urban100 100, 총 219장.
- protocol: 공개 x4 LR pair, MATLAB-compatible BT.601 Y, 4-pixel shave,
  MATLAB-style SSIM, full-image tiled inference.
- detail v1d Y PSNR/SSIM:
  - DIV2K `30.1602 / 0.83421`
  - Set5 `31.8892 / 0.89440`
  - Set14 `28.4123 / 0.77998`
  - Urban100 `25.8755 / 0.77875`
- frozen dual-context base 대비 Y PSNR gain:
  `+0.2027 / +0.2271 / +0.1682 / +0.3939 dB`.
- v1d는 네 dataset 모두에서 base의 PSNR/SSIM을 개선했다. v1d redesign은
  정식 full-image protocol에서도 유효하다.
- masked v2 step38000도 같은 219장 protocol을 완료했다. v1d 대비 Y PSNR은
  DIV2K `+0.0034`, Set5 `+0.0602`, Set14 `+0.0135`, Urban100 `+0.0167 dB`,
  전체 평균 `+0.0114 dB`이며 Y SSIM 전체 평균은 `+0.00118`이다. 네
  dataset 모두 개선했지만 시각적 돌파로 볼 크기는 아니다.
- 같은 clean-bicubic protocol에서 테스트한 RealESRNet/RealESRGAN보다
  fidelity는 높지만, real-world/perceptual 목적 모델과의 결과이므로 SOTA
  주장으로 해석하지 않는다.
- official SwinIR classical x4 DIV2K: `31.0838 / 0.85228`. V1d보다 Y PSNR
  `+0.9235 dB`, Y SSIM `+0.01807` 높다. 다음 clean-fidelity 병목은
  detail branch 용량보다 Stage2/base reconstruction 경로다.
- protocol/results: `docs/SR_BENCHMARK.md`,
  `metrics/formal_x4_benchmark_lusir_realesr_summary.json`.

### Stage2 clean-bicubic fidelity continuation

- 목적: SwinIR 대비 DIV2K `0.9235 dB` gap을 줄이기 위해 Stage2/base
  reconstruction을 clean bicubic 조건에 맞춰 continuation한다.
- config:
  `configs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue.yaml`
- init checkpoint:
  `checkpoints/stage2_photo130k_lsdir_dual_multiscale_best98000.pt`
- degradation: `benchmark_bicubic`
- data augmentation: hflip `0.5`, texture crop retries `4`; HR color jitter는
  사용하지 않는다.
- train: batch `8`, grad accumulation `4`, max `60000` micro steps, lr `5e-6`.
- eval: val100 every `1000` micro steps, no run-at-start eval to avoid spending
  several minutes before actual training begins.
- loss는 PSNR/SSIM fidelity 쪽으로 decoded pixel을 `1.5`로 올리고
  highpass/edge 비중을 낮춘다.
- `train_latent_pretrain.py`의 Stage2 dataset helper가 augmentation 옵션을
  실제 Dataset으로 전달하도록 수정했다.
- `tools/eval/run_sr_benchmark.py --variant stage2_base`를 추가했다. 새 Stage2
  checkpoint가 나오면 formal x4 benchmark에 바로 투입할 수 있다.
- 마지막 clean-fidelity continuation:
  - 종료 step: `17825`; 현재 tmux는 종료됨
  - W&B: <https://wandb.ai/jwheo/LuSIR/runs/xf7xdefw>
  - log:
    `/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_bicubic_fidelity_continue/train_console.log`
  - selected preserved checkpoint before LR probes:
    `.../checkpoints/step_0015000.pt`
  - val100 proxy best: step `15000`, decoded PSNR `25.057`
  - step `17000`: decoded PSNR `25.054`; 사실상 plateau
  - speed: 약 `1.15 micro-step/s`, GPU util `99-100%`, VRAM `37.8/46.1GB`.
- 주의: 위 `25.05`대 값은 task-specific val100 proxy다. 정식 full-image
  Y-channel benchmark의 LuSIR `30.1602` 및 SwinIR `31.0838`과 직접 비교하지
  않는다.
- LR probe 결론:
  - `20x` continuation은 첫 eval에서 `15.72 dB`로 붕괴했다.
  - `5x` continuation은 원본보다 낮았다.
  - `5x` from-init step `4000`은 `25.033`, 원래 LR step `4000`은
    `25.031`로 사실상 동률이다.
  - 따라서 원래 LR `5e-6`로 복귀했다. LR 부족이 핵심 병목은 아니다.
- 종료된 generative-detail v1 probe:
  - 종료 step: 약 `1650`; 현재 tmux는 종료됨
  - config:
    `configs/diffusion_photo130k_lsdir_highfreq_residual_v1_b8.yaml`
  - W&B: <https://wandb.ai/jwheo/LuSIR/runs/q3t4hzms>
  - log:
    `/home/ubuntu/scratch/sr-diffusion/runs/diffusion_photo130k_lsdir_highfreq_residual_v1_b8/train_console.log`
  - step 1 identity baseline: condition PSNR와 prediction PSNR 동일,
    residual L1 `0`, PSNR delta `0.000 dB`
  - 안정 구간 속도 약 `1.10 micro-step/s`, GPU util `99%`, VRAM 약
    `30.3/46.1GB`
  - eval PSNR delta vs condition: step500 `-0.002`, step1000 `-0.017`,
    step1500 `-0.037 dB`
  - fixed sample Laplacian energy는 `0.00592 -> 0.00766`으로 증가했지만
    GT Laplacian L1 오차도 `0.00971 -> 0.01058`로 악화했다.
  - 결론: 실제 detail 복원이 아니라 고주파 에너지만 추가하여 중단했다.
- 구현 완료, 종료된 연구 후보:
  - config: `configs/wavelet_residual_diffusion_v2_probe.yaml`
  - trainer: `tools/train/train_wavelet_residual_diffusion.py`
  - 18.44M U-Net이 `GT - detail v1d`의 signed Haar LH/HL/HH residual에
    직접 diffusion을 수행한다. LL 출력은 구조적으로 없다.
  - val8 clipped oracle: v1d PSNR `28.7012 -> 31.4039`,
    Laplacian L1 `0.018342 -> 0.010721`
  - probe: step3000까지 완료, eval every 250, 3 fixed seeds
  - 첫 full-noise probe는 step1000에서 중단했다. noise loss는 수렴했지만
    sampled residual energy가 target의 약 10배로 남았다.
  - condition-start probe step3000 강도 비교:
    - `t=15`: PSNR `28.1239`, v1d 대비 `-0.5768 dB`, energy ratio `0.708`
    - `t=25`: PSNR `27.4752`, v1d 대비 `-1.2256 dB`, energy ratio `1.116`
    - `t=50`: PSNR `25.0289`, v1d 대비 `-3.6719 dB`, energy ratio `2.503`
  - 시각적으로 `t=15`는 안전하지만 명확한 유효 detail이 없고, `t=25/50`은
    입자 노이즈가 남는다. 아직 public/default 승격 후보가 아니다.
  - 총 optimizer update가 `375`뿐이던 시점에는 학습 부족 가능성이 있어,
    동일 구조 장기 continuation을 step `20000`까지 수행했다.
  - long config:
    `configs/wavelet_residual_diffusion_v2_condition_start_long.yaml`
  - W&B: <https://wandb.ai/jwheo/LuSIR/runs/zh1fktq4>
  - log:
    `/home/ubuntu/scratch/sr-diffusion/runs/wavelet_residual_diffusion_v2_condition_start_long/train_console.log`
  - 완료: step `20000`, `2500` optimizer update.
  - 최종 val100:
    - `t=15`: v1d 대비 PSNR `-0.0880 dB`, SSIM `-0.00647`
    - `t=25`: v1d 대비 PSNR `-0.1392 dB`, SSIM `-0.01040`
    - `t=50`: v1d 대비 PSNR `-0.3152 dB`, SSIM `-0.02433`
  - 장기 학습으로 노이즈는 사라졌지만 residual/diversity도 함께 줄어
    conditional mean/zero residual로 수렴했다. 모든 강도에서 Laplacian/highpass
    오차도 v1d보다 나빠 승격하지 않는다.
  - 다음 generative detail 연구는 동일 continuation이 아니라 learned detail
    mask와 patch-level perceptual/adversarial supervision을 검토한다.
- reproducible GPU throughput comparison commands are documented in
  `docs/GPU_SPEED_BENCHMARK_KO.md`. The quick benchmark now runs the real
  Stage2 `train_latent_pretrain.py` DDP path via `torchrun` and automatically
  uses every visible CUDA GPU. The earlier cu132/cuDNN comparison found no speed
  gain from `cu132` or cuDNN `9.23.1` over the pinned PyTorch cuDNN `9.20.0`.

### 최신 완료 장기 실험

- perceptual continuation의 `+0.01~0.03 dB`는 고정 sample에서 거의 구분되지
  않아 사용자 체감 개선으로 보지 않는다.
- 기존 HQ-balanced manifest는 `203600` rows지만 고유 이미지는
  `103550`장뿐이었다. 반복 노출 대신 LSDIR 고유 이미지 `30000`장을 추가한다.
- 최종 manifest:
  `/home/ubuntu/scratch/sr-diffusion/data/manifest_photo130k_lsdir.csv`
  (`133450` unique train + `100` val).
- config:
  `configs/latent_pretrain_photo130k_lsdir_dual_multiscale_long.yaml`
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/4akqckxu>
- tmux: `stage2-lsdir-dual`
- log:
  `/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo130k_lsdir_dual_multiscale_long/train.log`
- 모델은 selected multiscale step `46000`의 55.50M 파라미터를 모두 불러오고,
  zero-output 초기화된 두 번째 multiscale context branch를 추가한 119.24M
  Stage 2다. 학습 시작 전 출력은 기존 checkpoint와 동일하다.
- batch `8`, grad accumulation `4`, effective batch `32`, max `100000`
  micro steps = `25000` optimizer updates.
- smoke: 초기 val100 decoded PSNR `24.48`, detail ratio `0.291`, VRAM
  `37.8/46.1GB`, GPU util `99%`, 약 `0.75 micro-step/s`.
- 실제 장기 run은 초기 eval 이후 step 50~75에서 약 `1.15 micro-step/s`,
  GPU util `100%`, VRAM `37.8/46.1GB`, 약 `306W`, `58°C`로 안정화됐다.
- 완료: `100000` micro steps = `25000` optimizer updates.
- 자동 best: step `98000`, `best_eval_decoded.pt`.
- final: step `100000`, `latest.pt`.
- 같은 compare tool 기준 selected step46000 대비:
  - `photo_detail_mix`: best `+0.1362 dB`, final `+0.1256 dB`
  - `mild`: best `+0.1086 dB`, final `+0.1025 dB`
  - `photo_v2`: best `+0.0540 dB`, final `+0.0668 dB`
  - `photo_v3_noise_mix`: best `-0.0356 dB`, final `+0.0132 dB`
- 판단: clean/mild에는 의미 있는 소폭 개선이 있고, final은 strong preset에서
  조금 더 안전하다. 하지만 perceptual detail 돌파는 아니므로 human visual
  review 없이 public 경로로 승격하지 않는다.
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset stage2_photo130k_lsdir_dual`
- 일반 milestone은 디스크 보호를 위해 `5000` micro-step마다 저장하고,
  val100 eval과 best checkpoint 갱신은 `1000` micro-step마다 수행한다.
- raw LSDIR 데이터는 GitHub/HF에 올리지 않는다.

### 이전 완료 detail branch v1b

- run: `detail_branch_v1b_aug_photo130k_lsdir`
- config: `configs/detail_branch_v1b_aug_photo130k_lsdir.yaml`
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/1o3aavi9>
- 완료: `40000` micro-steps = `10000` optimizer updates, 약 `1.199 epoch`
  over train `133450`.
- selected: step `39500`, `eval/detail_score` best.
- local checkpoint:
  `/home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1b_aug_photo130k_lsdir/checkpoints/best_eval_detail.pt`
- HF target:
  `checkpoints/detail_branch_v1b_aug_photo130k_lsdir_best39500.pt`
- selected val100:
  - base PSNR `24.6188`, detail PSNR `24.6649`, delta `+0.0461 dB`
  - base SSIM `0.80013`, detail SSIM `0.80281`, delta `+0.00268`
  - mean PSNR delta `+0.0575`, wins `98/100`, detail wins `100/100`
- nearby peaks:
  - PSNR delta best: step `38500`, `+0.0489 dB`
  - SSIM delta best: step `37000`, `+0.00336`
  - final step `40000`: `+0.0444 dB` PSNR, `+0.00277` SSIM, wins `98/100`
- 판단: v1 대비 수치상 진전이 있고 artifact-light지만, 눈으로는 아직 보수적이다.
  라임/털/풀/건물 edge에서 얇은 detail 보강이 보이나 GT 수준의 fine texture에는
  못 미친다.

### 최신 완료 실험

- 완료: Stage 2 multiscale-context + HQ-balanced long run, `50000` micro steps.
- 목표: decoded loss만으로 해결되지 않은 condition smoothing을 넓은 문맥과
  고품질 데이터 노출 비율 교정으로 해결.
- config:
  `configs/latent_pretrain_photo100k_multiscale_hqmix_long.yaml`
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/6zt2do4v>
- 완료된 run log:
  `/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo100k_multiscale_hqmix_long/train.log`
- 초기 val100: decoded PSNR `23.7387`, detail ratio `0.28167`. Zero-init context
  분기 때문에 기존 Stage 2 step 72000 출력과 정확히 같은 시작점이다.
- L40S 한 장: batch `8`, grad accumulation `4`, VRAM 약 `34.8/46.1GB`,
  steady 약 `1.25 micro-step/s`, GPU util `100%`.
- 선택 checkpoint: step `46000`.
- clean/mild에서는 기존 Stage 2보다 약 `+0.92~1.03 dB`, `97~99/100` wins이며
  detail ratio도 소폭 상승했다.
- strong `photo_v2/photo_v3_noise_mix`에서도 약 `+0.94~0.97 dB`지만 detail
  ratio는 약 `0.29~0.30 -> 0.22~0.23`으로 크게 감소했다.
- 결론: base reconstruction/denoising 개선에는 성공했지만 perceptual detail
  복원과 strong-input smoothing 문제는 해결하지 못했다.
- HF preset: `python scripts/download_hf_checkpoints.py --preset stage2_multiscale_hqmix`
- 완료: perceptual Stage 2 continuation, `12000` micro steps:
  `configs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue.yaml`
- selected step 46000에서 frozen ImageNet VGG16 feature loss로 이어 학습한다.
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/nrqhw05u>
- tmux: `stage2-perceptual`
- log:
  `/home/ubuntu/scratch/sr-diffusion/runs/latent_pretrain_photo100k_multiscale_hqmix_perceptual_continue/train.log`
- smoke: batch `4`, grad accumulation `8`, VRAM 약 `20.6/46.1GB`,
  steady 약 `2.62 micro-step/s`, GPU util `99~100%`.
- best checkpoint는 PSNR 단독이 아니라
  `decoded_psnr + 5 * laplacian_energy_ratio`로 선택한다.
- 자동 best: step `8000`, shortlist score `26.0092`.
- step 8000은 초기 step46000 대비 네 preset PSNR이 모두 소폭 상승:
  `photo_detail +0.0101`, `mild +0.0121`, `photo_v2 +0.0136`,
  `photo_v3 +0.0256 dB`.
- step 11000은 clean/mild/photo_v2에서 약 `+0.024~0.025 dB`로 가장 높지만
  `photo_v3_noise_mix`에서 `-0.0063 dB` 후퇴했다.
- 시각적으로 초기/step8000/step11000/step12000 차이는 거의 보이지 않고,
  missing fine detail과 smoothing은 해결되지 않았다.
- 결론: step8000을 안전한 실험 후보로 보존하되 public/default Stage2로는
  승격하지 않는다.
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset stage2_multiscale_perceptual`

- residual refiner v2 lower-LR continuation과 cross-preset 평가 완료.
- 완료: `40000` micro steps, best decoded PSNR step `39000`.
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/3v6wmf5o>
- 병목 없음: L40S util `99-100%`, VRAM 약 `41.8/46.1GB`, steady `0.87~0.91 step/s`.

| degradation | condition mean PSNR | refined mean PSNR | condition 대비 | wins |
| --- | ---: | ---: | ---: | ---: |
| `photo_detail_mix` | 25.3103 | 25.6410 | +0.3307 | 94/100 |
| `mild` | 25.0449 | 25.3161 | +0.2712 | 91/100 |
| `photo_v2` | 22.9271 | 23.0419 | +0.1148 | 81/100 |
| `photo_v3_noise_mix` | 22.9014 | 23.0787 | +0.1773 | 81/100 |

- step 11000 대비 `photo_detail_mix` 평균 이득은 `+0.1318 -> +0.3307 dB`,
  `mild`는 `+0.1178 -> +0.2712 dB`, `photo_v3_noise_mix`는
  `+0.1160 -> +0.1773 dB`로 상승했다.
- strong preset도 평균은 개선됐지만 `photo_v2`/`photo_v3_noise_mix` 승률은 각각
  `84 -> 81`, `88 -> 81`로 낮아져 더 공격적인 보정의 tail risk가 있다.
- 추론 guardrail:
  - full `1.0`: 평균 품질 최고.
  - balanced `0.75`: 평균 이득 대부분 유지, strong preset 승률 `83/100`.
  - conservative `0.5`: strong preset 승률 `86/100`, 평균 이득은 감소.
  - CLI `--residual-strength`, Colab WebUI `Correction strength`
    slider로 선택 가능.
- 공식 선택 checkpoint:
  `checkpoints/residual_refiner_stage2_xl_photo_detail_v2_best39000.pt`
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset residual_refiner_v2`

- detail-preserving degradation curriculum과 Stage4 장기 적응 학습 완료.
- 새 preset:
  - `photo_detail`: object/detail 신호를 보존하는 약한 blur/noise/compression 조합.
  - `photo_detail_mix`: clean `35%`, photo_detail `48%`, mild `15%`, photo_v2 `2%`.
- 기존 `photo_v3_noise_mix`는 clean 비중이 없고 강한 v2/v3가 `80%`라 denoise/cleanup
  편향이 과도했다.
- val100 degradation audit:
  - `photo_v3_noise_mix`: bicubic `22.3599`, chroma RMS `0.02040`
  - `photo_detail_mix`: bicubic `24.7357`, chroma RMS `0.00507`
- 기존 Stage2 XL은 새 mix에서도 condition-only `25.3103`, bicubic 대비 `+0.5745`라
  Stage2 재학습은 보류했다.
- Stage4 long run:
  - config:
    `configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_detail_b8_long.yaml`
  - W&B: <https://wandb.ai/jwheo/LuSIR/runs/so0lbyte>
  - 완료: `12000` micro steps = `3000` optimizer updates
  - 병목 없음: L40S util `99-100%`, VRAM 약 `45.0/46.1GB`, steady `0.856 step/s`

| 모델/checkpoint | SR PSNR | bicubic 대비 | condition 대비 | condition wins |
| --- | ---: | ---: | ---: | ---: |
| Stage2 condition-only | 25.3103 | +0.5745 | n/a | n/a |
| teacher Stage4 init | 25.3187 | +0.5829 | +0.0084 | 46/100 |
| photo-detail Stage4 best step 8000 | 25.3406 | +0.6049 | +0.0303 | 71/100 |
| photo-detail Stage4 latest step 12000 | 25.3337 | +0.5980 | +0.0235 | 67/100 |
| 기존 edge Stage4 step 4250 | 25.1176 | +0.3818 | -0.1927 | 13/100 |

- 공식 선택 checkpoint는 step `8000`의 `best_eval_condition_decoded.pt`.
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset stage4_photo_detail`
- 시각적으로 condition의 구조/선명도를 보존하면서 작은 보정을 더하고, 기존 edge
  Stage4처럼 전체를 과하게 덮어쓰지 않는다.
- 이번 결과는 Stage4 gated-residual이 처음으로 condition-only를 평균 PSNR과 승률
  모두에서 명확히 넘은 결과다.
- 한계:
  - absolute-Laplacian energy는 GT의 약 `29.7%`라 실제 fine-detail 생성은 여전히 약하다.
  - `photo_detail_mix`의 2% strong tail에서 동상 같은 밝은 점 artifact가 남는다.
  - step 12000은 step 8000보다 소폭 후퇴했으므로 더 긴 동일 continuation은 우선순위가 아니다.

다음 우선순위:

1. best step 8000을 별도 detail-focused/실사용 이미지 세트에서 평가.
2. strong tail `photo_v2` 2%를 별도 robustness curriculum 또는 별도 평가 slice로 분리 검토.
3. PSNR뿐 아니라 LPIPS/DISTS 계열 perceptual metric과 detail metric을 평가에 추가.
4. 같은 Stage4 continuation보다 teacher/refiner가 더 큰 실제 high-frequency residual을
   안전하게 전달하도록 개선.

- deterministic residual refiner teacher-supervision Stage4 probe를 `8000` micro steps까지 완료.
- config:
  `configs/diffusion_photo100k_xl_stage4_condition_v3_teacher_residual_photo_v3_b8_probe.yaml`
- W&B:
  - <https://wandb.ai/jwheo/LuSIR/runs/6h0124us>
  - <https://wandb.ai/jwheo/LuSIR/runs/0p3lfqt7>
- `photo_v3_noise_mix` sampled val100, condition init, 32 steps:

| checkpoint | start timestep | SR PSNR | bicubic 대비 | condition 대비 | condition wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher step 2000 | 25 | 22.9640 | +0.6041 | +0.0626 | 68/100 |
| teacher step 2000 | 50 | 22.9639 | +0.6040 | +0.0625 | n/a |
| teacher step 4000 | 25 | 22.9571 | +0.5972 | +0.0557 | 65/100 |
| teacher step 8000 | 25 | 22.9490 | +0.5891 | +0.0476 | 59/100 |

- step `2000`이 sampled 기준 최고이며, 이후 장기 학습은 개선되지 않았다.
- 시각적으로는 털/잎/건물/나뭇가지 detail을 복원하지 못하고 smoothing이 강하다.
  teacher step 2000의 absolute-Laplacian energy는 GT의 `21.8%`로, 기존 edge t25의
  `32.7%`보다 낮다.
- 결론: teacher supervision은 작은 PSNR cleanup 이득은 만들었지만 사용자 체감
  업스케일 detail 목표에는 실패했다.
- 다음 우선순위는 같은 Stage4 continuation이 아니라 `photo_v3_noise_mix`의 과도한
  노이즈 강도/비중을 줄이고 clean/mild 중심 detail 복원 curriculum을 재설계하는 것이다.
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset stage4_teacher_residual_probe`

- residual refiner standalone eval/inference 도구 추가:
  - `tools/eval/eval_residual_refiner.py`
  - `tools/infer/infer_residual_refiner.py`
- 같은 frozen sparse-gate refiner checkpoint step `500`으로 cross-degradation val100 평가 완료.

| degradation | bicubic PSNR | condition PSNR | refined PSNR | refined-condition | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mild` | 24.4778 | 25.0449 | 25.1178 | +0.0729 | 86/100 |
| `photo_v2` | 22.4103 | 22.9271 | 22.9767 | +0.0496 | 77/100 |
| `photo_v3_noise_mix` | 22.3599 | 22.9014 | 22.9600 | +0.0586 | 86/100 |

- 시각 판단:
  - refined는 condition과 매우 가깝고, 과한 fake texture나 색 손상은 보이지 않는다.
  - 대신 눈에 보이는 detail 회복도 작다.
  - 단일 DIV2K val 샘플에서 Stage4 XL edge는 더 많이 건드려 cleanup이 강하지만,
    Stage4/refiner 모두 GT fine texture를 안정적으로 복원하지는 못했다.
- 결론:
  - residual refiner는 v2/v3에서도 condition-only를 안정적으로 이기는 안전한 보정기다.
  - final SR 모델이라기보다 Stage4 residual teacher/warm-start로 쓰는 쪽이 타당하다.
  - 다음 우선순위는 refiner capacity/loss 확장 또는 Stage4 residual/gate supervision이다.
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset residual_refiner_stage2_xl_mild`

## 2026-06-05 추가 최신 실험 요약

- Stage2 residual/oracle diagnostic 완료.
- Stage2 condition은 구조/색/저주파를 이미 잘 맞추고, 남은 차이는 대부분 고주파 detail이다.
- `mild` val100 diagnostic:
  - bicubic PSNR: `24.4778`
  - condition decoded PSNR: `25.0543`
  - oracle full residual PSNR: `41.8207`, condition 대비 `+16.7664`
  - oracle highpass PSNR: `35.0872`, condition 대비 `+10.0329`
  - oracle lowpass PSNR: `25.0814`, condition 대비 `+0.0270`
  - residual highpass energy ratio: `0.8988`
- deterministic bounded residual refiner probe 완료.
- sparse-gate best step `500`:
  - condition mean PSNR: `25.0449`
  - refined mean PSNR: `25.1178`
  - condition 대비 `+0.0729`
  - condition 이긴 샘플 `86/100`
  - gate mean `0.2147`
- open-gate ablation step `500`:
  - refined mean PSNR: `25.0972`
  - condition 대비 `+0.0523`
  - condition 이긴 샘플 `73/100`
  - gate mean `0.8680`
- 결론: residual detail은 학습 가능하지만, gate를 무작정 열면 좋아지는 문제가 아니다.
  다음 Stage4는 긴 continuation보다 deterministic residual refiner를 teacher/warm-start로
  쓰거나 residual/gate supervision을 직접 넣는 방향이 우선이다.
- HF preset:
  `python scripts/download_hf_checkpoints.py --preset residual_refiner_stage2_xl_mild`

## 2026-06-05 최신 실험 요약

- `diffusion_photo100k_xl_stage4_condition_v3_rolesplit_mild_b8_probe` 학습 완료.
- W&B: <https://wandb.ai/jwheo/LuSIR/runs/lrb6nco9>
- 완료 step: `8000` micro steps = `2000` optimizer updates (`grad_accum_steps=4`)
- best checkpoint: step `7500`, `best_eval_condition_decoded.pt`
- `mild` val100 sampled eval 결과:
  - Stage2 condition-only: `25.0449` PSNR, bicubic `24.4778`, delta `+0.5672`
  - Stage4 role-split t25: `24.5747` PSNR, condition 대비 `-0.4702`, wins `3/100`
  - Stage4 role-split t10: `24.9185` PSNR, condition 대비 `-0.1264`, wins `3/100`
  - Stage4 role-split t5: `24.9935` PSNR, condition 대비 `-0.0514`, wins `6/100`
  - Stage4 role-split t1: `25.0335` PSNR, condition 대비 `-0.0114`, wins `10/100`
- 결론: role-split loss는 condition 손상을 줄였지만, Stage2 condition-only를 넘는
  유용한 SR detail을 안정적으로 추가하지 못했다. 다음은 loss weight 튜닝보다
  bounded/gated residual 방식의 Stage4 parameterization 변경을 검토한다.

다음으로 `gated_residual_x0` parameterization을 구현했고, 새 probe config를 추가했다:

```text
configs/diffusion_photo100k_xl_stage4_condition_v3_gated_residual_mild_b8_probe.yaml
```

핵심은 U-Net output을 noise가 아니라 `condition + bounded residual * learned gate`
로 해석하고, sampler noise는 해당 x0에서 역산하는 것이다. CUDA smoke는 batch 8
forward/backward까지 통과했다. 자세한 가설과 평가 기준은
`docs/TRIAL_AND_ERROR_KO.md`의 "실험 3" 섹션을 본다.

실험 3 중간 결과:

- W&B: <https://wandb.ai/jwheo/LuSIR/runs/edfko8e8>
- step `2000`에서 중단하고 sampled eval 완료.
- one-step decoded PSNR은 step `500` 이후 `23.47-23.48`로 보합.
- best one-step checkpoint: step `1000`
- sampled 기준 최고는 step `2000`, t25:
  - SR `25.0445`
  - bicubic `24.4778`
  - condition-only `25.0449`
  - condition 대비 `-0.0004 dB`
  - condition 이긴 샘플 `34/100`
- 결론: gated residual은 role-split보다 condition 보존을 크게 개선했지만,
  평균으로 condition-only를 넘지는 못했다. 다음은 더 오래 학습보다 residual/gate에
  필요한 위치와 크기를 더 직접적으로 지도하는 방향을 검토한다.

## 목표

LuSIR는 직접 학습하는 x4 vision-only latent diffusion super-resolution 모델이다.

- T2I pretrained diffusion 모델을 사용하지 않음.
- `LR 128x128 -> HR 512x512`가 현재 기본 목표.
- `LR 192x192 -> HR 768x768`은 이후 목표.
- photo / anime domain conditioning을 한 코드베이스에서 처리.
- PyTorch first.
- GPU/ROCm 우선, TPU/XLA는 나중에 고려.
- custom CUDA/ROCm op 없이 표준 PyTorch 위주로 유지.

## 모델 구조

```text
HR image
  -> factor-4 VAE / AutoencoderKL
  -> HR latent

LR image
  -> LR-to-latent condition encoder
  -> condition latent

noisy HR latent + condition latent + timestep + domain embedding
  -> conditional diffusion U-Net
  -> denoised HR latent
  -> VAE decoder
  -> x4 SR output
```

현재 파라미터 수:

```text
Stage 1 VAE:                  21.096M
Stage 2 LR-to-latent encoder:  2.388M
Stage 3 diffusion U-Net:      76.610M
Full inference path:         100.094M
```

500M급 확장 config:

```text
XL Stage 1 VAE:                  21.096M
XL Stage 2 LR-to-latent encoder: 18.944M
XL Stage 4 diffusion U-Net:     469.618M
XL full inference path:         509.658M
Stage 2 XL config: configs/latent_pretrain_photo100k_v3_noise_xl.yaml
Stage 4 XL config: configs/diffusion_photo100k_xl_stage4_condition_v3.yaml
```

## 라이선스 / 공개 상태

- GitHub: <https://github.com/BitIntx/LuSIR>
- Hugging Face: `jwheo/LuSIR`
- GitHub repo는 public.
- HF model repo도 public.
- code license: PolyForm Noncommercial 1.0.0.
- checkpoint/artifact license: CC BY-NC 4.0.
- 상업적 이용은 금지.
- raw training data는 repo/HF에 올리지 않음.

## 현재 구현된 것

- 프로젝트 scaffold.
- config system.
- manifest 기반 dataset loader.
- x4 degradation pipeline.
- Stage 1 VAE training/eval/inference.
- Stage 2 deterministic LR-to-HR-latent pretrain.
- Stage 3 conditional latent diffusion training.
- Stage 4 condition-start fine-tune prototype.
- W&B online logging.
- fixed sample image logging: LR / GT / Pred.
- sampled validation/eval tooling.
- HF artifact upload/download scripts.
- Colab demo: Gradio WebUI 기반. 유저 업로드가 기본이고, model, TTA inference,
  correction strength, tile overlap, tile batch size, diffusion steps를 조정한다.
  기본 model은 guarded-detail Stage2 v2 step `10000`, tile batch size 기본값은
  `1`이다. 출력은 bicubic/Stage 2 condition/Input LR nearest 중 하나와 SR
  output을 before/after slider로 비교한다.
- tiled inference:
  - `--tile`
  - 128x128 LR tiles
  - overlap feather blending
  - arbitrary-size LR image to x4 output

## 데이터 상태

Scratch root:

```text
/home/jwheojjang/scratch/sr-diffusion
```

현재 주요 manifests:

```text
data/manifest_photo10k.csv
data/manifest_photo100k.csv
```

`manifest_photo100k.csv`:

```text
photo/train: 103450
photo/val:   100
```

구성:

- DIV2K
- Flickr2K
- deterministic COCO train2017 subset

COCO train2017은 `min_size>=480` 조건으로는 45,897장밖에 안 나와서,
photo100k는 `min_size>=320` 기준으로 구성했다.

Scratch는 VM restart나 VM 이동 시 날아갈 수 있으므로 데이터는 복구
스크립트로 다시 받는 전제로 운영한다.

## 완료된 학습

### Stage 1: VAE / Autoencoder

```text
config: configs/autoencoder_photo10k.yaml
run: autoencoder_photo10k_b16_eval_online
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/autoencoder_photo10k_b16_eval_online/checkpoints/best_eval_recon.pt
finished step: 50000
eval/recon: 0.01198
eval/kl:    9.38684
eval/psnr:  40.19
```

HF:

```text
checkpoints/stage1_autoencoder_best_eval_recon.pt
```

### Stage 2: 10k LR-to-latent

```text
config: configs/latent_pretrain_photo10k.yaml
run: latent_pretrain_photo10k_b16
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo10k_b16/checkpoints/best_eval_latent.pt
finished step: 50000
best eval/latent_loss: step 48000, 0.21775
best decoded PSNR proxy: step 47000, 23.89
```

HF:

```text
checkpoints/stage2_latent_pretrain_best_eval_latent.pt
```

### Stage 3: 10k diffusion

```text
config: configs/diffusion_photo10k_b32.yaml
run: diffusion_photo10k_b32
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
finished step: 25000
best eval/noise_mse checkpoint: step 24000
sampled val100 t50 32-step:
  SR PSNR:      25.2216
  bicubic PSNR: 24.4778
  delta:        +0.7438
```

HF:

```text
checkpoints/stage3_diffusion_b32_best_eval_noise.pt
```

### Stage 4: 10k condition-start prototype

```text
config: configs/diffusion_photo10k_b32_stage4_condition.yaml
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt
best checkpoint step: 1000
val100 t25 32-step sampled SR PSNR: 25.2930
```

HF:

```text
checkpoints/stage4_condition_b32_best_eval_condition_decoded.pt
metrics/stage4_condition_val100_t25_32step_summary.json
```

### Stage 2: photo100k scale-up

```text
config: configs/latent_pretrain_photo100k.yaml
run: latent_pretrain_photo100k_b64
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 30000
best eval/latent_loss: step 28000, 0.21230
best decoded PSNR proxy: step 22000, 23.93
final eval: step 30000, eval/latent_loss 0.21267, decoded_psnr 23.88
```

HF:

```text
checkpoints/stage2_photo100k_b64_best_eval_latent.pt
metrics/stage2_photo100k_b64_summary.json
```

### Stage 3: photo100k scale-up

```text
config: configs/diffusion_photo100k_b32.yaml
run: diffusion_photo100k_b32
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo10k_b32/checkpoints/best_eval_noise.pt
condition encoder:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 60000
best eval/noise_mse: step 60000, 0.00680
best decoded PSNR proxy: step 53000, 24.57
final eval: step 60000, decoded_psnr 24.56, eval/x0_mse 0.08052
```

HF:

```text
checkpoints/stage3_photo100k_b32_best_eval_noise.pt
metrics/stage3_photo100k_b32_summary.json
```

Stage3 sampled eval:

```text
output:
  /home/jwheojjang/scratch/sr-diffusion/runs/eval_diffusion_photo100k_val100_t50_32step
val100, 32 DDIM steps:
  SR PSNR:      25.3745
  bicubic PSNR: 24.4778
  delta:        +0.8967
```

### Stage 4: photo100k condition-start

```text
config: configs/diffusion_photo100k_b32_stage4_condition.yaml
run: diffusion_photo100k_b32_stage4_condition
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition/checkpoints/best_eval_condition_decoded.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt
finished step: 5000
best decoded checkpoint: step 1500
sampled val100 t25 32-step:
  SR PSNR:      25.4072
  bicubic PSNR: 24.4778
  delta:        +0.9294
  vs Stage3:    +0.0327, wins 68 / losses 32
```

HF:

```text
checkpoints/stage4_photo100k_condition_b32_best_eval_condition_decoded.pt
metrics/stage4_photo100k_condition_val100_t25_32step_summary.json
metrics/stage4_photo100k_condition_compare_stage3_summary.json
```

### Stage 2: photo100k degradation v2 fine-tune

```text
config: configs/latent_pretrain_photo100k_v2.yaml
run: latent_pretrain_photo100k_v2_b64
degradation preset: photo_v2
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_b64/checkpoints/best_eval_latent.pt
finished step: 20000
best eval/latent_loss: step 19000, 0.28528
best decoded PSNR proxy: step 15000, 21.91
final eval: step 20000, eval/latent_loss 0.28704, decoded_psnr 21.60
```

`photo_v2`는 `mild`보다 훨씬 강한 LR degradation이므로 Stage2 mild의
decoded PSNR 23.9대와 직접 비교하면 안 된다. 이 checkpoint는 v2 LR 입력을
diffusion condition latent로 안정적으로 넘기기 위한 기준 checkpoint다.

HF:

```text
checkpoints/stage2_photo100k_v2_b64_best_eval_latent.pt
metrics/stage2_photo100k_v2_b64_summary.json
```

### Stage 3: photo100k degradation v2 fine-tune

```text
config: configs/diffusion_photo100k_b32_v2.yaml
run: diffusion_photo100k_b32_v2
degradation preset: photo_v2
condition encoder:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/checkpoints/best_eval_noise.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32/checkpoints/best_eval_noise.pt
finished step: 20000
best eval/noise_mse: step 19000, 0.00821
best decoded PSNR proxy: step 19000, 23.44
sampled val100 t50 32-step:
  SR PSNR:      22.6699
  bicubic PSNR: 22.4103
  delta:        +0.2595
  wins/losses:  63 / 37
```

정성 확인:

- 강한 noise/JPEG 계열에서는 bicubic보다 denoise가 되는 샘플이 있다.
- 일부 샘플에서는 색/대비 과보정, 녹색 점 artifact, 과한 texture smoothing이
  보인다.
- 따라서 Stage3 v2는 최종 품질 checkpoint가 아니라 Stage4 v2
  condition-start 안정화의 시작점으로 본다.

HF:

```text
checkpoints/stage3_photo100k_v2_b32_best_eval_noise.pt
metrics/stage3_photo100k_v2_b32_summary.json
metrics/stage3_photo100k_v2_val100_t50_32step_summary.json
```

### Stage 4: photo100k degradation v2 condition-start

```text
config: configs/diffusion_photo100k_b32_stage4_condition_v2.yaml
run: diffusion_photo100k_b32_stage4_condition_v2
degradation preset: photo_v2
condition encoder:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v2_b64/checkpoints/best_eval_latent.pt
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_stage4_condition_v2/checkpoints/best_eval_condition_decoded.pt
initialized from:
  /home/jwheojjang/scratch/sr-diffusion/runs/diffusion_photo100k_b32_v2/checkpoints/best_eval_noise.pt
finished step: 5000
best decoded checkpoint: step 1000
best decoded PSNR proxy: step 1000, 22.12
sampled val100 t25 32-step:
  SR PSNR:      22.8426
  bicubic PSNR: 22.4103
  delta:        +0.4323
  wins/losses:  70 / 30 vs bicubic
  vs Stage3 v2: +0.1727, wins 81 / losses 19
```

정성 확인:

- Stage3 v2보다 평균/승률은 개선됐고, 일부 overshoot가 완화됐다.
- 하지만 어두운 영역과 강한 artifact 샘플에서 cyan/green 점 artifact가
  아직 보인다.
- 다음 단계는 더 긴 v2 condition-start보다 artifact 억제 loss/샘플링 조정,
  A/B review가 우선이다.

HF:

```text
checkpoints/stage4_photo100k_condition_v2_b32_best_eval_condition_decoded.pt
metrics/stage4_photo100k_condition_v2_b32_summary.json
metrics/stage4_photo100k_condition_v2_val100_t25_32step_summary.json
metrics/stage4_photo100k_condition_v2_compare_stage3_v2_summary.json
```

### Stage 2: photo100k v3 noise XL condition encoder

```text
config: configs/latent_pretrain_photo100k_v3_noise_xl.yaml
run: latent_pretrain_photo100k_v3_noise_xl_b64
degradation preset: photo_v3_noise_mix
params: 18.944M
selected checkpoint:
  /home/jwheojjang/scratch/sr-diffusion/runs/latent_pretrain_photo100k_v3_noise_xl_b64/checkpoints/best_eval_latent.pt
finished step: 80000
best eval/latent_loss: step 66000, 0.27230
best decoded PSNR proxy: step 72000, 21.52
final eval: step 80000, eval/latent_loss 0.27592, decoded_psnr 21.51
```

비교:

```text
small v3 best:
  step 11000, eval/latent_loss 0.28187, decoded_psnr 21.30
XL best:
  step 66000, eval/latent_loss 0.27230, decoded_psnr 21.38
XL best PSNR proxy:
  step 72000, eval/latent_loss 0.27940, decoded_psnr 21.52
```

판단:

- XL condition encoder는 small v3를 확실히 넘었다.
- Stage4 XL을 바로 시작하지 않았다.
- Stage4 전에 `best_eval_latent.pt`, `step_0072000.pt`, `latest.pt` 세 후보의
  condition-only decoded sample을 같은 validation 이미지에서 비교하는 것이 좋다.
- `best_eval_latent.pt`는 latent loss 최선이고, `step_0072000.pt`와
  `latest.pt`는 decoded PSNR/latent MSE 측면에서 볼 가치가 있다.

HF:

```text
checkpoints/stage2_photo100k_v3_noise_xl_b64_best_eval_latent.pt
checkpoints/stage2_photo100k_v3_noise_xl_b64_step_0072000.pt
checkpoints/stage2_photo100k_v3_noise_xl_b64_latest.pt
metrics/stage2_photo100k_v3_noise_xl_b64_summary.json
configs/latent_pretrain_photo100k_v3_noise_xl.yaml
configs/diffusion_photo100k_xl_stage4_condition_v3.yaml
```

## 현재 관찰 / 판단

- 현재 Colab 기본 추론은 생성형 Stage 3/4가 아니라
  `LR -> Stage 2 XL step 72000 -> residual refiner v2 step 39000
  -> Stage 1 decoder` 경로다.
- 최신 multiscale Stage 2 step 46000은 기존 Stage 2보다 구조/색/PSNR이
  개선됐지만 public Colab 기본 경로로 아직 승격하지 않았다.
- 생성형 경로의 현재 선택 후보는 photo-detail Stage 4 step 8000이다.
  Stage 3과 Stage 4는 직렬 모듈이 아니라 서로 교체해서 쓰는 diffusion
  checkpoint다.
- 사용자 체감 병목은 여전히 missing fine detail과 strong-input smoothing이다.
- frozen VGG feature-supervised perceptual continuation은 12000 step에서
  정상 완료됐지만 시각적 detail 목표에는 실패했다. dual-multiscale LSDIR
  run은 완료됐고, clean/mild 수치 개선은 있으나 perceptual detail 돌파로
  보기는 어렵다.
- 2026-06-15 Stage1 decoder capacity audit 결과, Stage1 VAE recon은
  `photo_detail_mix` val100에서 mean PSNR `41.8121`, highpass ratio `0.9965`,
  laplacian ratio `0.9553`이다. 같은 val100에서 Stage2 dual-context best98000
  decoded base는 mean PSNR `26.4889`, highpass ratio `0.7886`, laplacian ratio
  `0.3191`이다. 현재 가장 약한 부분은 Stage1 decoder보다 Stage2 LR-to-latent
  predictor의 conditional-mean smoothing으로 판단한다.
- Stage2 trainer eval에는 `decoded_mean_psnr`, `decoded_ssim`,
  `highpass_energy_ratio`, `missing_energy`, `excess_energy`,
  `mean_psnr_detail_score`를 추가했다. 기존 `decoded_psnr`는 global MSE 기반,
  `decoded_mean_psnr`는 mean per-image PSNR이므로 숫자가 다르게 나온다.
- high-frequency detail branch v1b는 완료됐다. 이 branch는 Stage 2 dual-context
  best98000과 Stage 1 decoder를 frozen으로 두고, decoded base SR 위에
  image-space high-frequency residual만 더한다.
- 첫 v1 장기 run은 augmentation을 넣기 위해 step `7800`에서 조기 중단했다.
  이는 train `133450`장, batch `4` 기준 `0.234 epoch`다.
- v1b는 `configs/detail_branch_v1b_aug_photo130k_lsdir.yaml`로 회전 없이 hflip,
  texture-biased crop retry, 약한 HR color jitter만 추가했고 `40000` micro-steps
  까지 완료했다.
- 선택 checkpoint는 step `39500`의 `best_eval_detail.pt`다. val100 기준
  PSNR delta `+0.0461 dB`, SSIM delta `+0.00268`, mean delta `+0.0575`,
  wins `98/100`이다. final `40000`은 `+0.0444 dB`, `+0.00277`, wins `98/100`.
- v1c step `6000`은 v1b보다 개선됐고, v1d 3.02M capacity run은 그
  checkpoint에서 이어 정확히 3 epoch를 완료했다. 선택 step `99500`은
  ordinary val100 PSNR `+0.1646 dB`, SSIM `+0.00647`, wins `99/100`이다.
- `decoded_psnr + 5 * detail_ratio`는 shortlist score다. detail energy만
  높이는 인공 고주파/노이즈를 보상할 수 있으므로 이것만으로 승격하지 않는다.

## 다음 작업

우선순위:

1. 현재 Stage2 continuation은 원래 LR로 보존하되, 같은 objective의 장기
   continuation이 SwinIR gap이나 visible detail을 해결할 것으로 기대하지 않는다.
2. latent residual v1과 signed-wavelet residual v2는 모두 종료했다. v2는
   step20000까지 노이즈를 제거했지만 residual/diversity도 zero 쪽으로 수렴해
   실제 missing detail을 만들지 못했다. 같은 objective continuation은 하지 않는다.
3. 다음 생성형 detail 실험은 LR에서 근거가 있는 위치만 여는 learned
   uncertainty/detail mask와 patch-level perceptual/adversarial supervision을
   결합하는 two-head 구조를 우선 검토한다.
4. clean-fidelity gap 개선은 별도 Stage2/base 구조 연구로 유지하고,
   생성형 detail 목표와 섞지 않는다.
5. public Colab 기본 경로는 guarded-detail Stage2 v2 step10000으로 변경했다.
   tile batch size 기본값은 T4 안정성을 위해 `1`이다. residual refiner v2,
   detail branch v1d, masked v2는 WebUI 옵션으로 유지한다.
6. `detail-need mask` GT target/proxy/진단 metric 구현과 photo-detail val100
   진단은 완료됐다. GT target 상위 20%는 missing-detail `0.4878`을 포착하고
   밀도는 `2.4389x`다. 최고 observable proxy는 highpass disagreement이며
   correlation `0.5403`, top20 capture `0.3252`, excess capture `0.4838`이다.
   상세 정의와 결과는 `docs/DETAIL_NEED_MASK_KO.md`를 따른다.
7. learned mask predictor와 masked detail branch v2 검증은 완료됐다. selected
   step38000은 ordinary val100에서 PSNR `+0.18177 dB`, SSIM `+0.00755`,
   wins `100/100`이지만 시각적으로 v1d와 거의 같고 step50000까지 plateau했다.
8. masked v2 추론/benchmark 재현 경로와 HF preset은 추가됐다. formal
   clean-bicubic benchmark와 real-degradation 고정 visual set 결과를 확인한 뒤
   연구 옵션 유지 여부만 결정한다. public 기본값은 guarded-detail Stage2 v2로
   이동했으며, masked v2는 아직 연구 옵션으로 유지한다.
9. 다음 detail 실험은 같은 objective continuation이 아니라 frozen fidelity base와
   learned mask 위에 작은 bounded patch perceptual/adversarial head를 붙인다.
   lowpass drift, PSNR, strong-input artifact, blind visual review를 guardrail로 둔다.
10. Stage1 audit 이후 Stage2 guarded-detail v2 probe를 추가했다:
    `configs/latent_pretrain_photo130k_lsdir_dual_detail_guarded_v2.yaml`.
    이 config는 dual-context best98000에서 이어 받고 VGG/GAN 없이 decoded/highpass
    supervision만 보수적으로 강화한다. best metric은
    `eval/mean_psnr_detail_score = decoded_mean_psnr + 2 * highpass_energy_ratio`이며
    승격 기준이 아니라 shortlist 기준이다.
11. Stage2 guarded-detail v2는 20000 micro-step까지 완료됐고 붕괴 없이 plateau했다.
    최종 step20000은 detail 기준 최고가 아니므로 같은 objective continuation은
    중단한다. 선택 후보는 `best_eval_mean_psnr_detail.pt` = step10000이다.
    val100 guardrail: decoded PSNR `24.6296`, mean PSNR `26.5050`,
    highpass ratio `0.8084`, missing energy `0.01897`. L40S bf16 추론 실측은
    128x128 LR tile 기준 `tile_batch=1` reserved `1.03GB`,
    `tile_batch=4` reserved `3.28GB`, `tile_batch=8` reserved `6.25GB`다.
    따라서 이 후보의 deterministic Stage2->Stage1 추론은 8GB GPU에서도
    tile batch 1로 가능하고, 12-16GB GPU면 여유롭다. 이 후보를 Colab 기본값으로
    승격했고, WebUI에 TTA inference 옵션(`Horizontal flip x2`, `Full x8`)을
    추가했다. 장기 Stage2 학습은 여전히 L40S 48GB급을 권장한다.
12. guarded-detail Stage2 v2 step10000의 formal 219-image TTA sweep도 완료했다.
    빠른 PSNR-only 평가(`--skip-ssim`) 기준 off/hflip/x8 mean Y PSNR은
    `27.8539`/`27.9067`/`27.9496 dB`다. x8은 off 대비 `+0.0957 dB`지만
    visible 차이는 작고 runtime은 약 8배다. 기본값은 `Off` 유지, x8은 느린
    리뷰 옵션으로만 본다. masked detail v2는 같은 sweep에서 `28.1429 dB`로
    더 높다. 다음은 TTA 확대가 아니라 detail supervision/structure 변경이다.

## 새 VM에서 Codex에게 줄 짧은 프롬프트

```text
이 repo는 LuSIR, 즉 /home/.../sr-diffusion 의 x4 latent diffusion SR 프로젝트다.
docs/HANDOFF_KO.md 와 docs/VM_RECOVERY_KO.md 를 먼저 읽고 이어서 작업해줘.
Stage 번호는 학습 순서이며 추론 직렬 경로가 아니다. Colab 기본은
LR -> guarded-detail Stage2 v2 step10000 -> Stage1 decoder다. tile batch size
기본값은 1이고, TTA inference 옵션이 있다. residual refiner v2는 conservative
옵션으로 남아 있다.
multiscale Stage2 step46000에서 시작한 VGG perceptual continuation은 12000
step에서 완료됐고 시각적 detail 개선이 없어 승격하지 않았다. 이후 LSDIR
고유 이미지 30000장과 zero-init 두 번째 context branch를 추가한 119.24M
dual-multiscale Stage2 run도 100000 step까지 완료됐다. 자동 best는 step98000이고
final100000은 strong preset에서 조금 더 안전하다. 둘 다 HF에 보존되어 있으며,
public/default 승격 전 contact sheet human review가 필요하다.
image-space high-frequency detail branch v1d는 정확히 3 epoch를 완료했고 step99500을
선택해 HF public artifact로 보존했다. ordinary photo_detail_mix val100은 PSNR
+0.1646 dB, SSIM +0.00647, wins 99/100이다. strict-bicubic DIV2K five-crop
진단은 31.9513 dB이고, v1c보다 +0.1358 dB다. 509.658M Stage4 XL은 clean input
과수정 때문에 29.5487 dB다. 이 진단은 정식 SOTA benchmark가 아니다.
정식 full-image x4 benchmark에서 v1d는 DIV2K/Set5/Set14/Urban100 모두
dual-context base를 개선했다. DIV2K Y PSNR/SSIM은 30.1602/0.83421이며 official
SwinIR classical x4는 같은 evaluator에서 31.0838/0.85228이다.
clean-bicubic Stage2 continuation은 원래 LR 5e-6에서 step15000 val100 proxy
25.057까지 오른 뒤 plateau했다. LR20x는 15.72로 붕괴했고, LR5x from-init
step4000은 25.033으로 원래 LR의 25.031과 사실상 같아 LR 부족이 핵심 병목은
아니다. 원래 LR continuation은 step17825에서 종료했고 best는 step15000이다.
latent residual v1은 실제 detail 대신 고주파 에너지만 늘려 중단했다. 이후
signed Haar residual diffusion v2를 구현하고 step20000까지 학습했지만,
최종 val100에서 t15/t25/t50 모두 v1d보다 PSNR/SSIM/Laplacian/highpass가
낮았다. 장기 학습은 noise와 residual/diversity를 함께 zero 쪽으로 줄여
missing detail을 만들지 못했다. 같은 noise-MSE objective continuation은 하지
않는다. 다음 생성형 detail 연구는 learned uncertainty/detail mask와
patch-level perceptual/adversarial supervision을 결합하는 two-head 구조를
우선 검토한다. 상세 결과는 docs/HIGH_FREQUENCY_RESIDUAL_DIFFUSION_KO.md와
metrics/wavelet_residual_diffusion_v2_long_final_summary.json에 있다.
detail branch v1d와 masked v2는 Colab WebUI에서 단일 이미지/tiled inference
연구 옵션으로 선택 가능하다. public 기본값은 guarded-detail Stage2 v2 step10000이다.
masked v2는 step38000을 선택했고 ordinary val100에서 v1d보다 소폭 개선했지만
고정 grid가 거의 같아 missing texture 문제를 해결한 것으로 보지 않는다.
Colab demo는 notebooks/sr_diffusion_colab_demo.ipynb 에서 Gradio WebUI로 실행되며,
업로드/model 선택/TTA/tile 설정/before-after 비교 slider를 제공한다.
상업적 이용은 금지이고, raw dataset은 GitHub/HF에 올리지 않는다.
```
