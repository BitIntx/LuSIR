# LuSIR VM Handoff / 대화 기반 인수인계

이 문서는 원문 채팅 로그가 아니라, 현재 대화에서 결정하고 실행한 내용을
다른 VM의 Codex/작업자가 바로 이어받을 수 있게 정리한 공개용 요약입니다.

최신 실패/부분 성공/다음 가설 기록은 `docs/TRIAL_AND_ERROR_KO.md`에 누적합니다.

프로젝트 공개명은 **LuSIR**(**Latent Upscaling via Self-trained Image
Restoration**)입니다. GitHub/HF/W&B/로컬 경로의 `sr-diffusion` 표기는 기존
artifact와 링크 호환을 위한 저장소/식별자 이름입니다.

## 2026-06-11 현재 상태

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
- 현재 사용자용 public deterministic 기본값은 residual refiner v2이다.
- 기준 deterministic condition 후보는 multiscale Stage 2 step `46000`이고,
  최신 보존 연구 후보는 dual-context LSDIR Stage 2 step `98000`이다.
- 현재 detail 연구 후보는 dual-context LSDIR Stage 2 step `98000`과 Stage 1
  decoder를 frozen으로 두고 image-space high-frequency residual만 더하는
  detail branch v1b step `39500`이다. public Colab 기본값은 아직 아니며,
  단일 이미지/tiled inference runner 통합 전까지는 review artifact로 보존한다.

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
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/4akqckxu>
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

### 최신 완료 detail branch v1b

- run: `detail_branch_v1b_aug_photo130k_lsdir`
- config: `configs/detail_branch_v1b_aug_photo130k_lsdir.yaml`
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/1o3aavi9>
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
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/6zt2do4v>
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
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/nrqhw05u>
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
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/3v6wmf5o>
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
  - CLI `--residual-strength`, Colab WebUI `Residual correction strength`
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
  - W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/so0lbyte>
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
  - <https://wandb.ai/jwheo/sr-diffusion/runs/6h0124us>
  - <https://wandb.ai/jwheo/sr-diffusion/runs/0p3lfqt7>
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
- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/lrb6nco9>
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

- W&B: <https://wandb.ai/jwheo/sr-diffusion/runs/edfko8e8>
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

- GitHub: <https://github.com/BitIntx/sr-diffusion>
- Hugging Face: `jwheo/sr-diffusion`
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
- Colab demo: Gradio WebUI 기반. 유저 업로드가 기본이고, residual strength,
  tile overlap, tile batch size, diffusion steps를 slider로 조정한다. 출력은
  bicubic/Stage 2 condition/Input LR nearest 중 하나와 SR output을 before/after
  slider로 비교한다.
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
- `decoded_psnr + 5 * detail_ratio`는 shortlist score다. detail energy만
  높이는 인공 고주파/노이즈를 보상할 수 있으므로 이것만으로 승격하지 않는다.

## 다음 작업

우선순위:

1. 새 fixed review set `detail_v1`을 기준으로 이후 모델을 비교한다.
   - review set:
     `/home/ubuntu/scratch/sr-diffusion/review_sets/detail_v1/review_manifest.csv`
   - residual refiner v2 baseline outputs:
     `/home/ubuntu/scratch/sr-diffusion/review_outputs/residual_refiner_v2_detail_v1`
   - report:
     `/home/ubuntu/scratch/sr-diffusion/review_reports/residual_refiner_v2_detail_v1/report.html`
   - metrics:
     `/home/ubuntu/scratch/sr-diffusion/review_reports/residual_refiner_v2_detail_v1/summary.json`
2. detail branch v1b selected checkpoint를 같은 `detail_v1` fixed review set에서
   residual refiner v2 baseline과 비교한다.
   - config: `configs/detail_branch_v1b_aug_photo130k_lsdir.yaml`
   - review runner: `tools/eval/run_fixed_review_detail_branch.py`
   - selected checkpoint:
     `/home/ubuntu/scratch/sr-diffusion/runs/detail_branch_v1b_aug_photo130k_lsdir/checkpoints/best_eval_detail.pt`
   - HF preset target: `detail_branch_v1b`
3. `+0.01 dB` 수준 변화는 시각 개선 근거 없이 승격하지 않는다.
4. public Colab 기본 경로는 기존 residual refiner v2를 유지한다. detail branch를
   기본 경로로 승격하려면 먼저 단일 이미지/tiled inference runner와 WebUI 모델
   옵션을 추가해야 한다. Colab은 `notebooks/sr_diffusion_colab_demo.ipynb`에서
   `tools/demo/colab_webui.py`를 실행하는 WebUI 방식이다.
5. 다음 학습 ablation은 약한 SSIM/MS-SSIM loss, gate/residual 개방, 또는
   degradation-aware detail gate를 비교한다. SSIM만 무리하게 올리면 다시
   smoothing을 보상할 수 있으므로 시각 비교를 우선한다.

## 새 VM에서 Codex에게 줄 짧은 프롬프트

```text
이 repo는 LuSIR, 즉 /home/.../sr-diffusion 의 x4 latent diffusion SR 프로젝트다.
docs/HANDOFF_KO.md 와 docs/VM_RECOVERY_KO.md 를 먼저 읽고 이어서 작업해줘.
Stage 번호는 학습 순서이며 추론 직렬 경로가 아니다. Colab 기본은
LR -> Stage2 XL step72000 -> residual refiner v2 step39000 -> Stage1 decoder다.
multiscale Stage2 step46000에서 시작한 VGG perceptual continuation은 12000
step에서 완료됐고 시각적 detail 개선이 없어 승격하지 않았다. 이후 LSDIR
고유 이미지 30000장과 zero-init 두 번째 context branch를 추가한 119.24M
dual-multiscale Stage2 run도 100000 step까지 완료됐다. 자동 best는 step98000이고
final100000은 strong preset에서 조금 더 안전하다. 둘 다 HF에 보존되어 있으며,
public/default 승격 전 contact sheet human review가 필요하다.
image-space high-frequency detail branch v1b는 완료됐다. Stage2/Stage1은
frozen이고 Stage3/4 diffusion은 사용하지 않는다. 첫 v1 run은 7800 micro-steps =
0.234 epoch에서 멈췄고, v1b는 hflip/texture crop/약한 HR color jitter만 추가해
40000 micro-steps까지 완료했다. 선택 checkpoint는 step 39500 best_eval_detail이며
val100 PSNR delta +0.0461 dB, SSIM delta +0.00268, wins 98/100이다. fixed review는
detail_v1 set을 기준으로 residual refiner v2와 비교한다. public Colab 기본값으로
올리려면 detail branch 단일 이미지/tiled inference runner가 먼저 필요하다.
Colab demo는 notebooks/sr_diffusion_colab_demo.ipynb 에서 Gradio WebUI로 실행되며,
업로드/slider 조정/before-after 비교 slider를 제공한다.
상업적 이용은 금지이고, raw dataset은 GitHub/HF에 올리지 않는다.
```
