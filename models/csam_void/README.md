# CSAM Void Segmentation Lab

원본 61×81 또는 81×61 (H×W) grayscale 이미지에서 Void만 분할하는 로컬 실험 러너. 기존 SubspaceAD/NAGL과 독립적이다. 사내 이미지·마스크·결과는 GitHub에 업로드하지 않는다. 저장소가 공개 상태이므로 dataset과 결과 경로는 저장소 밖에 둔다.

## 1. 최초 환경 설정 (Python 3.11 권장)

기존 PyTorch 환경이 맞으면 재사용한다. CUDA는 로컬 driver에 맞는 공식 PyTorch wheel을 먼저 설치한다. 예: 기존 NAGL 환경의 torch 2.7.1 / torchvision 0.22.1 조합. 이후 저장소 루트에서:

    python -m pip install -r models/csam_void/requirements.txt

### Pretrained Weight 사전 다운로드 (폐쇄망 / 오프라인 실행)

| 실험 | 필요한 가중치 | 공식 다운로드 |
|---|---|---|
| Exp0 — PaDiM | ResNet18 ImageNet-1K V1 (`resnet18-f37072fd.pth`, 약 44.7 MiB) | [PyTorch 공식 가중치 다운로드](https://download.pytorch.org/models/resnet18-f37072fd.pth) |
| Exp1–7 — CNN, MixStyle, Semi-supervised 등 | 없음 (처음부터 학습) | — |

- 공식 모델 문서: [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- 인터넷 연결이 가능한 PC에서 가중치를 다운로드하고, 사내 보안 절차에 따라 로컬 PC로 반입한다.
- 예시 저장 경로: `D:/pretrained_weights/resnet18-f37072fd.pth`
- 파일의 SHA-256을 확인하려면 PowerShell에서 `Get-FileHash D:/pretrained_weights/resnet18-f37072fd.pth -Algorithm SHA256`을 실행한다. 공식 파일명에 포함된 SHA-256 접두사는 `f37072fd`이다. 접두사 확인은 전체 해시 검증을 대체하지 않는다.
- `--padim-weights`를 지정하면 Exp0이 로컬 가중치를 읽으며 torchvision의 자동 가중치 다운로드를 사용하지 않는다. Exp1–7은 외부 모델이 필요 없다. Python 의존성은 별도 설치되어 있어야 한다.

폐쇄망 전체 실험 실행 예시:

```powershell
python models/csam_void/run.py --manifest D:/csam_data/manifest.csv --out D:/csam_results/run01 --protocol both --padim-weights D:/pretrained_weights/resnet18-f37072fd.pth
```

인터넷 사용이 가능한 환경에서만 `--padim-weights`를 생략해 공식 가중치를 자동 다운로드하도록 할 수 있다. 모델 파일, 사내 데이터, 학습 결과는 GitHub에 커밋하지 않는다.

## 2. Dataset

기존 NAGL CSV 형식 재사용. 모든 경로는 CSV 기준 상대경로 또는 절대경로. 61행×81열 또는 81행×61열의 8-bit 또는 16-bit grayscale PNG/TIFF를 사용한다. 첫 이미지에서 원본 (H,W)를 자동 인식하고 모든 이미지와 mask가 동일한 shape인지 검사한다. transpose/회전하지 않는다. CNN은 원본 shape를 유지하고 PaDiM은 양 방향 letterbox 및 원본 좌표 복원을 지원한다. RGB/DAT는 사전에 변환한다. mask는 같은 크기에서 0=background, 양수=Void. label=0은 확인된 정상(FPN 포함), 1은 실제 Void가 있는 완전한 pixel mask 필수, -1은 미라벨 이미지(train만). 정상 mask는 빈 칸 가능. 미확인 정상은 -1로 넣는다. ref는 train으로 간주한다. 실제 Void 중 일부만 마킹된 이미지는 완전한 mask로 사용하지 않는다.

    image,mask,label,split,equipment,lot
    images/a.png,,0,train,E1,Y0001
    images/b.png,masks/b.png,1,train,E1,Y0001
    images/c.png,,0,val,E2,Y0002
    images/d.png,masks/d.png,1,val,E2,Y0002
    images/e.png,,0,test,E3,Y0003
    images/f.png,masks/f.png,1,test,E3,Y0003
    images/g.png,,-1,train,E1,Y0004

위 예시는 형식 설명용이며 충분한 평가 데이터가 아니다. 각 train/val/test에 정상과 실제 Void가 있어야 한다. PaDiM train 정상 최소 2장. 동일 root lot은 전체 설비에서 같은 lot ID로 적고 한 split에만 넣는다. 동일 pixel 이미지도 중복 거부한다. split을 임의 자동 생성하지 않는다(누수 방지). 설비별 일반화 평가에는 각 설비별 독립 lot 및 실제 Mask가 필요하다. 실제 test를 hard negative나 합성 donor로 사용하지 않는다.

## 3. 한 줄 실행

    python models/csam_void/run.py --manifest D:/csam_data/manifest.csv --out D:/csam_results/run01 --protocol both

기본 30 epochs, batch 8, seed 42, validation image recall 0.95, 1px 및 2–4px 연결 성분 hit recall 각각 0.90. --protocol both는 pooled 및 설비별 holdout 전체를 실행한다. holdout은 해당 설비를 train/val에서 제외하고 해당 설비 test만 평가한다. 평가 불가능한 fold는 이유와 함께 skip. --protocol pooled만 사용하면 빠른 비교 가능. 출력 폴더는 새 경로여야 한다. --experiments로 일부 단계만 선택 가능. --device cpu 또는 cuda 지정 가능. --epochs, --batch, --recall, --region-recall, --seed 조절 가능.

| 실험 | 내용 |
|---|---|
| Exp0 | ResNet18 ImageNet feature + 위치별 Gaussian/Mahalanobis, 독립 PaDiM-style baseline |
| Exp1 | 실제 Void mask가 있는 이미지와 그 배경으로 full-resolution CNN |
| Exp2 | 확인된 정상 이미지 추가, normal/void balanced sampling |
| Exp3 | 상위 1% background pixel BCE를 추가하는 online hard-negative mining |
| Exp4 | 실제 train mask에서 positive 밝기 residual을 추출하여 정상 이미지의 같은 위치에 합성 |
| Exp5 | gain/offset/mild sharpening augmentation |
| Exp6 | 첫 CNN layer 뒤 random batch MixStyle, 학습에서만 사용 |
| Exp7 | train label=-1 샘플에 EMA teacher와 confidence-filtered pseudo labels 적용 |

각 실험은 독립 초기화이며 Exp1~7은 누적 ablation이다. 기존 NAGL FP를 별도로 자동 import하지 않는다. 확인된 train 정상 데이터로 포함하면 Exp3에서 활용 가능. Copy-paste는 동일 위치에서 양의 residual이 존재할 때만 합성한다. 설비별 PSF를 복원하지 않으며 합성 이미지의 실제성은 확인 필요. MixStyle은 batch 내 무작위 pairing이며 반드시 다른 설비끼리 섞는 것은 아니다. Exp7은 미라벨 train 데이터가 없거나 confident positive pseudo-label이 없으면 skip한다. 재학습 없는 resume은 현재 미지원.

Exp0은 기존 사내 PaDiM checkpoint와 동일한 구현이라는 보장이 없다. resize/padding, feature layer, channel 수, smoothing 차이가 있을 수 있다. --padim-size, --padim-dim, --padim-smooth 조절 가능. CNN은 resize나 pooling 없이 원본 해상도를 유지한다.

## 4. 결과

출력 경로의 index.html을 브라우저에서 연다. pooled 및 각 holdout report.html, summary.csv, results.json, selected.json, split_audit.json, config.json을 생성한다. 모델별 model.pt, threshold.json, history.json(CNN), val/test scores.csv, raw NPY, mask PNG, overlay PNG를 저장한다. overlay 빨강=예측, 초록=정답, 노랑=겹침. 모든 val/test 이미지에 `<id>_panel.png` (Original | Anomaly map | Pred mask)도 저장한다. 세 패널은 확대 및 보간 없이 원본 해상도(61×81 또는 81×61, H×W) 그대로 배치한다. 제목, 여백, colorbar가 포함되어 전체 합성 PNG의 크기는 원본 이미지 한 장보다 크다. 기존 overlay 파일의 확대 방식은 변경하지 않는다. Heatmap 표시 범위는 실험별 validation threshold의 0~2배로 고정(0 이하 threshold는 0~1), 상한 초과값은 색상만 포화하며 raw NPY 및 예측 mask는 그대로 유지한다. Colorbar와 threshold/max score를 표시하고, pred mask는 흰색=불량, 검은색=배경이다. 결과에는 원본 경로 및 이미지 파생물이 있으니 사내 자료로 취급한다.

Threshold는 validation에서 이미지 recall 및 존재하는 1px/2–4px 연결성분 recall 목표를 충족하는 가장 높은 값으로 고정. test에서 다시 조정하지 않는다. best checkpoint 및 추천 모델은 validation 기준(작은 영역 recall 미달 우선, worst-equipment FPR, 전체 FPR, Dice)으로 선택한다. Test는 독립 보고만 한다. size bin은 지름이 아니라 8-connected component의 pixel 면적이다. region recall은 최소 1px overlap으로 정의하고 Dice/IoU로 정밀도를 보완한다. 해당 결함 크기가 없으면 null이며 recall 충족을 입증하지 못한다. 운영 적용 전 충분한 신규 lot/설비 및 작은 Void 수량이 필요하다.

## 5. 로컬 검증

    python -m unittest discover -s models/csam_void -p test_run.py
    python models/csam_void/make_demo.py --out D:/csam_demo
    python models/csam_void/run.py --manifest D:/csam_demo/manifest.csv --out D:/csam_demo_results --epochs 2 --min-steps 2 --padim-size 64 --padim-dim 16 --protocol pooled

합성 데모는 연결/출력 확인 용도이며 실제 성능 근거가 아니다. Exp7은 2 epoch smoke에서 skip될 수 있다. 일부 단계 실패/skip 시 나머지 실험을 계속하며 종료 코드 2 및 error.txt/보고서에 이유를 남긴다.

원리 참고: [PaDiM](https://arxiv.org/abs/2011.08785), [MixStyle](https://arxiv.org/abs/2104.02008). 이 구현은 논문 모델 재현이 아닌 CSAM 특화 실험용 독립 코드이다.

CI: `.github/workflows/csam-void-smoke.yml`은 CPU 합성 데이터에서 pooled/holdout Exp0~6과 Exp7을 실행한다. Exp0의 CI용 ResNet18은 다운로드 없는 random weights로 연결만 검증한다. Exp7의 실제 update 경로 확인용 낮은 confidence threshold는 합성 smoke 전용이며 실데이터 기본값은 `--pseudo-positive 0.98 --pseudo-negative 0.02`이다. Exp7의 선택 checkpoint는 pseudo-label update 이후 epoch로 제한된다.

## 6. 실시간 로그

실행 명령은 그대로 유지한다. 콘솔과 출력 루트의 `run.log`에 동일한 로그를 남긴다. 각 protocol의 설비별 normal/void/unlabeled/lot 수량, 현재 실험, epoch, batch, 평균 loss, validation recall/FPR/worst FPR/Dice/IoU/크기별 region recall, threshold, best 갱신, 경과 시간 및 ETA를 표시한다. CUDA 메모리는 PyTorch allocated/reserved/peak allocated와 GPU 전체 용량(MiB)이며 전체 시스템 GPU 사용량이나 GPU utilization(%)이 아니다. `--log-every 50`은 기본 batch 출력 간격이며 `--log-every 10`으로 더 자주 출력, `--log-every 0`은 batch 로그만 끈다. Epoch 로그는 계속 출력된다. ETA는 완료된 작업의 평균으로 계산하며 실험별 소요시간 차이 때문에 protocol ETA는 거친 추정치이다. PaDiM은 epoch 없이 정상 feature 집계와 covariance 계산을 표시한다.

`[Aug]`는 해당 epoch의 성공/시도 copy-paste 횟수, supervised domain augmentation 샘플 수, MixStyle 호출 횟수(실제 적용, Exp7의 student unlabeled forward 포함), accepted positive pseudo pixel 수를 보여준다. 같은 이미지를 반복 sampling할 수 있으므로 독립 이미지 수가 아니다. `history.json`은 epoch마다 업데이트되어 학습 중에도 읽을 수 있으며 best_epoch, best_updated, seconds 및 augmentation count가 추가되었다. Exp7이 pseudo label 부족으로 skip되어도 진행된 epoch의 history가 남는다. 학습 재개 기능은 제공하지 않는다. 데이터 audit 실패도 run.log에 남기므로 재시도 시 새로운 --out 경로를 사용한다. 로그와 결과에는 사내 데이터 경로·설비 식별자가 포함될 수 있으므로 GitHub로 올리지 않는다.

## 7. 독립형 통합 보고서 (오프라인)

전체 실험 완료 시 출력 루트에 `index.html`과 통합 `summary.csv`를 자동 생성한다. index.html 하나만 다른 PC로 복사해도 모든 protocol/실험의 validation·test 지표, validation-selected 모델, holdout 비교, 설비별 상세 지표, region hit/count, 0–100% 비교 막대그래프를 볼 수 있다. HTML/CSS가 내장되어 인터넷이나 Python, 다른 HTML 파일이 필요 없다. Test로 모델을 재선택하지 않는다. 실험 상태/오류 및 N/A도 그대로 표시한다. 통합 CSV는 overall/equipment scope와 split을 구분하고, 기존 하위 폴더의 summary.csv와 report.html은 유지한다.

기존 완료 결과에서 재학습 없이 통합 보고서 생성 (Python 표준 라이브러리만 사용):

    python models/csam_void/build_report.py --root D:/csam_results/run01

기존 출력 루트의 index.json, 각 완료 protocol의 results.json 및 selected.json, config.json(선택 기준·목표 표시)이 필요하다. 기존 하위 report.html이 없어도 성능 수치는 생성된다. 결과 JSON이 누락되면 해당 protocol에 missing 안내를 표시한다. 기존 index.html 및 루트 summary.csv는 새 버전으로 교체된다. 원본 이미지/overlay/개별 scores.csv는 HTML에 내장하지 않으므로 필요할 때만 원래 하위 폴더를 함께 복사한다. 설비명과 성능 수치도 사내 보안 정책에 따라 취급한다.

## 8. P0–P5 PaDiM Backbone adaptation (MixStyle)

기존 Exp0–7은 그대로 유지하고 별도 진입점으로 P0–P5를 순서대로 실행한다. 추가 dependency나 pretrained model 다운로드는 없다. 기존 ResNet18 공식 가중치 `resnet18-f37072fd.pth`를 반드시 로컬로 지정한다. 각 실험은 동일한 초기 가중치, seed, manifest split, 원본 H/W 각각 4배 확대(패딩 없음), feature layer1/2, 192개 중 40개 channel 선택 및 Gaussian fit 설정을 사용한다. 이전 실험의 weight를 이어받지 않는다.

    python models/csam_void/run_padim.py --manifest D:/csam_data/manifest.csv --out D:/csam_results/padim01 --protocol both --padim-weights D:/pretrained_weights/resnet18-f37072fd.pth

| 실험 | Backbone 학습 | 데이터 |
|---|---|---|
| P0 p0_pretrained | 없음 (기존 Exp0와 동일한 알고리즘, P suite 기본 geometry/depth/dim은 변경) | normal Gaussian fit |
| P1 p1_classification | local head의 max logit으로 image BCE | normal+masked void |
| P2 p2_classification_mixstyle | P1 + cross-equipment MixStyle | 동일 |
| P3 p3_consistency | 고정 pretrained teacher의 원본 layer1/2 feature와 photometric 변형 student의 상대 MSE + clean anchor | normal only |
| P4 p4_consistency_mixstyle | P3 + cross-equipment MixStyle | 동일 |
| P5 p5_pixel_mixstyle | P2 + native mask supervision (class-balanced BCE, hard background BCE, Dice) | normal+masked void |

P1/P2 및 P3/P4는 matched ablation pair다. P5는 P2에 pixel loss를 추가한다. P1–P5는 stem 및 BatchNorm 고정, layer1/2 fine-tuning (layer4/fc 미사용), 기본 10 epochs, backbone LR 1e-5, head LR 1e-3. Fixed final epoch를 채택하며 validation proxy는 진단용이다. 분류 head는 fused feature의 local logits을 native shape로 복원해 max pooling하고 P5는 같은 map을 mask와 비교한다. 초기 feature stride 때문에 1px 위치 복원은 제한될 수 있다. MixStyle은 stem, layer1 이후 확률 0.5, Beta alpha 0.1로 서로 다른 equipment의 통계를 혼합한다. 학습 batch는 equipment/class group sampling으로 구성한다. MixStyle 실험은 최소 2개 학습 설비 필요하며 실제 적용이 없으면 skipped. P3/4 teacher는 ImageNet weight 고정, normal train만 사용하고 미라벨/val/test는 학습에 사용하지 않는다. P1–P5는 학습 완료 후 head 제거 및 backbone freeze, MixStyle OFF. P0–P5 모두 실제 normal train만으로 원래 PaDiM Gaussian fit 및 동일한 inference를 수행한다.

- 기본 목표 image recall 95%, 1px/2–4px region recall 90% 유지. 별도로 validation-only threshold sweep과 image-only recall/FPR ceiling operating points를 저장해 threshold 문제와 표현력 문제를 구분한다. 1px, 2–4px는 connected component 면적이다. Test는 validation에서 정한 threshold 그대로 평가하며 Test 결과로 선택하지 않는다.
- `index.html`: 단독 열람 가능한 통합 지표, val/test Recall–FPR curves 및 operating points, backbone proxy 및 실제 weight 변화량. `summary.csv`: 전체 protocol/설비 수치. 각 실험 `model.pt`: PaDiM stats + backbone; P1–P5 `backbone.pt`, `history.json`, `backbone_summary.json`; P0–P5 `operating_points.json`, val/test map/overlay.
- 빠른 확인: `--protocol pooled --epochs 2`. 일부 실험: `--experiments p0_pretrained p3_consistency p4_consistency_mixstyle`. 기본 batch 8, `--backbone-lr`, `--head-lr`, `--mix-prob`, `--mix-alpha`, `--pixel-weight`, `--anchor-weight` 조절 가능. 기존 `--padim-size`, `--padim-dim`, `--padim-smooth`, `--log-every` 동일.
- P0는 같은 geometry/depth/dim을 지정하면 이 저장소의 Exp0 구현과 동일하며 사내 배포 PaDiM 설정과 동일하다는 보장은 없다. 실제 production 설정이 다르면 비교 전에 일치시킨다. 설비별 blur/PSF 차이는 MixStyle 통계 혼합만으로 해결되지 않을 수 있다. 정상 feature distillation도 실제 void 신호를 보존한다는 보장은 없다. 합성 CPU smoke는 실행 검증이며 실제 CSAM 성능 검증이 아니다.

### P0–P5 기본값: 기존 Best 설정

`run_padim.py`의 기본값은 `--upsample-factor 4 --max-layer-depth 2 --n-features 40`이다. 예: native (H,W)=(81,61) → bilinear (324,244), padding 없음 → layer1 (64,81,61), layer2 (128,41,31) → nearest feature alignment → 192 channels 중 seed로 40개 선택 → 위치별 40차원 Gaussian. Fine-tuning head와 fixed teacher도 동일한 layer1/2만 사용하고, MixStyle은 stem 및 layer1 출력에서 다음 layer 입력에 적용한다. layer3/4 및 fc는 사용하지 않고 weight 변경하지 않는다. 입력 및 mask 방향을 유지한다. Factor 4에서 layer1 grid가 native와 같아 score map 및 P5 logits를 불필요하게 4배 확대 후 다시 축소하지 않는다. 다른 factor에서는 native shape로 bilinear 복원한다.

옵션은 `--upsample-factor` / `--upsample_factor`, `--max-layer-depth` / `--max_layer_depth`, `--n-features` / `--n_features` / 기존 `--padim-dim`을 지원한다. `--padim-size 224`는 기존 square letterbox 방식 선택이며 factor와 동시 지정할 수 없다. depth 1/2/3에서 최대 channel 수는 64/192/448이다. 기존 `run.py`의 기본값은 기존대로 224 square, depth 3, 100 features이며 같은 Best 옵션을 명시해서 Exp0와 P0를 직접 비교할 수 있다. 저장된 `model.pt`에 geometry, layer depth, n_features, feature_shape 및 선택 channel index를 남긴다. 기존 사내 모델의 bilinear/normalization/feature selection/score smoothing까지 일치하는지는 별도 확인해야 한다.


## 9. 기존 anomalib 실험과 평가 metric 통일

| Split | Metric (JSON key) | 입력 |
|---|---|---|
| Validation | Image AUROC (`image_auroc`) | pred_score, gt_label |
| Validation | Pixel AUROC (`pixel_auroc`) | anomaly_map, gt_mask |
| Test | Image F1 (`image_f1`) | pred_label, gt_label |
| Test | Pixel AUPRO@0.30 (`pixel_aupro`) | anomaly_map, gt_mask |
| Test | Pixel AUROC (`pixel_auroc`) | anomaly_map, gt_mask |

모든 Exp0–7/P0–5의 최종 validation/test 평가에 적용하고, protocol별 results.json/summary.csv/report.html 및 통합 index.html/summary.csv에 출력한다. 기존 auroc 키는 Image AUROC 호환용으로 유지한다. 실무용 Recall/FPR/Worst Equipment FPR/Region Recall/Dice/IoU 및 운영 threshold와 validation 기반 모델 선택 규칙은 변경하지 않는다. Epoch별 CNN 모델 선택용 metric은 기존대로 유지한다.

pred_score는 native anomaly map의 max, pred_label은 validation에서 결정된 threshold 이상 여부, gt_label은 manifest label, gt_mask는 원본 binary mask이다. AUROC는 전체 split pixel 또는 image를 모아서 계산하며 이미지별 평균이 아니다. F1은 2TP/(2TP+FP+FN), 분모가 0이면 0. Pixel AUPRO는 각 이미지의 8-connected GT component에 동일한 총 weight를 부여하여 PRO(영역별 pixel overlap 평균) 대 pixel FPR 곡선을 계산한다. 모든 score tie를 함께 처리하며 FPR 0.30에서 선형 보간한 뒤 0~0.30 면적을 0.30으로 나눈다. GT positive 또는 negative가 없으면 해당 AUROC/AUPRO는 null(N/A). 새로운 의존성은 없다.

이 구현은 anomalib 클래스와 같은 입력 및 표준 지표 정의를 따르는 독립 구현이며 anomalib 버전별 AUPRO 구현, score normalization/clipping, image/pixel threshold, map 복원/보간, 연결성 차이까지 수치 일치가 보장되지는 않는다. 완전한 수치 비교에는 동일한 split, 해상도, score/label 생성 규칙, anomalib 버전과 설정으로 동일한 raw map을 교차 평가해야 한다. 참고: https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/metrics/index.html

이전에 생성한 결과 JSON에는 새 지표가 없으므로 build_report.py로 HTML만 재생성하면 N/A로 표시된다. 새 평가 결과를 얻으려면 새 출력 경로로 실험을 실행한다. 픽셀 지표 계산은 전체 split 점수 정렬을 사용하므로 큰 데이터셋에서는 추가 CPU 메모리와 시간이 필요하다.
