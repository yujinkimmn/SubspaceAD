# PaDiM P0–P5 실험 정리

## 1. 실험 목적

작은 void를 찾아야 하는 검사에서는 정상 이미지의 밝기, 대비, 배경 texture 등이 달라지면 불량이 아닌 곳에도 anomaly score가 높게 나올 수 있다. 이를 **overkill(정상을 불량으로 오판)**이라고 한다.

이번 실험은 PaDiM의 검출 방식을 유지하면서 **정상과 실제 불량을 더 잘 구분하는 backbone feature를 학습하면 overkill을 줄일 수 있는지** 확인하기 위한 ablation(한 번에 한 가지 요소를 바꾸어 효과를 분리하는 실험)이다. 1px/2–4px처럼 작은 불량의 검출 능력을 잃지 않는지도 반드시 확인한다.

실험 방법과 코드 기준을 설명하는 문서이며, 실제 데이터 및 성능 수치는 공개하지 않는다. 수치 예시는 원리 설명용이다.

## 2. 공통 PaDiM 파이프라인

~~~mermaid
flowchart TD
    A["동일한 pretrained ResNet18"] --> B["P0: 고정 / P1–P5: backbone 학습"]
    B --> C["Backbone 고정, head 제거, MixStyle OFF"]
    C --> D["정상 train feature 추출"]
    D --> E["위치별 Gaussian fit"]
    E --> F["Validation anomaly map 및 threshold 보정"]
    F --> G["Test anomaly map 및 독립 평가"]
~~~

PaDiM은 위치마다 정상 feature가 어느 범위에 분포하는지 기억한다. 예를 들어 여러 정상 이미지의 같은 위치에서 추출한 40차원 feature 벡터들을 모으면 평균 벡터 μ와 covariance(공분산) Σ를 계산할 수 있다. 새로운 이미지의 해당 위치 feature f가 정상 분포에서 얼마나 떨어져 있는지 Mahalanobis distance로 계산한다.

```text
d = sqrt((f - μ)^T Σ^-1 (f - μ))
```

가상의 1차원 예에서 평균이 10, 표준편차가 2라면 feature 11의 거리는 0.5, feature 16의 거리는 3이다. 실제 구현은 정상 feature 40차원 전체의 공분산을 고려하며 covariance에 0.01 × I를 더해 안정화한 뒤 Cholesky 분해를 사용한다. 위치별 거리 전체가 anomaly map이고, map 최댓값은 image score가 된다.

모든 P0–P5의 최종 검출 방식은 같으며, **P1–P5는 Gaussian fit 전에 backbone을 조정한다.** 학습한 image classification/segmentation head 자체가 최종 검사 모델이 아니다.

### 공통 조건 (기존 PaDiM 기준)

| 항목 | 기본값 |
|---|---|
| Backbone | timm `resnet18.a1_in1k` |
| Pretrained checkpoint | `resnet18_a1_0-d63eafa0.pth` |
| 입력 전처리 | `ToImage → Resize(antialias=True) → ToDtype(torch.float32, scale=True)`, grayscale을 RGB 3채널로 복제 |
| ImageNet mean/std | 미적용 |
| Input geometry | `upsample_factor=4`, 원본 H/W 각각 4배, padding 없음 |
| Feature layers | `max_layer_depth=2` (ResNet18 layer1, layer2) |
| Feature channels | `n_features=40` (192개 중 seed로 고정된 동일 index 선택) |
| Score smoothing | `padim_smooth=0` (Gaussian blur 미적용) |
| Seed / batch | 42 / 8 |
| Gaussian fit | 확인된 정상 train 이미지 전용, 최소 2장 |

예: 원본 (H,W)=(81,61) → (324,244) → layer1 (64,81,61), layer2 (128,41,31) → layer2를 layer1 위치에 nearest alignment → 192 channels 중 40개 선택 → 위치별 Gaussian. 기본 설정의 feature grid는 이미 원본 해상도이므로 불필요한 확대/축소 없이 anomaly map을 얻는다.

가중치 파일은 [공식 timm 다운로드](https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth), [공식 모델 정보](https://huggingface.co/timm/resnet18.a1_in1k)를 참고한다. torchvision ResNet18 checkpoint와 구조가 유사해도 학습된 weight는 다르다. 이 연구용 구현이 기존 anomalib 배포 모델과 수치까지 완전히 같은 것은 아니다.

## 3. P0~P5 비교

| ID / 이름 | Backbone을 어떻게 학습하는가? | 학습에 사용하는 데이터 | 무엇을 확인하는가? |
|---|---|---|---|
| P0 `p0_pretrained` | 학습 없음; pretrained feature 그대로 | Gaussian fit용 정상 | 비교 기준 |
| P1 `p1_classification` | local head의 최대 logit으로 image BCE | 정상 + mask 있는 void | 정상/불량 분류 학습 효과 |
| P2 `p2_classification_mixstyle` | P1 + cross-equipment MixStyle | P1과 동일 | classification feature의 설비 일반화 |
| P3 `p3_consistency` | 정상 이미지 변형 전후 feature가 고정 teacher에 가깝도록 학습 | 정상만 | 불필요한 밝기/대비 차이에 대한 강건성 |
| P4 `p4_consistency_mixstyle` | P3 + cross-equipment MixStyle | P3와 동일 | 정상 consistency 학습에서 MixStyle 효과 |
| P5 `p5_pixel_mixstyle` | P2 + pixel mask loss | 정상 + mask 있는 void | 실제 void 위치 정보 학습 효과 |

### P0: pretrained PaDiM (baseline)

- 공식 ResNet18에서 그대로 feature를 추출한다.
- 확인된 정상 train feature를 위치별로 Gaussian fit한다.
- 다른 모든 실험은 이 기준과 동일한 PaDiM 설정으로 시작한다.

### P1 / P2 / P5: supervised adaptation

Layer1/2 feature를 합친 후 임시 1×1 convolution head로 위치별 logit을 만든다. Native shape로 복원한 logit의 최댓값과 image label을 BCE(binary cross-entropy, 정상/불량 판정 오차)로 비교하여 layer1/2를 fine-tune한다.

- P1: image label loss만 적용. 실제 void 위치를 맞췄는지는 직접 학습하지 않는다.
- P2: P1과 같은 image loss에 MixStyle을 추가한다.
- P5: P2에 실제 mask와 비교하는 pixel loss를 추가한다. Positive/negative 각각의 평균 BCE와 정상 배경 중 손실이 가장 큰 상위 1% pixel BCE를 평균하고, positive 이미지에 Dice loss를 더한다. 기본 `pixel_weight=1`이다.

P5의 목적은 단순히 image label을 구분하는 데 유리한 특징뿐 아니라 작은 void의 **위치**에 유용한 feature를 유도하는 것이다. Head와 MixStyle은 학습 후 제거되고 생성된 feature로 다시 PaDiM Gaussian을 fit한다.

### P3 / P4: normal-only consistency adaptation

고정된 pretrained teacher는 변형하지 않은 정상 이미지를 입력받고, student backbone에는 gain/offset/mild sharpening을 적용한 정상 이미지를 입력한다.

Student는 teacher의 원래 layer1/2 feature와의 relative MSE(평균 제곱 오차를 teacher feature 크기로 나눈 값)를 줄인다. Clean 이미지에서도 teacher와 멀어지지 않도록 anchor loss를 추가한다. 기본 `anchor_weight=1`이다.

- P3: photometric consistency + clean anchor.
- P4: P3의 변형 input 경로에 MixStyle 추가.

이 방식은 실제 불량을 학습하지 않아도 적용 가능하지만, 정상 변화에 무감각해지는 과정에서 실제 void도 구분하기 어려워질 수 있으므로 검증이 필요하다.

### MixStyle: P2 / P4 / P5

같은 정상 형태여도 설비마다 feature map의 평균 μ와 표준편차 σ가 다를 수 있다. MixStyle은 학습 batch에서 다른 equipment의 채널별 통계를 섞어 backbone이 특정 설비의 스타일에 과도하게 의존하지 않도록 유도한다.

```text
mixed_μ = λ μ_A + (1 − λ) μ_B
mixed_σ = λ σ_A + (1 − λ) σ_B
mixed_feature = (feature_A − μ_A) / σ_A × mixed_σ + mixed_μ
```

설명용 예시: λ=0.7, μ_A=0.2, μ_B=0.5이면 혼합 평균은 0.29이다. 실제 코드는 채널별 통계를 공간 방향으로 계산하고 λ를 Beta(alpha, alpha)에서 샘플링한다.

- 기본 `mix_prob=0.5`, `mix_alpha=0.1`.
- stem 및 layer1 출력(다음 layer 입력)에서 적용한다.
- Batch 내 다른 equipment를 partner로 고른다. 학습 설비가 2개 미만이거나 실제 MixStyle 호출이 없으면 해당 실험은 skip한다.
- Gaussian fit과 inference에는 적용하지 않는다. 실제 설비 간 blur/PSF 변화까지 해결한다는 보장은 없다.

### Backbone 학습 조건

P1–P5는 독립적으로 같은 checkpoint에서 시작한다. Stem, BatchNorm의 running statistics와 affine parameter, 사용하지 않는 상위 layer는 고정하고 layer1/2만 학습한다. Supervised 실험의 임시 head도 함께 학습한다.

| 파라미터 | 값 |
|---|---|
| Epochs | 10 |
| 최소 steps / epoch | 20 (또는 ceil(train 이미지 수 / batch), 둘 중 큰 값) |
| Optimizer | AdamW, weight decay 1e-4 |
| Backbone learning rate | 1e-5 |
| Head learning rate | 1e-3 |
| Sampling | 정상/불량·설비 group (P1/2/5), 설비 group (P3/4) |
| Checkpoint | Fixed final epoch; validation proxy는 진단용 |

## 4. 데이터 분할과 비교 조건

Manifest는 `image,mask,label,split,equipment,lot` 열을 사용한다. 0은 확인된 정상(검출 대상이 아닌 nuisance 포함), 1은 실제 void와 완전한 mask가 있는 이미지, -1은 미라벨 train이다. 미라벨 데이터는 P0–P5 adaptation이나 Gaussian fit에 사용하지 않는다. 선택 컬럼 `wafer` 또는 `wafer_id`는 시각화용으로만 사용한다.

- Train / val / test는 미리 지정하며 동일 root lot이 여러 split에 섞이지 않아야 한다.
- 같은 pixel 데이터의 중복을 검사한다. 각 split에 정상과 void가 있어야 한다.
- **Pooled:** 모든 설비의 지정 train/val/test를 합쳐 일반적인 성능을 평가한다.
- **Holdout:** 설비 하나를 train/val에서 제외하고 그 설비의 지정 test만 평가한다. 학습하지 않은 설비로 일반화하는지 확인한다.

Pooled와 각 holdout은 데이터 조건이 다르다. 반드시 같은 protocol 안의 P0–P5를 비교하며 test 결과로 학습 방법/threshold를 다시 고르지 않는다.

## 5. Threshold 및 평가 지표

각 모델/프로토콜의 validation에서 image raw score와 pixel raw map에 대해 각각 F1-adaptive threshold, min, max를 계산하고 독립적으로 정규화한다.

```text
normalized = clip((raw_score - threshold) / (max - min) + 0.5, 0, 1)
prediction = normalized > 0.5
```

따라서 0.5는 정상/불량을 구분하는 정규화된 기준이며, 서로 다른 모델의 raw score 값이나 raw threshold가 동일하다는 뜻이 아니다. Test에서는 validation의 값만 재사용한다. Image pred_label과 pixel pred_mask는 독립적으로 생성한다.

| Split | Metric | 의미 |
|---|---|---|
| Val | Image AUROC | Image 정상/불량 점수의 순위 분리 정도 |
| Val | Pixel AUROC | 정상/불량 pixel 점수의 순위 분리 정도 |
| Test | Image F1 | Image precision(불량 예측 중 실제 불량 비율)과 recall의 조화평균 |
| Test | Pixel AUPRO@0.30 | Pixel FPR 0~0.30 구간에서 각 정답 영역의 검출 비율을 동일 가중치로 평균한 곡선의 정규화된 면적 |
| Test | Pixel AUROC | Pixel 정상/불량 점수의 순위 분리 정도 |

실무 관점에서 추가로 아래 지표를 확인한다.

| 지표 | 의미 | 선호 방향 |
|---|---|---|
| Image FPR (overkill) | 정상 중 잘못 불량 판정한 비율 | ↓ |
| Worst equipment FPR | 설비별 FPR 중 가장 큰 값 | ↓ |
| Image Recall | 실제 불량 이미지 중 검출한 비율 | ↑ |
| Region Recall | 실제 void 연결 영역 중 예측 mask가 한 pixel 이상 겹친 비율 | ↑ |
| Dice / IoU | 예측 mask와 정답 mask의 겹침 정도 | ↑ |

예를 들어 (가상 수치), 정상 100장 중 5장을 불량으로 판정하면 FPR 5%, 실제 불량 20장 중 19장을 검출하면 recall 95%이다. 하지만 1px void 영역 10개 중 6개만 검출했다면 1px region recall은 60%이다. Image recall만으로 작은 void의 위치 검출 능력을 판단하면 안 된다.

### Selected model의 의미

선정 모델은 protocol 안에서 validation 결과만으로 아래 순서대로 비교한 후보이다.

1. 1px/2–4px 중 관측된 구간의 가장 낮은 region recall이 목표(기본 90%)에 미달하는 정도가 작은 모델.
2. Worst equipment FPR이 낮은 모델.
3. 전체 FPR이 낮은 모델.
4. Dice가 높은 모델.

Test 성능 최고 모델이나 Image AUROC 최고 모델을 뜻하지 않는다. Image recall 95%/small-region recall 90%는 목표이며, F1-adaptive 최종 threshold에서 두 목표를 자동으로 달성하지는 않는다. 크기별 정답 영역이 없는 경우 해당 region metric은 N/A로 처리한다.

### Recall–FPR trade-off

최종 F1 threshold의 결과와 별개로 validation raw score에서 공통 image/pixel threshold를 움직여 FPR 및 recall의 변화도 확인할 수 있다.

- 같은 validation image recall 95%를 달성할 때 어느 방법의 FPR이 더 낮은가?
- 같은 validation FPR ceiling 5%에서 어느 방법이 1px/2–4px를 더 많이 검출하는가?
- Validation에서 결정한 threshold를 그대로 test에 적용했을 때 위 장점이 유지되는가?

이 곡선의 threshold는 최종 정규화 0.5와 다른 raw diagnostic threshold이며, test 곡선은 진단용이다.

## 6. 어떤 비교를 통해 결론을 내릴까?

| matched pair | 확인할 질문 |
|---|---|
| P0 → P1 | Image label을 활용한 feature 학습이 기존 PaDiM보다 도움이 되는가? |
| P1 → P2 | 같은 classification 조건에서 MixStyle 자체가 overkill을 줄이는가? |
| P0 → P3 | Normal-only consistency가 정상 설비 편차를 줄이는가? |
| P3 → P4 | Normal-only 학습에서도 MixStyle이 추가 효과를 주는가? |
| P2 → P5 | 정확한 pixel supervision이 작은 void 검출을 개선하는가? |

P0 대비 P2/P4/P5의 개선 여부만 보면 여러 요소가 동시에 바뀌었으므로 MixStyle만의 효과라고 해석할 수 없다. 먼저 validation에서 matched pair로 원인을 분리한다. FPR 감소와 작은 영역 recall을 함께 비교하고, test 및 holdout의 독립 결과로 일반화 여부를 확인한다. AUROC 차이가 작더라도 특정 FPR/recall operating point에서 차이가 클 수 있다.

Backbone 학습 진단용 `weight_max_abs_change`가 0보다 크고 P2/P4/P5의 `mixstyle_calls`가 0보다 큰지도 확인한다. 임시 head AUROC/Dice 또는 consistency loss가 좋아졌다는 사실만으로 PaDiM이 개선되었다고 결론 내리지 않는다.

## 7. 실행 및 결과 확인

저장소 루트에서 PowerShell 예시 (경로는 가상):

```powershell
python -m pip install -r models/csam_void/requirements.txt
python -m pip check
python models/csam_void/run_padim.py --manifest D:/csam_data/manifest.csv --out D:/csam_results/padim01 --protocol both --padim-weights D:/pretrained_weights/resnet18_a1_0-d63eafa0.pth
```

`--protocol pooled`만 지정하면 pooled 평가만 실행한다. 초기 확인용으로 `--epochs 2`를 지정할 수 있고 일부 실험은 아래와 같이 선택한다. 항상 새로운 출력 경로를 사용한다.

```powershell
python models/csam_void/run_padim.py --manifest D:/csam_data/manifest.csv --out D:/csam_results/padim02 --protocol pooled --padim-weights D:/pretrained_weights/resnet18_a1_0-d63eafa0.pth --experiments p0_pretrained p3_consistency p4_consistency_mixstyle
```

| 출력 | 어디에 활용하는가? |
|---|---|
| `index.html` | 처음 보는 사람을 위한 안내, 실험/지표 설명, validation-selected 모델, matched pair 비교, FPR/recall 그래프, 설비별 성능 |
| `summary.csv` | 통합 metric 및 overall/equipment 상세 수치 |
| `selected.json`, `results.json` | 각 protocol의 선정 모델 및 상세 수치 |
| `config.json`, `split_audit.json` | 실제 적용 설정과 데이터 분할 감사 |
| `model.pt`, `threshold.json` | Backbone + Gaussian 통계 + validation postprocessor |
| `backbone_summary.json`, `history.json` | P1–P5 backbone 학습 diagnostics |
| `operating_points.json` | Validation 기준 raw threshold의 val/test FPR–Recall 변화 |
| `val/test/scores.csv` | 이미지별 label, lot, wafer, equipment, raw/normalized image score 및 예측 |
| `val/test/*_panel.png` | 원본 이미지 · 정규화 anomaly map · prediction mask (각각 native resolution), lot/wafer/equipment/GT 메타데이터와 image 예측 표시 |

수치 분석용 raw/normalized NPY map은 필요할 때 `--save-raw-maps` 옵션으로 추가 저장한다.

기존 결과에 대해 보고서만 갱신하려면 학습 없이 다음을 실행한다.

```powershell
python models/csam_void/build_report.py --root D:/csam_results/padim01
```

기존 JSON에 기록된 수치만 읽기 때문에 옛 결과에 없는 metric을 새로 계산하지는 않는다. 사내 image, checkpoint, mask, 설비 식별자, 실제 결과는 공개 GitHub에 올리지 않는다.

## 참고

- 실행 및 자세한 설정: [CSAM README](README.md)
- 구현: [run.py](run.py) · [padim_backbone.py](padim_backbone.py) · [run_padim.py](run_padim.py) · [build_report.py](build_report.py)
- 연구 참고: [PaDiM](https://arxiv.org/abs/2011.08785), [MixStyle](https://arxiv.org/abs/2104.02008)
- 이 코드는 논문 전체 재현이 아닌 PaDiM 기반 사내 적용 아이디어를 비교하기 위한 실험 구현이다. 실제 데이터에 대한 개선 여부는 별도 검증이 필요하다.
