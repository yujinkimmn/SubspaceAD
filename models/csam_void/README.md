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

출력 경로의 index.html을 브라우저에서 연다. pooled 및 각 holdout report.html, summary.csv, results.json, selected.json, split_audit.json, config.json을 생성한다. 모델별 model.pt, threshold.json, history.json(CNN), val/test scores.csv, raw NPY, mask PNG, overlay PNG를 저장한다. overlay 빨강=예측, 초록=정답, 노랑=겹침. 결과에는 원본 경로 및 이미지 파생물이 있으니 사내 자료로 취급한다.

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
