# SubspaceAD 실험 환경

이 저장소는 [TUE-AIMS/SubspaceAD](https://github.com/TUE-AIMS/SubspaceAD)의
공식 코드를 기반으로 합니다. 초기 가져온 upstream 커밋은
`ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7`이며 라이선스는 Apache-2.0입니다.

## 1. 로컬 또는 GPU 서버 준비

공식 README의 Python 3.10 환경을 따릅니다. GitHub Actions의 Static smoke test는
대규모 의존성이나 모델을 설치하지 않지만, 별도 MVTec few-shot 실험 workflow는
CPU용 PyTorch와 공개 DINOv2 모델을 내려받아 실제 inference를 실행합니다.

`requirements.txt`에는 현재 로컬 RTX 4050 Laptop GPU(드라이버 576.80)에 맞춰
**torch 2.11.0+cu128 / torchvision 0.26.0+cu128**을 고정했습니다.
`--extra-index-url`을 사용하므로 일반 패키지는 PyPI에서 받고, 해당 CUDA wheel은
공식 PyTorch 저장소에서 받습니다. `nvidia-smi`에 CUDA 12.9가 표시되더라도
드라이버가 CUDA 12.8 빌드를 실행할 수 있습니다.

이 요구사항은 NVIDIA GPU용입니다. CPU 전용 환경이나 다른 CUDA 환경에서는
[PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)를 확인해
`torch`, `torchvision`, 추가 인덱스 설정을 교체하세요.
GitHub Actions의 공개 MVTec few-shot 실험은 별도 CPU용 버전(2.5.1/0.20.1)을
설치하므로 이 로컬 GPU 설정의 영향을 받지 않습니다.

```bash
conda create -n subspacead python=3.10 -y
conda activate subspacead
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt`는 CUDA 빌드의 PyTorch/torchvision 및 직접 사용하는 패키지를 설치하며,
공식 코드에서 import하지 않는 anomalib/FrEIA는 포함하지 않습니다. 일부 패키지는 버전 범위만 제한되어 있으므로
실험 시 실제 설치 버전을 기록하세요.

## 2. 공식 예제로 실행

공식 자료 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)는
이 저장소에 넣지 않고 별도의 데이터 경로에서 관리합니다.

```bash
python main.py \
  --dataset_name mvtec_ad \
  --dataset_path /path/to/mvtec-ad \
  --categories bottle \
  --model_ckpt facebook/dinov2-with-registers-giant \
  --image_res 672 \
  --k_shot 1 \
  --aug_count 30 \
  --pca_ev 0.99 \
  --outdir results/bottle_1shot
```

첫 실행은 Hugging Face에서 `--model_ckpt`의 pretrained 모델을 내려받을
수 있습니다. 인터넷 연결이 제한된 서버에서는 승인된 내부 모델 경로로 교체하세요.

## 3. 실험 범위 및 데이터 보안

- GitHub Actions의 `Static smoke test`는 syntax, 패키징, import만 검사합니다.
  모델 inference, 논문 metric 재현, GPU 호환성까지 검증하는 것은 아닙니다.
- `datasets/`, `results/`, `logs/`, `checkpoints/`와 대표적인 모델 파일 확장자는
  `.gitignore`에 추가되어 있습니다.
- 사내 wafer 이미지, lot ID, 경로, 보고서 및 내부 모델 가중치는 이 공개 저장소에
  업로드하지 마세요. 데이터는 승인된 사내 환경에만 둡니다.
- 정상 데이터의 noise 특성이 매우 다양하다면 단일 1-shot 결과만 판단하지 말고,
  정상 샘플 수(1/2/4/full-shot)와 lot-holdout 조건을 바꿔 비교하는 것이 좋습니다.

## 4. 권장 첫 검증

1. `Static smoke test` Action 통과를 확인합니다.
2. 소규모 공개 MVTec AD 1개 카테고리에서 공식 예제를 실행합니다.
3. 정상 이미지 1/2/4장 설정을 바꿔 anomaly map과 image-level score를 비교합니다.
4. 데이터 구조가 다른 사내 wafer 이미지의 적용은 별도 loader 및 평가 설계 이후 진행합니다.
