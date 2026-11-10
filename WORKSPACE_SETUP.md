# SubspaceAD 실험 환경

이 저장소는 [TUE-AIMS/SubspaceAD](https://github.com/TUE-AIMS/SubspaceAD)의
공식 코드를 기반으로 합니다. 초기 가져온 upstream 커밋은
`ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7`이며 라이선스는 Apache-2.0입니다.

## 1. 로컬 또는 GPU 서버 준비

공식 README의 Python 3.10 환경을 따릅니다. 대규모 dependency 설치 및 DINOv2
가중치 다운로드는 GitHub Actions에서 수행하지 않습니다.

```bash
conda create -n subspacead python=3.10 -y
conda activate subspacead
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt`의 버전이 고정되어 있지 않으므로 첫 실행에서 실제 설치 버전을
기록하고, GPU 서버의 CUDA/PyTorch 조합이 맞는지 확인하세요.

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
