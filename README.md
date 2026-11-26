# Anomaly Detection Lab

| 모델 | 실행 및 환경 |
|---|---|
| SubspaceAD | [PCA 기반 few-shot 모델](models/subspacead/README.md) |
| CSAM Void Segmentation | [Exp0–7 자동 실험, 설비 holdout 및 HTML 결과 비교](models/csam_void/README.md) |
| NAGL | [정상·불량 참고 이미지를 이용한 학습 및 설비별 평가](models/nagl/README.md) |

SubspaceAD의 소스, 벤치마크 스크립트, 데이터 준비 도구와 테스트는 모두 `models/subspacead/`에 있습니다. 기존 환경에서는 루트의 `python main.py`도 사용할 수 있습니다.

각 모델은 별도 환경과 `requirements.txt`를 사용합니다. 루트의 `requirements.txt`는 SubspaceAD용입니다. CSAM/PaDiM 실험은 적합한 PyTorch/CUDA 환경에서 `python -m pip install -r models/csam_void/requirements.txt`로 설치하고 `python -m pip check`로 의존성 충돌을 확인하세요. NAGL은 [해당 모델의 설치 안내](models/nagl/README.md)를 따르세요.

## PaDiM pretrained checkpoint

- Backbone: timm `resnet18.a1_in1k` (Exp0 / P0–P5)
- [공식 checkpoint 다운로드 (`resnet18_a1_0-d63eafa0.pth`)](https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth)
- [공식 모델 정보](https://huggingface.co/timm/resnet18.a1_in1k) · [설치 및 오프라인 로드 방법](models/csam_void/README.md#pretrained-weight-사전-다운로드-폐쇄망--오프라인-실행)
