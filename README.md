# Anomaly Detection Lab

| 모델 | 실행 및 환경 |
|---|---|
| SubspaceAD | [PCA 기반 few-shot 모델](models/subspacead/README.md) |
| CSAM Void Segmentation | [Exp0–7 자동 실험, 설비 holdout 및 HTML 결과 비교](models/csam_void/README.md) |
| NAGL | [정상·불량 참고 이미지를 이용한 학습 및 설비별 평가](models/nagl/README.md) |

SubspaceAD의 소스, 벤치마크 스크립트, 데이터 준비 도구와 테스트는 모두 `models/subspacead/`에 있습니다. 기존 환경에서는 루트의 `python main.py`도 사용할 수 있습니다.
