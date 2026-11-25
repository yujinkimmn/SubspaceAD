# MVTec AD bottle: 작은 1/2/4-shot 실험

이 실험은 공식 SubspaceAD 구현의 **feature extraction → PCA fit → anomaly map
→ AUROC 평가** 경로를 실제 데이터에서 확인하기 위한 CPU 검증입니다.
논문의 전체 benchmark를 재현한 것은 아닙니다.

## 실행 방법

GitHub 저장소 Actions에서 **MVTec bottle few-shot CPU experiment**를 실행합니다.
결과 CSV는 **bottle-fewshot-metrics** artifact에서 받을 수 있습니다.

## 고정 조건

| 항목 | 설정 |
| --- | --- |
| 데이터 | MVTec AD bottle, 공개 [foersben/mvtec-ad](https://huggingface.co/datasets/foersben/mvtec-ad) 미러 |
| 데이터 라이선스 | CC BY-NC-SA 4.0, 비상업적 연구 용도만 허용 |
| Train | 정상 209장 pool에서 seed 42로 1/2/4장 선택 |
| Test | 정상 12장 + 불량 12장 (defect type별 균형), seed 42 |
| Backbone | facebook/dinov2-with-registers-small (frozen) |
| 입력 크기 | 224×224 |
| Feature | 마지막 3개 layer의 mean |
| PCA | explained variance 0.99 |
| Augmentation | 사용하지 않음 |
| 실행 환경 | GitHub-hosted Ubuntu CPU, Python 3.10 |

원본 논문은 다른 backbone 및 해상도를 사용하므로 여기서 측정되는 AUROC를
논문 결과와 직접 비교하면 안 됩니다. 테스트 24장만으로 산출한 수치는
불확실성이 크며, shot 설정별 차이도 seed와 정상 샘플 구성에 민감합니다.

## 결과 해석

- `dataset_manifest.json`: 미러 commit, train pool 크기, 시험 이미지 목록
- `bottle_fewshot_summary.csv`: 1/2/4-shot의 Image AUROC, Pixel AUROC, AU-PRO 등
- F1: validation 데이터가 없는 조건에서 threshold를 보정하지 않으므로
  `NaN`으로 기록되는 것이 정상입니다.

**데이터 보호:** MVTec 이미지와 pretrained 모델은 Actions 실행 중
runner에서만 다운로드합니다. 이미지와 모델 가중치는 GitHub 저장소나
결과 artifact에 업로드하지 않습니다.

사내 wafer 데이터는 이 공개 저장소에 넣지 마세요.
