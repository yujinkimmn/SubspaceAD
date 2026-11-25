# 웨이퍼 이미지 NAGL 실험

공식 [NAGL](https://github.com/JasonKyng/NAGL/tree/5466edf5f099364addd5c84ab30dc7ced653d4f4) 코드는 `upstream/`에 있습니다. 이 폴더의 `run.py`는 단일 장치에서 DINOv2-S/14를 고정하고 NAGL head를 학습합니다. RGB 또는 grayscale 8-bit 이미지를 letterbox로 입력하고 원본 크기로 anomaly map을 돌려줍니다. 기본 크기는 224입니다.

## RTX 4050에서 실행

저장소 루트에서 다음을 실행하세요. 최초 실행은 DINOv2 소스와 가중치를 내려받습니다.

```powershell
conda create -n nagl python=3.11 -y
conda activate nagl
python -m pip install -r models/nagl/requirements-cu128.txt
python experiments/nagl/make_demo.py --out datasets/nagl_demo
python models/nagl/run.py check --manifest datasets/nagl_demo/manifest.csv --by-equipment
python models/nagl/run.py train --manifest datasets/nagl_demo/manifest.csv --by-equipment --epochs 2 --out results/nagl_demo
python models/nagl/run.py eval --manifest datasets/nagl_demo/manifest.csv --checkpoint results/nagl_demo/best.pt --out results/nagl_demo/test
```

CPU에서는 torch==2.7.1과 torchvision==0.22.1을 PyTorch CPU index로 설치한 뒤 `pip install -r models/nagl/requirements.txt`를 실행합니다.

## 실데이터 준비

합성 데모의 manifest.csv를 예시로 사용하세요. 각 행은 `image,mask,label,split,equipment,lot` 순서입니다. 경로는 CSV 기준입니다. label은 정상 0, 불량 1입니다. ref(고정 참고 세트)와 train의 불량에는 원본과 같은 크기의 결함 마스크가 필요합니다. 배경은 0, 결함은 양수입니다. val과 test는 이미지 단위 라벨만 있으면 됩니다. 동일 root lot과 같은 픽셀의 이미지를 split 간 분리하세요. 통과시킬 FPN에는 정상 라벨을 부여하세요.

`--by-equipment`는 설비마다 고정 참고 이미지를 선택합니다. 기본은 설비당 정상·불량 각 1장입니다. 공통 참고 세트는 옵션을 생략합니다. `--size 448`은 train 시 지정하며 eval은 체크포인트 설정을 따릅니다. val에서 목표 recall(기본 0.95)을 만족하는 threshold를 고정하고 test에서 설비별 결과를 확인합니다. `scores.csv`는 이미지별 판정, `metrics.json`은 FP/FN·실제 recall·정상 1,000장당 오검출, `maps/*.npy`는 원본 크기의 점수 배열입니다. best.pt에는 참고 이미지 텐서가 들어 있어 원본 데이터와 같은 기준으로 보관해야 합니다.
