# PIKANs: Physics-Informed KAN for Interferometry

3파장 4-bucket 간섭계 높이 복원을 위한 Physics-Informed KAN(Kolmogorov-Arnold Network) 프로젝트입니다.

## 프로젝트 구조

본 프로젝트는 4+1 뷰 아키텍처 모델을 기반으로 설계되었습니다. 자세한 내용은 `doc/architecture` 디렉토리를 참고하세요.

- `src/`: 핵심 소스 코드
- `configs/`: 실험 설정 파일
- `data/`: 원본 및 전처리된 데이터
- `outputs/`: 학습된 모델, 로그, 결과물
- `notebooks/`: 데이터 탐색 및 실험을 위한 주피터 노트북
- `tests/`: 단위 테스트 코드

## 설치

```bash
pip install -r requirements.txt
```

## 실행

### 1. 시뮬레이션 데이터(PNG) 학습

프로젝트에 포함된 기본 더미 데이터 생성 스크립트(`generate_4w_3b_data.py`)를 사용하여 PNG 형식의 이미지 세트를 생성하고 학습을 진행할 수 있습니다.

```bash
# (선택 사항) 더미 PNG 이미지 데이터 생성 (5개 샘플)
python scripts/generate_4w_3b_data.py --num_samples 5

# PNG 데이터용 설정 파일로 학습 실행
python train.py --config configs/4w3b_config.yaml
```

### 2. 실제 데이터(BMP) 학습

사용자의 실제 측정 데이터(.bmp 파일)를 사용하여 학습을 진행할 수 있습니다.

**가. 데이터 폴더 구조화**

실제 데이터는 여러 '세트(set)' 또는 '샘플(sample)'로 구성될 수 있습니다. 각 샘플 폴더는 아래와 같은 구조를 가져야 합니다.

```
data/
└── my_bmp_data/             <- 사용자 데이터의 루트 폴더 (이름은 자유롭게 지정 가능)
    ├── set_01/              <- 각 측정 세트(샘플) 폴더
    │   ├── raw/             <- 실제 측정 이미지(.bmp) 저장
    │   │   ├── w0_b0.bmp
    │   │   ├── w0_b1.bmp
    │   │   ├── w0_b2.bmp
    │   │   ├── w1_b0.bmp
    │   │   └── ... (총 12개 이미지: 4 파장 x 3 버킷)
    │   └── gt/              <- Ground Truth 데이터 저장
    │       └── height.npy   <- Ground Truth 높이 맵 (Numpy 배열)
    │
    └── set_02/
        └── ...
```

- **파일 명명 규칙**: `w{파장인덱스}_b{버킷인덱스}.bmp` 형식을 따라야 합니다. (인덱스는 0부터 시작)
- **`height.npy`**: Ground Truth 높이 맵은 `float32` 타입의 Numpy 배열로 저장되어야 합니다.

**나. 설정 파일 수정**

`configs/bmp_data_config.yaml` 파일을 열어 `data.path`를 위에서 생성한 사용자 데이터의 루트 폴더 경로로 수정합니다.

```yaml
# configs/bmp_data_config.yaml

data:
  # 이 부분을 실제 데이터가 있는 폴더 경로로 수정합니다.
  # 예: "data/my_bmp_data"
  path: "data/bmp_dummy_data"

  layout:
    # ...
    file_pattern: "w{w_idx}_b{b_idx}.bmp" # 파일 형식이 .bmp인지 확인
```

**다. BMP 데이터 학습 실행**

수정된 설정 파일을 사용하여 학습을 시작합니다.

```bash
python train.py --config configs/bmp_data_config.yaml
```

### 3. 추론

학습된 모델을 사용하여 새로운 데이터 샘플에 대한 높이 맵을 예측합니다.

```bash
# 형식: python predict.py --model_path <모델경로> --data_path <데이터샘플경로>
# 예시:
python predict.py --model_path outputs/models/20250822_212403/final_model.pth --data_path data/bmp_dummy_data/sample_000
```