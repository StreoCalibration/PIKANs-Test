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


## Sim-to-Real 하이브리드 학습 (고급)

정확한 Ground Truth(GT) 데이터 없이 실제 BGA 샘플 데이터만으로 모델을 학습 및 미세조정하는 2단계 하이브리드 전략입니다.

### 개요

1.  **1단계 (사전학습)**: 컴퓨터 시뮬레이션으로 생성한 **합성 데이터**를 사용하여 모델이 BGA 볼의 형상과 간섭계의 물리 법칙에 대한 기본적인 이해를 갖도록 학습시킵니다.
2.  **2단계 (미세조정)**: 사전학습된 모델을 **실제 측정 데이터**에 적응시킵니다. 이 과정에서 모델은 정답 없이 물리 법칙만을 기반으로 스스로 학습하여 실제 센서의 노이즈나 왜곡 등 현실과의 차이를 보정합니다.

### 1단계: 합성 데이터로 사전학습

**가. 합성 BGA 데이터 생성**

먼저, 모델 사전학습에 사용할 가상의 BGA 간섭 무늬 데이터를 생성합니다.

-   **스크립트**: `scripts/generate_bga_synthetic_data.py`
-   **설명**: 이 스크립트는 BGA의 물리적 사양(볼 직경, 높이 등)에 맞춰 이상적인 높이 맵(.npy)과 해당 간섭 무늬 이미지(.png)를 생성합니다. 현재 시스템 환경에 맞춰 **4-파장, 3-버킷**을 기본 설정으로 사용합니다.
-   **실행 명령어**:

    ```bash
    # data/bga_synthetic 폴더에 10개의 샘플 데이터를 생성합니다.
    python scripts/generate_bga_synthetic_data.py --num-samples 10
    ```

**나. 사전학습 실행**

생성된 합성 데이터로 모델을 사전학습합니다.

-   **스크립트**: `train.py`
-   **설정 파일**: `configs/pretrain_bga_config.yaml`
-   **실행 명령어**:

    ```bash
    python train.py --config configs/pretrain_bga_config.yaml
    ```

-   **결과**: 학습이 완료되면 `outputs/models/YYYYMMDD_HHMMSS/` 와 같은 경로에 사전학습된 모델(`final_model.pth`)이 저장됩니다. **이 모델 파일의 경로는 2단계에서 필요하므로 잘 기록해 두어야 합니다.**

### 2단계: 실제 데이터로 미세조정

**가. 실제 데이터 준비**

미세조정에 사용할 실제 BGA 측정 데이터를 `README.md`의 "실제 데이터(BMP) 학습" 섹션에 설명된 구조에 따라 배치합니다. **이 과정에서는 GT(`height.npy`) 파일이 필요하지 않습니다.**

**나. 미세조정 설정 파일 수정**

`configs/finetune_bga_config.yaml` 파일을 열어 다음 두 경로를 수정합니다.

1.  `data.path`: 실제 데이터가 있는 폴더 경로로 수정합니다. (예: `"data/my_real_bga_data"`)
2.  `training.pretrained_model_path`: **1단계에서 저장된 사전학습 모델의 정확한 경로**로 수정합니다. (예: `"outputs/models/20250823_072229/final_model.pth"`)

**다. 미세조정 실행**

수정된 설정 파일을 사용하여 미세조정을 시작합니다.

-   **스크립트**: `finetune.py`
-   **설명**: 이 스크립트는 정답(GT) 데이터 없이, 오직 물리 법칙(Physics-Informed Loss)만을 사용하여 사전학습된 모델을 실제 데이터의 특성에 맞게 세밀하게 조정합니다.
-   **실행 명령어**:

    ```bash
    python finetune.py --config configs/finetune_bga_config.yaml
    ```

-   **결과**: 학습이 완료되면 `outputs/models/finetune_YYYYMMDD_HHMMSS/` 경로에 실제 데이터에 완벽하게 적응된 최종 모델(`final_bga_model.pth`)이 저장됩니다. 이 모델을 `predict.py`와 함께 사용하여 최종 높이 맵을 추론할 수 있습니다.
```