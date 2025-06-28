# Logical View - PIKANs 간섭계 높이 복원

## 주요 모듈 구조

1. **데이터 획득 모듈**
   - N-파장, M-bucket 간섭 영상 획득 (파장/버킷 수는 설정 파일에서 정의)
   - 벤치마크/시뮬/실측 데이터 대응

2. **전처리 모듈**
   - Intensity normalization
   - 이미지 alignment 및 ROI 선정 (`src/preprocessing/filters.py` 등)
   - 이상치/노이즈 필터링

3. **특징 추출 및 벡터화**
   - 각 픽셀의 (N x M)-channel intensity 값을 1차원 벡터로 변환하여 네트워크 입력으로 사용 (`src/data_loader/datasets.py`)

4. **PIKANs 네트워크 모듈**
   - **KAN 기반 신경망**: B-spline을 활성화 함수로 사용하는 `KANLayer`를 여러 층으로 쌓아 구성 (`src/model/pikans.py`)
   - **Physics-Informed Loss**: MSE Loss와 물리 제약 조건(I = A + B*cos(phi)) 기반 Loss를 결합하는 구조.

5. **후처리/시각화**
   - Height map 생성
   - 위상 unwrapping, error map, 정보가 포함된 이미지(컬러바, 제목)로 시각화/저장 (`src/utils/visualization.py`)

> 각 모듈은 독립적으로 테스트 및 교체 가능하도록 설계