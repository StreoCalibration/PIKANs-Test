# Scenario View (유스케이스) - PIKANs 간섭계 높이 복원

## 시나리오 1: 벤치마크 데이터 기반 학습 및 검증

1. 벤치마크 N-파장 M-bucket 이미지 준비 (e.g., `scripts/generate_4w_3b_data.py`로 4파장 3-bucket 데이터 생성).
2. 전처리/ROI/정규화 수행 (`src/data_loader/datasets.py`).
3. PIKANs 네트워크 학습 (설정 파일 기반, 물리식을 반영한 Physics-Informed loss 적용).
4. Height map 예측, GT와 비교 평가

## 시나리오 2: 실제 AOI 계측 데이터 복원

1. AOI 장비에서 실측 데이터 수집
2. 데이터 형식에 맞는 설정 파일 작성 후 동일 파이프라인 적용, 오프라인 복원
3. 결과: Height map, 위상 map, error map 산출

## 데이터셋/벤치마크

- **Raw 데이터**: N-파장 × M-bucket raw 이미지셋. 파일명 규칙은 설정 파일(`data.layout.file_pattern`)에서 정의 (e.g., `w{w_idx}_b{b_idx}.png`).
- **GT**: 픽셀별 height map. 현재 구현은 **NumPy (`.npy`)** 포맷을 사용.
- **데이터 generator**: 가상 데이터를 생성하는 시뮬레이터 구현 (`scripts/generate_4w_3b_data.py`).