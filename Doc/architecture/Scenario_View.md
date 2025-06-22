# Scenario View (유스케이스) - PIKANs 3파장 4-bucket 간섭계 높이 복원

## 시나리오 1: 벤치마크 데이터 기반 학습 및 검증

1. 벤치마크 3파장 4-bucket 이미지 준비
2. 전처리/ROI/정규화 수행 (`src/data_loader/datasets.py`)
3. PIKANs 네트워크 학습 (현재 Physics-Informed loss는 플레이스홀더로 동작)
4. Height map 예측, GT와 비교 평가

## 시나리오 2: 실제 AOI 계측 데이터 복원

1. AOI 장비에서 실측 데이터 수집
2. 동일 파이프라인 적용, 실시간/오프라인 복원
3. 결과: Height map, 위상 map, error map 산출

## 데이터셋/벤치마크

- 3파장(λ1,λ2,λ3) × 4-bucket raw 이미지셋
- **GT**: 픽셀별 height map. 현재 구현은 **NumPy (`.npy`)** 포맷을 사용.
- **데이터 generator**: 가상 데이터를 생성하는 시뮬레이터 구현 완료 (`scripts/generate_dummy_data.py`)