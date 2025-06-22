# Logical View - PIKANs 3파장 4-bucket 간섭계 높이 복원

## 주요 모듈 구조

1. **데이터 획득 모듈**
   - 3파장, 각 파장별 4-bucket 간섭 영상 획득
   - 벤치마크/시뮬/실측 데이터 대응

2. **전처리 모듈**
   - Intensity normalization
   - 이미지 alignment 및 ROI 선정
   - 이상치/노이즈 필터링

3. **특징 추출 및 벡터화**
   - (x, y, λ1~3 각 파장별 4-bucket intensity) → 네트워크 입력 포맷 변환

4. **PIKANs 네트워크 모듈**
   - KAN 기반 신경망
   - Physics-Informed Loss 설계

5. **후처리/시각화**
   - Height map 생성
   - 위상 unwrapping, error map, 시각화/저장

> 각 모듈은 독립적으로 테스트 및 교체 가능하도록 설계