# Process View - PIKANs 3파장 4-bucket 간섭계 높이 복원

## 데이터 처리 흐름

1. **이미지 입력/로드**
2. **전처리** (정규화, ROI, 필터링 등)
3. **특징 추출/벡터화**
4. **PIKANs 입력/추론**
5. **Loss 계산 및 역전파**: MSE Loss와 Physics-Informed Loss를 결합하여 역전파 수행. (현재 Physics-Informed Loss는 플레이스홀더)
6. **Height map 복원 및 저장/시각화**

## 병렬 처리 및 확장성

- 배치별/ROI별 병렬 처리
- 멀티스레드/멀티GPU 지원 가능
- 각 프로세스별 독립 실행 및 디버깅 가능

> 각 처리 단계는 로그 및 에러 처리 체계 포함