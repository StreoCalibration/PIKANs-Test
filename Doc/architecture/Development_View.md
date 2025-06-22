# Development View - PIKANs 3파장 4-bucket 간섭계 높이 복원

## 구현 컴포넌트 및 인터페이스

- **데이터셋/Loader**
  - 벤치마크/시뮬레이션/실측 데이터셋 지원
  - PyTorch Dataset, DataLoader 등 활용

- **전처리 파이프라인**
  - OpenCV 기반 이미지 I/O 및 intensity normalization 등 구현

- **PIKANs 네트워크**
  - PyTorch 등 DL 프레임워크 기반
  - **Custom KAN 레이어**: B-spline 기반으로 구현 완료 (`src/model/layers.py`).
  - **Physics-Informed 손실함수**: 간섭계 물리식(`I = A + B*cos(phi)`)을 기반으로 구현 완료 (`src/loss/physics_informed_loss.py`).

- **Config 및 파라미터 관리**
  - 실험 단위 파라미터화, 재현성 확보
  - YAML/JSON 기반 설정 관리

- **훈련/평가 루프**
  - 자동화 스크립트, 로그 및 체크포인트 관리

> 각 컴포넌트는 모듈화 및 단위 테스트 가능 구조로 작성