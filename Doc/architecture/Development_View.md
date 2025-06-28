# Development View - PIKANs 간섭계 높이 복원

## 구현 컴포넌트 및 인터페이스

- **데이터셋/Loader**
  - 벤치마크/시뮬레이션/실측 데이터셋 지원
  - PyTorch Dataset, DataLoader 등 활용

- **전처리 파이프라인**
  - OpenCV 기반 이미지 I/O 및 intensity normalization 등 구현

- **PIKANs 네트워크**
  - PyTorch 등 DL 프레임워크 기반
  - **Custom KAN 레이어**: B-spline 기반으로 구현 (`src/model/layers.py`).
  - **Physics-Informed 손실함수**: 간섭계 물리식(`I = A + B*cos(phi)`)을 기반으로 구현. 사용 파장은 설정 파일에서 주입 (`src/loss/physics_informed_loss.py`).

- **Config 및 파라미터 관리**
  - 실험 단위 파라미터화 및 재현성 확보
  - YAML 기반 설정 관리. 데이터 경로, 레이아웃, 모델 하이퍼파라미터 등을 모두 포함.

- **훈련/평가 루프**
  - 자동화 스크립트, 로그 및 체크포인트 관리

> 각 컴포넌트는 모듈화 및 단위 테스트 가능 구조로 작성