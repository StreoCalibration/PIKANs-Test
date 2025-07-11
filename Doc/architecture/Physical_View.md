# Physical View - PIKANs 간섭계 높이 복원

## 물리적 배치 및 환경

- **실행 환경**
  - 서버, 워크스테이션, 로컬PC 모두 대응
  - GPU 1개 이상, RAM 16GB+ 권장

- **파일/데이터 관리**
  - **`configs/`**: `*.yaml` 설정 파일 관리.
  - **`data/`**: 원본 데이터셋 저장. 하위 디렉토리 구조는 `data/<dataset_name>/<sample_name>/{raw/*.png, gt/*.npy}` 형태를 따름.
  - **`outputs/`**: 학습 결과(로그, 모델) 및 추론 결과(예측 이미지) 저장. 타임스탬프 기반으로 생성.
  - **`scripts/`**: 데이터 생성 등 유틸리티 스크립트 관리 (`generate_4w_3b_data.py` 등).
  - **`src/`**: 핵심 소스 코드.

- **Docker 컨테이너**
  - 환경 재현성, 배포 용이성 지원(옵션)

## 운영/확장

- 자원 사용 모니터링 및 장애 복구 계획
- 장비 확장/서버 이전 용이