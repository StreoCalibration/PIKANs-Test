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

### 학습

```bash
python train.py --config configs/default_config.yaml
```

### 추론

```bash
python predict.py --model_path outputs/models/20250622_203136/final_model.pth --data_path data/raw/benchmark/sample_005
```