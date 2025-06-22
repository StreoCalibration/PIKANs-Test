# PIKANs Test Project

본 레포지토리는 3파장 4-bucket 간섭계 데이터를 이용한 높이 복원 네트워크 "PIKANs"의 예시 구현을 위한 기본 구조를 제공합니다.

## 구조

- `pikan/` : 데이터셋, 전처리, 모델, 학습 루프 등 핵심 파이썬 모듈
- `Doc/architecture/` : 4+1 뷰 아키텍처 설계 문서

## 사용법

```bash
pip install -r requirements.txt
```

```python
from pathlib import Path
from pikan.config import TrainConfig
from pikan.train import train

config = TrainConfig(data_dir=Path("./data"))
train(config)
```
