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

패키지를 모듈로 실행하면 상대 임포트 문제가 발생하지 않습니다.

```bash
python -m pikan.train   # 또는 python pikan/train.py
```

## VSCode에서 실행하기

레포지토리 루트에는 VSCode용 `launch.json`이 포함되어 있습니다. VSCode에서
프로젝트 폴더를 열고 F5(또는 "실행" 버튼)을 누르면 `pikan.train` 모듈이 자동으로
실행됩니다. 데이터 경로 등 파라미터는 `launch.json`이나 커맨드 라인 인수로
수정할 수 있습니다.
