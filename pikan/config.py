from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-3
    device: str = "cuda"
