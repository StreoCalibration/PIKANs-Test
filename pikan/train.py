from pathlib import Path
from torch.utils.data import DataLoader
import torch

from .dataset import InterferogramDataset
from .preprocessing import normalize_intensity
from .model import PIKANsNetwork, physics_informed_loss
from .config import TrainConfig


def load_dataset(data_dir: Path) -> InterferogramDataset:
    image_paths = sorted(str(p) for p in data_dir.glob("*.png"))
    return InterferogramDataset(image_paths)


def train(config: TrainConfig) -> None:
    dataset = load_dataset(config.data_dir)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = PIKANsNetwork().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        for images, _ in loader:
            images = normalize_intensity(images.numpy())
            images = torch.from_numpy(images).unsqueeze(1).to(device)
            # 타겟은 예시로 0 사용
            target = torch.zeros_like(images)
            pred = model(images)
            loss = physics_informed_loss(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
