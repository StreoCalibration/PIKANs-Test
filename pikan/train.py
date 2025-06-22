from pathlib import Path
from torch.utils.data import DataLoader
import torch
import argparse

# 절대 임포트를 사용해 스크립트 단독 실행 시에도 패키지가 정상 인식되도록 한다.
from pikan.dataset import InterferogramDataset
from pikan.preprocessing import normalize_intensity
from pikan.model import PIKANsNetwork, physics_informed_loss
from pikan.config import TrainConfig


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


def main() -> None:
    """커맨드 라인에서 실행할 수 있도록 기본 진입점을 제공한다."""
    parser = argparse.ArgumentParser(description="Train PIKANs model")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="학습 이미지 폴더")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에폭 수")
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--device", type=str, default="cuda", help="사용할 디바이스")
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()
