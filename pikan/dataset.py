import os
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
import cv2


class InterferogramDataset(Dataset):
    """3파장 4-bucket 간섭계 이미지를 로드하는 Dataset."""

    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        path = self.image_paths[idx]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"{path}를 열 수 없습니다.")
        return image.astype(np.float32), os.path.basename(path)
