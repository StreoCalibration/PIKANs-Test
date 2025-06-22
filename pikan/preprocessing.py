import cv2
import numpy as np


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """이미지 intensity를 0~1 범위로 정규화."""
    image = image.astype(np.float32)
    min_val, max_val = image.min(), image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def apply_roi(image: np.ndarray, roi: tuple) -> np.ndarray:
    """주어진 ROI를 잘라낸다."""
    x, y, w, h = roi
    return image[y:y + h, x:x + w]


def filter_noise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """간단한 가우시안 블러를 적용한다."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)
