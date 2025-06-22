import cv2
import numpy as np

def normalize_intensity(image, method="minmax"):
    """
    Normalizes image intensity.
    As per Logical View, this is a key preprocessing step.
    """
    image = image.astype(np.float32)
    if method == "minmax":
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image
    elif method == "standard":
        mean, std = np.mean(image), np.std(image)
        if std > 1e-6: # 0으로 나누는 것을 방지
            return (image - mean) / std
        return image - mean
    return image

def select_roi(image, roi):
    """
    Selects a Region of Interest (ROI) from an image.
    """
    if roi is None:
        return image
    x, y, w, h = roi
    return image[y:y+h, x:x+w]