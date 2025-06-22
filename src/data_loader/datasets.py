import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from tqdm import tqdm
from ..preprocessing.filters import select_roi, normalize_intensity

class InterferometryDataset(Dataset):
    """
    PyTorch Dataset for loading 3-wavelength, 4-bucket interferometry images.
    As per Development View, this class handles benchmark, simulation, or real data.

    This implementation pre-loads all pixel data into memory.
    For each pixel, it creates a vector of 12 intensity values (3 wavelengths * 4 buckets)
    and its corresponding ground truth height.
    """
    def __init__(self, data_dir, gt_dir, roi=None, normalization_method='minmax'):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.roi = roi
        self.normalization_method = normalization_method

        self.pixel_intensities = []
        self.pixel_heights = []

        self._load_data()

    def _load_data(self):
        print("Loading and vectorizing dataset. This may take a while...")
        # 가정: data_dir는 'sample_001', 'sample_002'와 같은 각 샘플의 하위 디렉토리를 포함
        sample_dirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

        for sample_name in tqdm(sample_dirs, desc="Processing Samples"):
            sample_path = os.path.join(self.data_dir, sample_name)
            
            # GT 파일 탐색. '.npy' 형식으로 가정 (예: 'sample_001_gt.npy')
            gt_path = os.path.join(self.gt_dir, f"{sample_name}_gt.npy")
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth for sample {sample_name} not found at '{gt_path}'. Skipping.")
                continue

            # 12개의 intensity 이미지 로드 (L1_I1 to L3_I4)
            intensities = []
            valid_sample = True
            for w in range(1, 4): # 파장 1, 2, 3
                for b in range(1, 5): # Bucket 1, 2, 3, 4
                    img_path = os.path.join(sample_path, f"L{w}_I{b}.png") # PNG 포맷으로 가정
                    if not os.path.exists(img_path):
                        print(f"Warning: Image {img_path} not found for sample {sample_name}. Skipping sample.")
                        valid_sample = False
                        break
                    
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}. Skipping sample.")
                        valid_sample = False
                        break
                    
                    intensities.append(img)
                if not valid_sample:
                    break
            
            if not valid_sample:
                continue

            # Ground truth 높이 맵 로드
            gt_img = np.load(gt_path)
            if gt_img is None:
                print(f"Warning: Could not read GT numpy file {gt_path}. Skipping sample.")
                continue

            # --- 전처리 ---
            if self.roi:
                intensities = [select_roi(img, self.roi) for img in intensities]
                gt_img = select_roi(gt_img, self.roi)

            if self.normalization_method:
                intensities = [normalize_intensity(img.astype(np.float32), self.normalization_method) for img in intensities]

            # --- 스택 및 벡터화 ---
            intensity_stack = np.stack(intensities, axis=-1) # (H, W, 12)
            num_pixels = intensity_stack.shape[0] * intensity_stack.shape[1]
            intensity_vectors = intensity_stack.reshape(num_pixels, 12)
            gt_vector = gt_img.reshape(num_pixels, 1)

            self.pixel_intensities.append(intensity_vectors)
            self.pixel_heights.append(gt_vector)

        if not self.pixel_intensities:
            raise RuntimeError("No valid data was loaded. Check data paths and file structure.")

        self.pixel_intensities = np.concatenate(self.pixel_intensities, axis=0)
        self.pixel_heights = np.concatenate(self.pixel_heights, axis=0)

        print(f"Dataset loaded successfully. Total pixels: {len(self.pixel_intensities)}")

    def __len__(self):
        return len(self.pixel_intensities)

    def __getitem__(self, idx):
        intensity_vector = self.pixel_intensities[idx]
        height = self.pixel_heights[idx]
        return torch.from_numpy(intensity_vector).float(), torch.from_numpy(height).float()

def load_inference_data(data_path):
    """
    Loads and preprocesses data for a single inference run.
    This function is for predicting a full height map, not for training.
    """
    print(f"Loading inference data from {data_path}")
    # TODO: Implement data loading for prediction
    # This will likely involve loading 12 images, normalizing,
    # and stacking them into a tensor of shape (H*W, 12).
    return None # Placeholder