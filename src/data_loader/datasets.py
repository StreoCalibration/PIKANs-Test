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
    def __init__(self, data_dir, data_layout, roi=None, normalization_method='minmax'):
        self.data_dir = data_dir
        # data_layout example: {'num_wavelengths': 4, 'num_buckets': 3, 'file_pattern': 'w{w_idx}_b{b_idx}.png'}
        self.data_layout = data_layout
        self.roi = roi
        self.normalization_method = normalization_method
        self.num_wavelengths = self.data_layout['num_wavelengths']
        self.num_buckets = self.data_layout['num_buckets']

        self.pixel_intensities = []
        self.pixel_heights = []

        self._load_data()

    def _load_data(self):
        print("Loading and vectorizing dataset. This may take a while...")
        # 가정: data_dir는 'sample_001', 'sample_002'와 같은 각 샘플의 하위 디렉토리를 포함
        sample_dirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

        for sample_name in tqdm(sample_dirs, desc="Processing Samples"):
            sample_path = os.path.join(self.data_dir, sample_name)
            
            # GT 파일 탐색. 각 샘플 폴더 내의 'gt/height.npy'로 가정
            gt_path = os.path.join(sample_path, "gt", "height.npy")
            if not os.path.exists(gt_path):
                print(f"Warning: Ground truth for sample {sample_name} not found at '{gt_path}'. Skipping.")
                continue

            # Intensity 이미지 로드 (data_layout에 따라 동적으로)
            intensities = []
            valid_sample = True
            raw_data_path = os.path.join(sample_path, "raw")
            file_pattern = self.data_layout['file_pattern']

            for w in range(self.num_wavelengths):
                for b in range(self.num_buckets):
                    img_filename = file_pattern.format(w_idx=w, b_idx=b)
                    img_path = os.path.join(raw_data_path, img_filename)
                    if not os.path.exists(img_path):
                        print(f"Warning: Image {img_path} not found for sample {sample_name}. Skipping sample.")
                        valid_sample = False
                        break
                    
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # PNG 포맷으로 가정
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
            num_channels = self.num_wavelengths * self.num_buckets
            intensity_stack = np.stack(intensities, axis=-1) # (H, W, num_channels)
            num_pixels = intensity_stack.shape[0] * intensity_stack.shape[1]
            intensity_vectors = intensity_stack.reshape(num_pixels, num_channels)
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

def load_inference_data(data_path, data_layout, normalization_method='minmax', roi=None):
    """
    Loads and preprocesses data for a single inference run.
    This function is for predicting a full height map, not for training.
    """
    print(f"Loading inference data from {data_path}")
    raw_data_path = os.path.join(data_path, "raw")
    if not os.path.isdir(raw_data_path):
        raise FileNotFoundError(f"Raw data directory 'raw' not found in: {data_path}")

    intensities = []
    original_shape = None

    num_wavelengths = data_layout['num_wavelengths']
    num_buckets = data_layout['num_buckets']
    file_pattern = data_layout['file_pattern']

    for w in range(num_wavelengths):
        for b in range(num_buckets):
            img_filename = file_pattern.format(w_idx=w, b_idx=b)
            img_path = os.path.join(raw_data_path, img_filename)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Required image file not found: {img_path}")

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Could not read image: {img_path}")

            if original_shape is None:
                original_shape = img.shape
            elif original_shape != img.shape:
                raise ValueError("All intensity images must have the same dimensions.")

            intensities.append(img)

    # Preprocessing
    if roi:
        intensities = [select_roi(img, roi) for img in intensities]
        # Update shape after ROI
        original_shape = intensities[0].shape

    if normalization_method:
        intensities = [normalize_intensity(img.astype(np.float32), normalization_method) for img in intensities]

    # Stack and vectorize
    num_channels = num_wavelengths * num_buckets
    intensity_stack = np.stack(intensities, axis=-1)  # (H, W, num_channels)
    intensity_vectors = intensity_stack.reshape(-1, num_channels)

    return torch.from_numpy(intensity_vectors).float(), original_shape


class RealDataFinetuneDataset(Dataset):
    """
    PyTorch Dataset for loading real interferometry data for unsupervised fine-tuning.
    This dataset is similar to InterferometryDataset but does NOT load ground truth (GT) data.
    It only loads the intensity images.
    """
    def __init__(self, data_dir, data_layout, roi=None, normalization_method='minmax'):
        self.data_dir = data_dir
        self.data_layout = data_layout
        self.roi = roi
        self.normalization_method = normalization_method
        self.num_wavelengths = self.data_layout['num_wavelengths']
        self.num_buckets = self.data_layout['num_buckets']

        self.pixel_intensities = []
        self._load_data()

    def _load_data(self):
        print("Loading and vectorizing real dataset for fine-tuning (no GT)...")
        sample_dirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

        for sample_name in tqdm(sample_dirs, desc="Processing Samples"):
            sample_path = os.path.join(self.data_dir, sample_name)

            # This dataset does NOT look for GT files.
            # It only loads the raw intensity images.
            intensities = []
            valid_sample = True
            raw_data_path = os.path.join(sample_path, "raw")
            file_pattern = self.data_layout['file_pattern']

            for w in range(self.num_wavelengths):
                for b in range(self.num_buckets):
                    img_filename = file_pattern.format(w_idx=w, b_idx=b)
                    img_path = os.path.join(raw_data_path, img_filename)
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

            # --- Preprocessing ---
            if self.roi:
                intensities = [select_roi(img, self.roi) for img in intensities]

            if self.normalization_method:
                intensities = [normalize_intensity(img.astype(np.float32), self.normalization_method) for img in intensities]

            # --- Stack and Vectorize ---
            num_channels = self.num_wavelengths * self.num_buckets
            intensity_stack = np.stack(intensities, axis=-1)
            num_pixels = intensity_stack.shape[0] * intensity_stack.shape[1]
            intensity_vectors = intensity_stack.reshape(num_pixels, num_channels)

            self.pixel_intensities.append(intensity_vectors)

        if not self.pixel_intensities:
            raise RuntimeError("No valid data was loaded. Check data paths and file structure.")

        self.pixel_intensities = np.concatenate(self.pixel_intensities, axis=0)

        print(f"Dataset loaded successfully. Total pixels: {len(self.pixel_intensities)}")

    def __len__(self):
        return len(self.pixel_intensities)

    def __getitem__(self, idx):
        # Returns only the intensity vector, no ground truth.
        intensity_vector = self.pixel_intensities[idx]
        return torch.from_numpy(intensity_vector).float()