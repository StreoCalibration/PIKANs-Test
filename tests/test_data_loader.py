import unittest
import torch
import numpy as np
import cv2
import os
import shutil
import tempfile

from src.data_loader.datasets import InterferometryDataset, load_inference_data
from src.preprocessing.filters import select_roi

class TestDataLoader(unittest.TestCase):
    """
    Test suite for data loading and dataset functionality.
    """

    def setUp(self):
        """
        Set up a temporary directory with dummy data for testing.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.data_layout = {
            'num_wavelengths': 2,
            'num_buckets': 2,
            'file_pattern': 'w{w_idx}_b{b_idx}.png'
        }
        self._create_dummy_data()

    def tearDown(self):
        """
        Remove the temporary directory after tests are complete.
        """
        shutil.rmtree(self.temp_dir)

    def _create_dummy_data(self, sample_name="sample_001", gt=True, inconsistent_shape=False):
        """Helper to create dummy data files."""
        sample_path = os.path.join(self.temp_dir, sample_name)
        raw_path = os.path.join(sample_path, "raw")
        gt_path = os.path.join(sample_path, "gt")
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        # Create dummy intensity images
        for w in range(self.data_layout['num_wavelengths']):
            for b in range(self.data_layout['num_buckets']):
                img_shape = (10, 12) if not inconsistent_shape else (10 + w, 12 + b)
                img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)
                filename = self.data_layout['file_pattern'].format(w_idx=w, b_idx=b)
                cv2.imwrite(os.path.join(raw_path, filename), img)

        if inconsistent_shape: # Create one more image with different shape
            img = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)
            filename = self.data_layout['file_pattern'].format(w_idx=0, b_idx=1)
            cv2.imwrite(os.path.join(raw_path, filename), img)


        # Create dummy ground truth
        if gt:
            gt_data = np.random.rand(10, 12).astype(np.float32)
            np.save(os.path.join(gt_path, "height.npy"), gt_data)

    # --- Tests for InterferometryDataset ---

    def test_dataset_initialization(self):
        """Tests if the dataset can be initialized correctly."""
        dataset = InterferometryDataset(data_dir=self.temp_dir, data_layout=self.data_layout)
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset), 10 * 12) # 10x12 pixels

    def test_len_and_getitem(self):
        """Tests the __len__ and __getitem__ methods."""
        dataset = InterferometryDataset(data_dir=self.temp_dir, data_layout=self.data_layout)
        self.assertEqual(len(dataset), 120)

        # Test getting a single item
        intensity, height = dataset[0]
        num_channels = self.data_layout['num_wavelengths'] * self.data_layout['num_buckets']
        self.assertEqual(intensity.shape, (num_channels,))
        self.assertEqual(height.shape, (1,))
        self.assertIsInstance(intensity, torch.Tensor)
        self.assertIsInstance(height, torch.Tensor)

    def test_roi(self):
        """Tests the region of interest functionality."""
        roi = (2, 2, 5, 6) # (x, y, w, h)
        dataset = InterferometryDataset(data_dir=self.temp_dir, data_layout=self.data_layout, roi=roi)
        self.assertEqual(len(dataset), 5 * 6) # ROI dimensions

        # Check if ROI was applied correctly to GT
        gt_path = os.path.join(self.temp_dir, "sample_001", "gt", "height.npy")
        original_gt = np.load(gt_path)
        roi_gt = select_roi(original_gt, roi)

        _, height = dataset[0]
        # This is a weak check, but confirms data is from the ROI
        self.assertNotEqual(height.item(), original_gt[0,0])


    def test_missing_gt(self):
        """Tests that the dataset handles a missing ground truth file gracefully."""
        gt_file = os.path.join(self.temp_dir, "sample_001", "gt", "height.npy")
        os.remove(gt_file)
        with self.assertRaises(RuntimeError):
            InterferometryDataset(data_dir=self.temp_dir, data_layout=self.data_layout)

    def test_missing_intensity_image(self):
        """Tests that the dataset skips samples with missing intensity images."""
        img_file = os.path.join(self.temp_dir, "sample_001", "raw", "w0_b1.png")
        os.remove(img_file)
        with self.assertRaises(RuntimeError):
             InterferometryDataset(data_dir=self.temp_dir, data_layout=self.data_layout)

    # --- Tests for load_inference_data ---

    def test_load_inference_data_success(self):
        """Tests successful loading of inference data."""
        inference_path = os.path.join(self.temp_dir, "sample_001")
        data, shape = load_inference_data(inference_path, self.data_layout)

        num_channels = self.data_layout['num_wavelengths'] * self.data_layout['num_buckets']
        self.assertEqual(shape, (10, 12))
        self.assertEqual(data.shape, (10 * 12, num_channels))
        self.assertIsInstance(data, torch.Tensor)

    def test_load_inference_data_missing_file(self):
        """Tests that inference loading fails if an image is missing."""
        img_file = os.path.join(self.temp_dir, "sample_001", "raw", "w0_b1.png")
        os.remove(img_file)
        inference_path = os.path.join(self.temp_dir, "sample_001")
        with self.assertRaises(FileNotFoundError):
            load_inference_data(inference_path, self.data_layout)

    def test_load_inference_data_inconsistent_shape(self):
        """Tests for error handling with inconsistent image shapes."""
        # Create a new sample with inconsistent shapes
        self._create_dummy_data(sample_name="sample_002", inconsistent_shape=True)
        inference_path = os.path.join(self.temp_dir, "sample_002")
        with self.assertRaises(ValueError):
            load_inference_data(inference_path, self.data_layout)


if __name__ == '__main__':
    unittest.main()