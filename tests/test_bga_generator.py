import unittest
import numpy as np
import os
import shutil
import tempfile

# Add src to path to allow direct script import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.generate_bga_synthetic_data import create_bga_height_map, simulate_interferometry

class TestBgaGenerator(unittest.TestCase):
    """
    Test suite for the BGA synthetic data generator script.
    """

    def test_create_bga_height_map(self):
        """
        Tests the core height map generation function.
        """
        img_shape = (100, 120)
        ball_height_um = 50.0

        height_map = create_bga_height_map(
            img_shape=img_shape,
            ball_diameter_um=100.0,
            ball_height_um=ball_height_um,
            ball_pitch_um=(80.0, 80.0),
            pixels_per_um=0.5,
            randomize=False
        )

        # 1. Check shape
        self.assertEqual(height_map.shape, img_shape)

        # 2. Check dtype
        self.assertEqual(height_map.dtype, np.float32)

        # 3. Check if max height is close to expected (in nm)
        # The generated profile is added to a zero substrate, so max should be ball height
        expected_max_height_nm = ball_height_um * 1000.0
        self.assertAlmostEqual(height_map.max(), expected_max_height_nm, delta=1e-3)

        # 4. Check if there are non-zero values (i.e., balls were generated)
        self.assertTrue(np.any(height_map > 0))

    def test_simulate_interferometry(self):
        """
        Tests the interferometry simulation function.
        """
        height_map_nm = np.random.rand(50, 60).astype(np.float32) * 800.0
        wavelengths_nm = [633.0, 532.0, 450.0]
        num_buckets = 4

        images = simulate_interferometry(
            gt_height_map_nm=height_map_nm,
            wavelengths_nm=wavelengths_nm,
            num_buckets=num_buckets
        )

        # 1. Check number of images
        expected_num_images = len(wavelengths_nm) * num_buckets
        self.assertEqual(len(images), expected_num_images)

        # 2. Check image properties
        for img in images:
            self.assertEqual(img.shape, height_map_nm.shape)
            self.assertEqual(img.dtype, np.uint8)
            self.assertTrue(np.all(img >= 0) and np.all(img <= 255))

if __name__ == '__main__':
    unittest.main()
