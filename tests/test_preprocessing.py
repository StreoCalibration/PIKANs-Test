import unittest
import numpy as np
from src.preprocessing.filters import normalize_intensity, select_roi

class TestPreprocessing(unittest.TestCase):
    """
    Test suite for preprocessing functions.
    """

    def test_normalize_intensity_minmax(self):
        """
        Tests the min-max normalization.
        """
        img = np.array([[0, 50], [100, 200]], dtype=np.float32)
        normalized_img = normalize_intensity(img, method='minmax')

        # Expected output after min-max normalization to [0, 1]
        # (X - min) / (max - min)
        # min = 0, max = 200
        # (0 - 0) / 200 = 0
        # (50 - 0) / 200 = 0.25
        # (100 - 0) / 200 = 0.5
        # (200 - 0) / 200 = 1.0
        expected = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)

        self.assertTrue(np.allclose(normalized_img, expected))
        self.assertEqual(normalized_img.min(), 0.0)
        self.assertEqual(normalized_img.max(), 1.0)

    def test_select_roi(self):
        """
        Tests the select_roi function.
        """
        img = np.arange(25).reshape(5, 5)
        # roi = (x, y, w, h)
        roi = (1, 1, 3, 3)

        roi_img = select_roi(img, roi)

        expected = np.array([
            [6, 7, 8],
            [11, 12, 13],
            [16, 17, 18]
        ])

        self.assertTrue(np.array_equal(roi_img, expected))

if __name__ == '__main__':
    unittest.main()
