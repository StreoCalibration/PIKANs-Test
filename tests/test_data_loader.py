import unittest
# from src.data_loader.datasets import InterferometryDataset

class TestDataLoader(unittest.TestCase):
    def test_dataset_initialization(self):
        """
        Tests if the dataset can be initialized.
        """
        # TODO: Create dummy data and test initialization
        # dataset = InterferometryDataset(data_dir="dummy/data", gt_dir="dummy/gt")
        # self.assertIsNotNone(dataset)
        pass

    def test_data_loading(self):
        """
        Tests if a single item can be loaded.
        """
        # TODO: Test the __getitem__ method
        pass

if __name__ == '__main__':
    unittest.main()