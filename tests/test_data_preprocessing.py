import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_and_merge_data, handle_missing_values, handle_categorical_data, handle_lagged_features_na

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
       data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
       self.df = load_and_merge_data(data_path)
       if self.df is not None:
           self.df = handle_missing_values(self.df, method='fillna_0')

    def test_load_and_merge_data(self):
        self.assertIsNotNone(self.df)

    def test_handle_missing_values(self):
      if self.df is not None:
        self.assertFalse(self.df.isnull().values.any(), "Missing values should have been handled")

    def test_handle_categorical_data(self):
        if self.df is not None:
            df = handle_categorical_data(self.df)
            self.assertTrue('Type_A' in df.columns)
            self.assertTrue('Type_B' in df.columns)
            self.assertTrue('Type_C' in df.columns)


if __name__ == '__main__':
    unittest.main()