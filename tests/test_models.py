import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_and_merge_data, handle_missing_values, handle_categorical_data, handle_lagged_features_na
from src.feature_engineering import create_lagged_features, add_time_features
from src.models import train_prophet, predict_prophet, evaluate_model, split_data, create_future_dataframe, forecast_sarima, train_sarima, scale_target_variable, inverse_scale_target_variable #added inverse_scale_target_variable import


class TestModels(unittest.TestCase):
    def setUp(self):
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        df = load_and_merge_data(data_path)
        if df is not None:
            df = handle_missing_values(df, method='fillna_0')
            df = handle_categorical_data(df)
            df = add_time_features(df, 'Date')
            df = create_lagged_features(df, 'Weekly_Sales', lags=[1, 2, 3, 4])
            df = handle_lagged_features_na(df)

            self.train_df, self.test_df = split_data(df, test_size=0.2)

    def test_train_prophet(self):
        if self.train_df is not None and self.test_df is not None:
            prophet_train_df, min_sales, max_sales = scale_target_variable(self.train_df.rename(columns={'Date':'ds', 'Weekly_Sales': 'y'}), 'y')
            regressors = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                        'MarkDown5','CPI', 'Unemployment', 'Size', 'Type_A', 'Type_B', 'Type_C', 'month', 'year', 'dayofweek', 'weekofyear',
                        'Weekly_Sales_lag_1', 'Weekly_Sales_lag_2', 'Weekly_Sales_lag_3', 'Weekly_Sales_lag_4']
            prophet_model = train_prophet(prophet_train_df, regressors=regressors)
            self.assertIsNotNone(prophet_model)

    def test_evaluate_model(self):
        if self.train_df is not None and self.test_df is not None:
            prophet_train_df, min_sales, max_sales = scale_target_variable(self.train_df.rename(columns={'Date':'ds', 'Weekly_Sales': 'y'}), 'y')
            prophet_test_df = self.test_df.rename(columns={'Date':'ds', 'Weekly_Sales': 'y'}).copy()
            prophet_test_df = scale_target_variable(prophet_test_df, 'y')[0]
            regressors = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                        'MarkDown5','CPI', 'Unemployment', 'Size', 'Type_A', 'Type_B', 'Type_C', 'month', 'year', 'dayofweek', 'weekofyear',
                        'Weekly_Sales_lag_1', 'Weekly_Sales_lag_2', 'Weekly_Sales_lag_3', 'Weekly_Sales_lag_4']
            prophet_model = train_prophet(prophet_train_df, regressors=regressors)
            prophet_predictions = predict_prophet(prophet_model, prophet_test_df)
            prophet_test_df = inverse_scale_target_variable(prophet_test_df, 'y', min_sales, max_sales)
            prophet_mae, prophet_rmse = evaluate_model(prophet_test_df['original_y'], prophet_predictions['yhat'])

            self.assertIsInstance(prophet_mae, float)
            self.assertIsInstance(prophet_rmse, float)
        
    def test_train_sarima(self):
        if self.train_df is not None:
            sarima_train_df = self.train_df[['Date', 'Weekly_Sales']].set_index('Date')
            order = (2, 1, 2)
            seasonal_order = (0, 1, 0, 52)
            sarima_model = train_sarima(sarima_train_df, order, seasonal_order)
            self.assertIsNotNone(sarima_model)
    
    def test_create_future_dataframe(self):
        if self.train_df is not None:
            future_df = create_future_dataframe(self.train_df, periods=12, freq='W')
            self.assertIsNotNone(future_df)

if __name__ == '__main__':
    unittest.main()
