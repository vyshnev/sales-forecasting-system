import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_sarima(train_data, order, seasonal_order):
    """Trains a SARIMA model on the given data."""

    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    return model_fit

def predict_sarima(model, forecast_steps):
    """Make predictions using a trained SARIMA model."""
    predictions= model.get_forecast(steps=forecast_steps)
    return predictions.predicted_mean

def train_prophet(train_data, regressors=[]):
    """Train a Prophet model"""
    model = Prophet()
    if len(regressors) > 0:
        for regressor in regressors:
            model.add_regressor(regressor)
    model.fit(train_data)
    return model

def predict_prophet(model, future_df):
    """Makes predictions using a trained Prophet model."""
    predictions = model.predict(future_df)
    return predictions


def evaluate_model(y_true, y_pred):
    """Evaluates model performance."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Other metrics
    return mae, rmse

def split_data(df, test_size=0.2, time_column='Date', sort=True):
    """Split data into training and test sets.
        Args:
            df: Pandas dataframe
            test_size: Proportion of the data to be used for testing
            time_column: Column that specifies the time.
            sore: Boolean, that specifies if the data needs to be sorted by the time"""
    
    if sort:
        df = df.sort_values(by=time_column)
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    return train_df, test_df
