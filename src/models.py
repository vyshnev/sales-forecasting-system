import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_sarima(train_data, order, seasonal_order):
    """Trains a SARIMA model."""
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False, low_memory=True,
                    simple_differencing=True)
    model_fit = model.fit(disp=False, method='lbfgs', maxiter=50)
    return model_fit

def predict_sarima(model, forecast_steps, test_data):
    """Makes predictions using a trained SARIMA model."""
    predictions = model.get_forecast(steps=forecast_steps)
    predictions_series = pd.Series(predictions.predicted_mean, index = test_data.index) #align with index
    predictions_series = predictions_series.fillna(0)
    return predictions_series

def forecast_sarima(model, future_dates, train_data):
    """Make forecast using a trained SARIMA model."""
    start = train_data.index[-1]
    predictions = model.get_forecast(steps = len(future_dates), index = future_dates)
    predictions_series = pd.Series(predictions.predicted_mean, index=future_dates)
    predictions_series = predictions_series.fillna(0)
    return predictions_series

def train_prophet(train_data, regressors=[]):
    """Trains a Prophet model."""
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
    """Splits data into training and testing sets
        Args:
            df: Pandas dataframe
            test_size: Proportion of data to be used for testing
            time_column: Column that specifies the time
            sort: Boolean, specifying if the data needs to be sorted by the time
    """
    if sort:
      df = df.sort_values(by=time_column)
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    return train_df, test_df

def create_future_dataframe(df, periods, freq='W', time_column = 'Date'):
      """Creates a dataframe with future dates."""
      last_date = df[time_column].max()
      future_dates = pd.date_range(start=last_date, periods=periods, freq=freq)
      future_df = pd.DataFrame({time_column:future_dates})
      return future_df

def scale_target_variable(df, target_variable):
   """Scales a target variable between 0 and 1 using min-max scaling"""
   min_value = df[target_variable].min()
   max_value = df[target_variable].max()
   df['scaled_' + target_variable] = (df[target_variable] - min_value) / (max_value - min_value)
   return df, min_value, max_value

def inverse_scale_target_variable(df, target_variable, min_value, max_value):
    """Inverse scales a scaled target variable"""
    df['original_' + target_variable] = df['scaled_' + target_variable] * (max_value - min_value) + min_value
    return df