from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_and_merge_data, handle_missing_values, handle_categorical_data, handle_lagged_features_na
from src.feature_engineering import create_lagged_features, add_time_features
from src.models import  train_prophet, predict_prophet, create_future_dataframe, forecast_sarima, train_sarima, scale_target_variable, inverse_scale_target_variable

app = Flask(__name__)

# Load Model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

PROPHET_MODEL_PATH = os.path.join(MODEL_DIR, 'prophet_model.pkl')
SARIMA_MODEL_PATH = os.path.join(MODEL_DIR, 'sarima_model.pkl')
MIN_MAX_SCALER_PATH = os.path.join(MODEL_DIR, 'min_max_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) #get json data from API call
        #load model, parameters and data
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        df = load_and_merge_data(data_path)
        if df is not None:
          df = handle_missing_values(df, method='fillna_0')
          df = handle_categorical_data(df)
          df = add_time_features(df, 'Date')
          df = create_lagged_features(df, 'Weekly_Sales', lags=[1, 2, 3, 4])
          df = handle_lagged_features_na(df)
          # Train on all the data to get full model
          prophet_full_train_df, min_sales_full, max_sales_full = scale_target_variable(df.rename(columns={'Date':'ds', 'Weekly_Sales': 'y'}), 'y')
          
          #load the models
          with open(PROPHET_MODEL_PATH, 'rb') as f:
              full_prophet_model = pickle.load(f)
          with open(SARIMA_MODEL_PATH, 'rb') as f:
            full_sarima_model = pickle.load(f)
          with open(MIN_MAX_SCALER_PATH, 'rb') as f:
              min_max_scaler = pickle.load(f)


          # Generate Future dates for 1 quarter.
          future_dates_df = create_future_dataframe(df, periods = 12, freq='W')
          future_dates_df = future_dates_df.rename(columns={'Date':'ds'})

          regressors = ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                'MarkDown5','CPI', 'Unemployment', 'Size', 'Type_A', 'Type_B', 'Type_C', 'month', 'year', 'dayofweek', 'weekofyear',
                'Weekly_Sales_lag_1', 'Weekly_Sales_lag_2', 'Weekly_Sales_lag_3', 'Weekly_Sales_lag_4']
           # Create all the regressors for future dates. This assumes that we can compute the other regressors
          # Note that for this example we will be filling all the future regressors as 0, but we would
          # typically use forecast methods to predict them as well.
          for regressor in regressors:
            future_dates_df[regressor] = 0 #set to 0 for this example
          # Use the trained Prophet model to make predictions on future dates.
          future_prophet_predictions = predict_prophet(full_prophet_model, future_dates_df)
          future_prophet_predictions = scale_target_variable(future_prophet_predictions, 'yhat')[0]
          future_prophet_predictions = inverse_scale_target_variable(future_prophet_predictions, 'yhat', min_max_scaler['min_sales_full'], min_max_scaler['max_sales_full'])

          future_sarima_dates = pd.date_range(start=df['Date'].max(), periods=12, freq='W')
          future_sarima_predictions = forecast_sarima(full_sarima_model, future_sarima_dates, df[['Date', 'Weekly_Sales']].set_index('Date') )

          return jsonify({
              'prophet_predictions': future_prophet_predictions[['ds', 'original_yhat']].to_dict('records'),
              'sarima_predictions': future_sarima_predictions.rename(lambda x: x.strftime('%Y-%m-%d')).to_dict()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Train and save model if it doesn't exist
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = load_and_merge_data(data_path)
    if df is not None:
      df = handle_missing_values(df, method='fillna_0')
      df = handle_categorical_data(df)
      df = add_time_features(df, 'Date')
      df = create_lagged_features(df, 'Weekly_Sales', lags=[1, 2, 3, 4])
      df = handle_lagged_features_na(df)

      if not os.path.exists(PROPHET_MODEL_PATH) or not os.path.exists(SARIMA_MODEL_PATH) :
          print("Model does not exist, Training Model")
          prophet_full_train_df, min_sales_full, max_sales_full = scale_target_variable(df.rename(columns={'Date':'ds', 'Weekly_Sales': 'y'}), 'y')
          full_prophet_model = train_prophet(prophet_full_train_df, regressors=[
                  'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                    'MarkDown5','CPI', 'Unemployment', 'Size', 'Type_A', 'Type_B', 'Type_C', 'month', 'year', 'dayofweek', 'weekofyear',
                    'Weekly_Sales_lag_1', 'Weekly_Sales_lag_2', 'Weekly_Sales_lag_3', 'Weekly_Sales_lag_4'
          ])
          sarima_full_train_df = df[['Date', 'Weekly_Sales']].set_index('Date')
          order = (2, 1, 2)
          seasonal_order = (0, 1, 0, 52)
          full_sarima_model = train_sarima(sarima_full_train_df, order, seasonal_order)

          with open(PROPHET_MODEL_PATH, 'wb') as f:
              pickle.dump(full_prophet_model, f)

          with open(SARIMA_MODEL_PATH, 'wb') as f:
            pickle.dump(full_sarima_model, f)
          
          with open(MIN_MAX_SCALER_PATH, 'wb') as f:
             pickle.dump({
                 'min_sales_full':min_sales_full,
                 'max_sales_full':max_sales_full
             }, f)

    app.run(debug=True, host='0.0.0.0')