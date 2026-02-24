# Comparative Study of Machine Learning Models for Urban Traffic Flow Prediction

This repository contains a complete Python machine learning project for predicting urban traffic flow and comparing multiple forecasting models.

## Project Objective
Build and compare the performance of:
- ARIMA (statistical time-series model)
- Random Forest Regressor
- XGBoost Regressor
- LSTM (deep learning sequence model)

## Dataset Assumptions
Input dataset is a CSV file with these columns:
- `traffic_flow` (target)
- `vehicle_speed`
- `time_of_day`
- `day_of_week`
- `weather_condition` (optional)
- `date_time`

## Features Implemented
- Data loading with date parsing and validation
- Missing value handling
- Normalization for numeric features
- Feature engineering:
  - Lag features (`traffic_lag_1`, `traffic_lag_2`)
  - Rolling mean (`traffic_roll_mean_3`)
- Chronological train-test split (80-20)
- Model evaluation metrics:
  - MAE
  - RMSE
  - MAPE
- Comparison table for all trained models
- Plots:
  - Actual vs Predicted
  - Error comparison bar chart

## Project Structure
- `traffic_flow_prediction.py`: Main pipeline implementation with required functions:
  - `load_data()`
  - `preprocess_data()`
  - `train_arima()`
  - `train_random_forest()`
  - `train_xgboost()`
  - `train_lstm()`
  - `evaluate_model()`

## Setup
```bash
pip install -r requirements.txt
```

## Usage
1. Place your dataset in the project root and name it `traffic_data.csv` (or edit `CSV_PATH` in script).
2. Run:

```bash
python traffic_flow_prediction.py
```

## Outputs
After execution:
- Printed model comparison table in terminal
- Saved plots under `outputs/`:
  - `actual_vs_predicted.png`
  - `error_comparison.png`

## Notes
- `xgboost` and `tensorflow` are optional at runtime. If unavailable, the script skips corresponding models and continues with available ones.
- Code is written with clear structure and comments to align with final-year engineering project expectations.
