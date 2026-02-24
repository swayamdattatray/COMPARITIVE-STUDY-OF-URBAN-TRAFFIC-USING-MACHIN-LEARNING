"""
Comparative Study of Machine Learning Models for Urban Traffic Flow Prediction.

This module implements an end-to-end workflow suitable for a final-year engineering
project. It includes:
- Data loading from CSV
- Preprocessing and feature engineering
- Model training (ARIMA, Random Forest, XGBoost, LSTM)
- Evaluation with MAE, RMSE, MAPE
- Comparison table and visualizations

Expected input CSV columns:
- traffic_flow (target)
- vehicle_speed
- time_of_day
- day_of_week
- weather_condition (optional)
- date_time
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Optional dependencies (kept optional so the script still runs if absent)
try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency import guard
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency import guard
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    TENSORFLOW_AVAILABLE = False


@dataclass
class DatasetBundle:
    """Container for prepared datasets and metadata."""

    raw_data: pd.DataFrame
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    feature_columns: List[str]


@dataclass
class ModelResult:
    """Store model predictions and metrics."""

    model_name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    mae: float
    rmse: float
    mape: float


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load traffic data from CSV, parse datetime, and sort chronologically."""
    data = pd.read_csv(file_path)

    if "date_time" not in data.columns:
        raise ValueError("Input CSV must contain a 'date_time' column.")

    data["date_time"] = pd.to_datetime(data["date_time"], errors="coerce")
    data = data.dropna(subset=["date_time"]).sort_values("date_time").reset_index(drop=True)

    required_columns = {"traffic_flow", "vehicle_speed", "time_of_day", "day_of_week"}
    missing = required_columns.difference(set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return data


def preprocess_data(data: pd.DataFrame, train_ratio: float = 0.8) -> DatasetBundle:
    """Handle missing values, engineer lag/rolling features, and split data (80-20)."""
    df = data.copy()

    # Basic temporal features from date_time
    df["hour"] = df["date_time"].dt.hour
    df["month"] = df["date_time"].dt.month

    # Feature engineering for time-series signal
    df["traffic_lag_1"] = df["traffic_flow"].shift(1)
    df["traffic_lag_2"] = df["traffic_flow"].shift(2)
    df["traffic_roll_mean_3"] = df["traffic_flow"].rolling(window=3).mean()

    # Forward/backward filling can stabilize generated NaNs from lag/rolling
    df = df.ffill().bfill()

    split_index = int(len(df) * train_ratio)
    train_data = df.iloc[:split_index].copy()
    test_data = df.iloc[split_index:].copy()

    feature_columns = [
        "vehicle_speed",
        "time_of_day",
        "day_of_week",
        "hour",
        "month",
        "traffic_lag_1",
        "traffic_lag_2",
        "traffic_roll_mean_3",
    ]
    if "weather_condition" in df.columns:
        feature_columns.append("weather_condition")

    return DatasetBundle(
        raw_data=df,
        train_data=train_data,
        test_data=test_data,
        feature_columns=feature_columns,
    )


def _build_ml_preprocessor(train_df: pd.DataFrame, feature_columns: List[str]) -> ColumnTransformer:
    """Create a pipeline to impute missing values and normalize numeric columns."""
    numeric_cols = [
        col
        for col in feature_columns
        if col != "weather_condition" and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    categorical_cols = [col for col in feature_columns if col not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def train_arima(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Train ARIMA using traffic_flow as univariate time-series and forecast test horizon."""
    model = ARIMA(train_df["traffic_flow"], order=(2, 1, 2))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test_df))
    return np.asarray(forecast)


def train_random_forest(bundle: DatasetBundle) -> np.ndarray:
    """Train Random Forest Regressor and return predictions."""
    X_train = bundle.train_data[bundle.feature_columns]
    y_train = bundle.train_data["traffic_flow"]
    X_test = bundle.test_data[bundle.feature_columns]

    preprocessor = _build_ml_preprocessor(bundle.train_data, bundle.feature_columns)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_xgboost(bundle: DatasetBundle) -> np.ndarray:
    """Train XGBoost Regressor and return predictions."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Install with: pip install xgboost")

    X_train = bundle.train_data[bundle.feature_columns]
    y_train = bundle.train_data["traffic_flow"]
    X_test = bundle.test_data[bundle.feature_columns]

    preprocessor = _build_ml_preprocessor(bundle.train_data, bundle.feature_columns)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    objective="reg:squarederror",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _create_lstm_sequences(series: np.ndarray, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1D series into LSTM supervised sequences."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        y.append(series[i])
    X_array = np.array(X)
    y_array = np.array(y)
    return X_array.reshape((X_array.shape[0], X_array.shape[1], 1)), y_array


def train_lstm(bundle: DatasetBundle, lookback: int = 10, epochs: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train an LSTM on normalized traffic_flow sequence.

    Returns:
        tuple(y_true_adjusted, y_pred), where y_true_adjusted matches LSTM forecast horizon.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("tensorflow is not installed. Install with: pip install tensorflow")

    scaler = StandardScaler()
    train_series = scaler.fit_transform(bundle.train_data[["traffic_flow"]]).flatten()

    full_series = np.concatenate(
        [train_series, scaler.transform(bundle.test_data[["traffic_flow"]]).flatten()]
    )
    train_cutoff = len(train_series)

    X_train, y_train = _create_lstm_sequences(full_series[:train_cutoff], lookback=lookback)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    test_scaled = scaler.transform(bundle.test_data[["traffic_flow"]]).flatten()
    context_series = np.concatenate([train_series[-lookback:], test_scaled])

    X_test, _ = _create_lstm_sequences(context_series, lookback=lookback)
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    y_true = bundle.test_data["traffic_flow"].to_numpy()
    y_true_adjusted = y_true[: len(y_pred)]
    return y_true_adjusted, y_pred


def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelResult:
    """Compute MAE, RMSE, and MAPE for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    return ModelResult(
        model_name=model_name,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        mae=mae,
        rmse=rmse,
        mape=mape,
    )


def plot_results(results: Dict[str, ModelResult], output_dir: str | Path = "outputs") -> None:
    """Plot actual vs predicted values and model error comparison chart."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Actual vs Predicted for each model
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)), sharex=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(res.y_true, label="Actual", color="black", linewidth=2)
        ax.plot(res.y_pred, label=f"Predicted ({name})", linestyle="--")
        ax.set_title(f"Actual vs Predicted - {name}")
        ax.set_ylabel("Traffic Flow")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.savefig(output_path / "actual_vs_predicted.png", dpi=300)
    plt.close()

    # Plot 2: Error comparison
    comparison_df = pd.DataFrame(
        {
            "Model": [res.model_name for res in results.values()],
            "MAE": [res.mae for res in results.values()],
            "RMSE": [res.rmse for res in results.values()],
            "MAPE": [res.mape for res in results.values()],
        }
    )

    ax = comparison_df.set_index("Model").plot(kind="bar", figsize=(12, 6), rot=0)
    ax.set_title("Model Error Comparison")
    ax.set_ylabel("Error Value")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "error_comparison.png", dpi=300)
    plt.close()


def run_pipeline(csv_path: str | Path) -> pd.DataFrame:
    """Execute complete workflow and print comparison table."""
    data = load_data(csv_path)
    bundle = preprocess_data(data)

    y_test = bundle.test_data["traffic_flow"].to_numpy()
    results: Dict[str, ModelResult] = {}

    # ARIMA
    arima_pred = train_arima(bundle.train_data, bundle.test_data)
    results["ARIMA"] = evaluate_model("ARIMA", y_test, arima_pred)

    # Random Forest
    rf_pred = train_random_forest(bundle)
    results["Random Forest"] = evaluate_model("Random Forest", y_test, rf_pred)

    # XGBoost
    try:
        xgb_pred = train_xgboost(bundle)
        results["XGBoost"] = evaluate_model("XGBoost", y_test, xgb_pred)
    except ImportError as err:
        print(f"[INFO] Skipping XGBoost: {err}")

    # LSTM
    try:
        lstm_y_true, lstm_pred = train_lstm(bundle)
        results["LSTM"] = evaluate_model("LSTM", lstm_y_true, lstm_pred)
    except ImportError as err:
        print(f"[INFO] Skipping LSTM: {err}")

    comparison_table = pd.DataFrame(
        [
            {
                "Model": res.model_name,
                "MAE": res.mae,
                "RMSE": res.rmse,
                "MAPE": res.mape,
            }
            for res in results.values()
        ]
    ).sort_values(by="RMSE")

    print("\nModel Comparison Table (lower is better):")
    print(comparison_table.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    plot_results(results)
    print("\nPlots saved in: ./outputs")

    return comparison_table


if __name__ == "__main__":
    # Update path as needed, e.g.: data/urban_traffic.csv
    CSV_PATH = "traffic_data.csv"
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(
            f"CSV file '{CSV_PATH}' not found. Place your dataset in project root or modify CSV_PATH."
        )

    run_pipeline(CSV_PATH)
