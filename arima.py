import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns


def forecast_hvac_load(data: pd.Series, forecast_horizon: int = 24) -> Tuple[pd.Series, pd.Series, ARIMA]:
    # Fit ARIMA model
    model = ARIMA(data, order=(1, 1, 1))  # Example order, may need tuning
    results = model.fit()

    # Generate forecast
    forecast = results.forecast(steps=forecast_horizon)

    # Calculate confidence intervals
    conf_int = results.get_forecast(steps=forecast_horizon).conf_int()

    return forecast, conf_int, results


def plot_time_series(data: pd.Series, forecast: pd.Series, conf_int: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data, label='Historical Data')
    plt.plot(forecast.index, forecast, color='red', label='Forecast')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('HVAC Load: Historical Data and Forecast')
    plt.xlabel('Timestamp')
    plt.ylabel('Load')
    plt.legend()
    plt.show()


def plot_residuals(results: ARIMA):
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.tight_layout()
    plt.show()


def plot_diagnostics(results: ARIMA):
    plt.figure(figsize=(12, 8))
    results.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(data: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data, ax=ax1)
    plot_pacf(data, ax=ax2)
    plt.tight_layout()
    plt.show()


def evaluate_forecast(actual: pd.Series, forecast: pd.Series):
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, forecast)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")


def plot_seasonal_decompose(data: pd.Series):
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(data, model='additive', period=24)  # Assuming hourly data
    result.plot()
    plt.tight_layout()
    plt.show()


def main():
    # Load your time series data
    df = pd.read_csv('hvac_load.csv', parse_dates=['timestamp'], index_col='timestamp')
    load_series = df['load']

    # Plot original time series
    plt.figure(figsize=(12, 6))
    load_series.plot()
    plt.title('Original HVAC Load Time Series')
    plt.show()

    # Plot ACF and PACF
    plot_acf_pacf(load_series)

    # Plot seasonal decomposition
    plot_seasonal_decompose(load_series)

    # Perform forecasting
    forecast, conf_int, results = forecast_hvac_load(load_series)

    # Plot time series with forecast
    plot_time_series(load_series, forecast, conf_int)

    # Plot residuals
    plot_residuals(results)

    # Plot model diagnostics
    plot_diagnostics(results)

    # Evaluate forecast (assuming we have actual values for the forecast period)
    if len(load_series) > len(forecast):
        actual = load_series[-len(forecast):]
        evaluate_forecast(actual, forecast)

    print("Forecast:")
    print(forecast)
    print("\nConfidence Intervals:")
    print(conf_int)


if __name__ == "__main__":
    main()