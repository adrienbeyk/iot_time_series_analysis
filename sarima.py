"""
HVAC IoT Data Analysis with SARIMA

This script performs advanced time series analysis on HVAC IoT data,
including seasonal decomposition, stationarity testing, SARIMA modeling,
and various visualizations to understand energy consumption patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns


# Load and preprocess the data
def load_and_preprocess_data(file_path):
    """
    Load the HVAC IoT data from a CSV file and preprocess it.

    Args:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    pd.DataFrame: Preprocessed hourly data with interpolated missing values.
    """
    data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    return data.resample('H').mean().interpolate()


data = load_and_preprocess_data('hvac_iot_data_small.csv')


# Plot the time series
def plot_time_series(data, column='energy_consumption'):
    """
    Plot the time series data for a given column.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    column (str): The column name to plot (default: 'energy_consumption').
    """
    plt.figure(figsize=(15, 10))
    plt.plot(data[column])
    plt.title(f'{column.replace("_", " ").title()} Over Time')
    plt.xlabel('Date')
    plt.ylabel(column.replace("_", " ").title())
    plt.show()


plot_time_series(data)


# Perform seasonal decomposition
def perform_seasonal_decomposition(data, column='energy_consumption', period=24):
    """
    Perform seasonal decomposition on the time series data.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    column (str): The column name to decompose (default: 'energy_consumption').
    period (int): The period of the seasonality (default: 24 for hourly data).
    """
    decomposition = seasonal_decompose(data[column], model='additive', period=period)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    plt.show()


perform_seasonal_decomposition(data)


# Perform Augmented Dickey-Fuller test
def perform_adf_test(data, column='energy_consumption'):
    """
    Perform the Augmented Dickey-Fuller test for stationarity.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    column (str): The column name to test (default: 'energy_consumption').
    """
    result = adfuller(data[column].dropna())
    print('Augmented Dickey-Fuller Test Results:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')


perform_adf_test(data)


# Plot ACF and PACF
def plot_acf_pacf(data, column='energy_consumption'):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    column (str): The column name to analyze (default: 'energy_consumption').
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    plot_acf(data[column].dropna(), ax=ax1)
    plot_pacf(data[column].dropna(), ax=ax2)
    plt.show()


plot_acf_pacf(data)


# Fit SARIMA model and make predictions
def fit_sarima_model(data, column='energy_consumption', order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    """
    Fit a SARIMA model to the time series data and make predictions.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    column (str): The column name to model (default: 'energy_consumption').
    order (tuple): The (p, d, q) order of the SARIMA model.
    seasonal_order (tuple): The (P, D, Q, s) seasonal order of the SARIMA model.

    Returns:
    SARIMAXResultsWrapper: The fitted SARIMA model results.
    """
    model = SARIMAX(data[column], order=order, seasonal_order=seasonal_order)
    results = model.fit()

    # Make predictions
    forecast = results.get_forecast(steps=24)
    forecast_ci = forecast.conf_int()

    plt.figure(figsize=(15, 10))
    plt.plot(data[column].iloc[-168:], label='Actual')
    plt.plot(forecast.predicted_mean, label='Forecast')
    plt.fill_between(forecast_ci.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1], color='k', alpha=.2)
    plt.title('SARIMA Forecast')
    plt.legend()
    plt.show()

    return results, forecast


results, forecast = fit_sarima_model(data)


# Calculate SARIMA model accuracy
def calculate_model_accuracy(actual, predicted):
    """
    Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for the model.

    Args:
    actual (pd.Series): The actual values.
    predicted (pd.Series): The predicted values.
    """
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    print(f'SARIMA MSE: {mse}')
    print(f'SARIMA MAE: {mae}')


calculate_model_accuracy(data['energy_consumption'].iloc[-24:], forecast.predicted_mean)


# Analyze correlation between variables
def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for all variables in the dataset.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    """
    correlation = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


plot_correlation_heatmap(data)


# Analyze energy consumption patterns
def analyze_consumption_patterns(data, column='energy_consumption'):
    """
    Analyze and plot energy consumption patterns by hour and day of week.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    column (str): The column name to analyze (default: 'energy_consumption').
    """
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek

    plt.figure(figsize=(15, 10))
    sns.boxplot(x='hour', y=column, data=data)
    plt.title(f'{column.replace("_", " ").title()} by Hour')
    plt.show()

    plt.figure(figsize=(15, 10))
    sns.boxplot(x='day_of_week', y=column, data=data)
    plt.title(f'{column.replace("_", " ").title()} by Day of Week')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()


analyze_consumption_patterns(data)


# Analyze the relationship between temperature and energy consumption
def plot_temperature_vs_consumption(data):
    """
    Plot a scatter plot of temperature vs energy consumption.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    """
    plt.figure(figsize=(15, 10))
    plt.scatter(data['temperature'], data['energy_consumption'])
    plt.title('Temperature vs Energy Consumption')
    plt.xlabel('Temperature')
    plt.ylabel('Energy Consumption')
    plt.show()


plot_temperature_vs_consumption(data)


# Perform a rolling average analysis
def perform_rolling_average_analysis(data, column='energy_consumption', window=24):
    """
    Perform and plot rolling average analysis.

    Args:
    data (pd.DataFrame): The dataframe containing the time series data.
    column (str): The column name to analyze (default: 'energy_consumption').
    window (int): The rolling window size (default: 24 for daily rolling average).
    """
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()

    plt.figure(figsize=(15, 10))
    plt.plot(data.index, data[column], label='Original')
    plt.plot(data.index, rolling_mean, label='Rolling Mean')
    plt.plot(data.index, rolling_std, label='Rolling Std')
    plt.title('Rolling Statistics')
    plt.legend()
    plt.show()


perform_rolling_average_analysis(data)

# Print model summary
print(results.summary())

# Main execution
if __name__ == "__main__":
    print("HVAC IoT Data Analysis Complete")