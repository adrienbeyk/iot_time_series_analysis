import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    try:
        # Load data
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Log the columns we have in the dataframe
        logging.info(f"Columns in the dataframe: {df.columns.tolist()}")

        # Resample to hourly data and forward fill missing values
        df = df.resample('h').mean().ffill()

        # Calculate rolling averages for available columns
        for column in ['temperature', 'power', 'cooling_output']:
            if column in df.columns:
                df[f'{column}_moving_avg'] = df[column].rolling(window=24).mean()

        # Create time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df
    except Exception as e:
        logging.error(f"Error in load_and_preprocess_data: {str(e)}")
        raise


def detect_anomalies(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.Series:
    if column not in df.columns:
        logging.warning(f"Column {column} not found in dataframe. Skipping anomaly detection.")
        return pd.Series()
    z_scores = np.abs(stats.zscore(df[column]))
    return df[column][z_scores > threshold]


def calculate_energy_efficiency(df: pd.DataFrame) -> pd.Series:
    if 'cooling_output' in df.columns and 'power' in df.columns:
        return df['cooling_output'] / df['power']
    else:
        logging.warning("Required columns for energy efficiency calculation not found.")
        return pd.Series()


def peak_demand_analysis(df: pd.DataFrame) -> Tuple[pd.Timestamp, float]:
    if 'power' in df.columns:
        peak_demand = df['power'].max()
        peak_time = df['power'].idxmax()
        return peak_time, peak_demand
    else:
        logging.warning("Power column not found for peak demand analysis.")
        return pd.Timestamp.now(), 0.0


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_correlate = [col for col in ['temperature', 'humidity', 'power', 'cooling_output'] if col in df.columns]
    return df[columns_to_correlate].corr()


def visualize_daily_pattern(df: pd.DataFrame) -> plt.Figure:
    if 'power' in df.columns:
        daily_pattern = df.groupby(df.index.hour)['power'].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        daily_pattern.plot(kind='bar', ax=ax)
        ax.set_title('Average Hourly Power Consumption')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Power Consumption (kW)')
        return fig
    else:
        logging.warning("Power column not found for daily pattern visualization.")
        return plt.Figure()


def main():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data('hvac_data.csv')

        # Detect anomalies in power consumption
        anomalies = detect_anomalies(df, 'power')
        logging.info("Power consumption anomalies:")
        logging.info(anomalies)

        # Calculate and analyze energy efficiency
        df['efficiency'] = calculate_energy_efficiency(df)
        if not df['efficiency'].empty:
            logging.info(f"Average energy efficiency: {df['efficiency'].mean():.2f}")
            logging.info(f"Min energy efficiency: {df['efficiency'].min():.2f}")
            logging.info(f"Max energy efficiency: {df['efficiency'].max():.2f}")

        # Perform peak demand analysis
        peak_time, peak_demand = peak_demand_analysis(df)
        logging.info(f"Peak demand of {peak_demand:.2f} kW occurred at {peak_time}")

        # Perform correlation analysis
        correlation = correlation_analysis(df)
        logging.info("Correlation analysis:")
        logging.info(correlation)

        # Visualize daily power consumption pattern
        fig = visualize_daily_pattern(df)
        if fig:
            fig.savefig('daily_power_pattern.png')
            logging.info("Daily power consumption pattern visualization saved as 'daily_power_pattern.png'")

    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")


if __name__ == "__main__":
    main()