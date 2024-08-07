import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


# Analyze HVAC Data Practice
# - Calculates energy consumption from power data.
# - Resamples the data from 15-minute intervals to hourly intervals.

def load_and_process_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Calculate energy consumption for 15-minute intervals
    # Energy (kWh) = Power (kW) * Time (hours)
    df['energy_consumption'] = df['power'] * (15 / 60)  # 15 minutes = 0.25 hours

    # Resample data to hourly averages
    hourly_data = df.resample('h').mean()

    # Calculate hourly energy consumption by summing the 15-minute intervals
    hourly_data['energy_consumption'] = df['energy_consumption'].resample('h').sum()

    # Notes:
    # Energy consumption is the integral of power over time.
    # For discrete time intervals, we multiply power by the duration of the interval.
    # When resampling to a larger time interval (like 15 minutes to 1 hour),
    # summing the energy consumption values is more accurate
    # than averaging the power and then calculating energy.

    return df, hourly_data


def visualize_data_comparison(original_df: pd.DataFrame, resampled_df: pd.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    fig.suptitle('HVAC Data: Original vs Resampled', fontsize=16)

    # Color palette
    colors = sns.color_palette("husl", 2)

    # Temperature plot
    axs[0].plot(original_df.index, original_df['temperature'], color=colors[0], alpha=0.5, label='Original')
    axs[0].plot(resampled_df.index, resampled_df['temperature'], color=colors[1], label='Resampled')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].legend()

    # Power consumption plot
    axs[1].plot(original_df.index, original_df['power'], color=colors[0], alpha=0.5, label='Original')
    axs[1].plot(resampled_df.index, resampled_df['power'], color=colors[1], label='Resampled')
    axs[1].set_ylabel('Power Consumption (kW)')
    axs[1].legend()

    # Cooling output plot
    axs[2].plot(original_df.index, original_df['cooling_output'], color=colors[0], alpha=0.5, label='Original')
    axs[2].plot(resampled_df.index, resampled_df['cooling_output'], color=colors[1], label='Resampled')
    axs[2].set_ylabel('Cooling Output (kW)')
    axs[2].legend()

    # X-axis formatting
    for ax in axs:
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def analyze_hvac_data(file_path: str) -> Dict[str, float]:
    df, hourly_data = load_and_process_data(file_path)

    # Find peak hours (top 10% of energy consumption)
    # - quantile(0.9) returns the value below which 90% of the observations fall
    # - In other words, this is the energy consumption level that is higher than 90% of all hourly readings
    # - We use this as our definition of 'peak' energy consumption
    peak_threshold = hourly_data['energy_consumption'].quantile(0.9)

    # Select hours where energy consumption is greater than or equal to the peak threshold
    # - This creates a new DataFrame containing only the rows (hours) where energy consumption
    #   is at or above our calculated threshold
    # - These represent the hours of highest energy usage in our dataset
    # - The number of rows in peak_hours will be approximately 10% of the total hours
    peak_hours = hourly_data[hourly_data['energy_consumption'] >= peak_threshold]

    # Calculate efficiency
    hourly_data['efficiency'] = hourly_data['cooling_output'] / hourly_data['energy_consumption']

    # Create and save the visualization
    fig = visualize_data_comparison(df, hourly_data)
    fig.savefig('hvac_data_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'total_energy': float(hourly_data['energy_consumption'].sum()),
        'avg_efficiency': float(hourly_data['efficiency'].mean()),
        'peak_hours_count': int(len(peak_hours)),
        'max_temperature': float(df['temperature'].max()),
        'min_temperature': float(df['temperature'].min())
    }


# Usage
results = analyze_hvac_data('hvac_data.csv')
print(results)
print("Visualization saved as 'hvac_data_comparison.png'")
