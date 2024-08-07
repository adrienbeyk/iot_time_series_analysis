import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
# This ensures that the random elements in our data generation are consistent across runs
np.random.seed(42)

# Generate timestamp range (3 months)
# We choose a 3-month period to capture seasonal trends while keeping the dataset manageable
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 3, 31)
# Use 30-minute intervals to simulate frequent IoT sensor readings without overwhelming the system
date_range = pd.date_range(start=start_date, end=end_date, freq='30T')

# Create base DataFrame
# We initialize an empty DataFrame with our date range as the index and columns for our key metrics
df = pd.DataFrame(index=date_range, columns=['energy_consumption', 'temperature', 'humidity'])

# Generate base patterns for energy consumption
# We use sine waves to simulate cyclical patterns commonly seen in HVAC systems
# The base load is 50 kWh, representing constant energy usage
df['energy_consumption'] = 50 + 30 * np.sin(np.arange(len(df)) * (2 * np.pi / (24 * 2)))  # Daily pattern (24 hours * 2 readings per hour)
df['energy_consumption'] += 20 * np.sin(np.arange(len(df)) * (2 * np.pi / (24 * 2 * 7)))  # Weekly pattern (7 days)
df['energy_consumption'] += 10 * np.sin(np.arange(len(df)) * (2 * np.pi / (24 * 2 * 90)))  # Seasonal pattern (90 days)

# Add random fluctuations to energy consumption
# This simulates unpredictable variations in energy usage
df['energy_consumption'] += np.random.normal(0, 5, size=len(df))

# Generate temperature data (Celsius)
# We simulate temperature with a seasonal trend and daily fluctuations
df['temperature'] = 20 + 10 * np.sin(np.arange(len(df)) * (2 * np.pi / (24 * 2 * 90)))  # Seasonal pattern
df['temperature'] += 5 * np.sin(np.arange(len(df)) * (2 * np.pi / (24 * 2)))  # Daily pattern
df['temperature'] += np.random.normal(0, 1, size=len(df))  # Random fluctuations

# Generate humidity data (%)
# Humidity is simulated with seasonal and daily patterns, plus random variations
df['humidity'] = 60 + 20 * np.sin(np.arange(len(df)) * (2 * np.pi / (24 * 2 * 90)))  # Seasonal pattern
df['humidity'] += 10 * np.sin(np.arange(len(df)) * (2 * np.pi / (24 * 2)))  # Daily pattern
df['humidity'] += np.random.normal(0, 3, size=len(df))  # Random fluctuations
df['humidity'] = df['humidity'].clip(0, 100)  # Ensure humidity is between 0 and 100%

# Add anomalies to energy consumption
# This simulates unusual events or malfunctions in the HVAC system
anomaly_indices = np.random.choice(len(df), size=20, replace=False)
df.loc[df.index[anomaly_indices], 'energy_consumption'] *= np.random.uniform(1.5, 2.5, size=20)

# Introduce missing data
# This simulates sensor failures or communication issues
missing_indices = np.random.choice(len(df), size=100, replace=False)
df.loc[df.index[missing_indices], :] = np.nan

# Simulate longer sensor failures
# This represents extended periods of downtime or maintenance
for _ in range(2):
    start_idx = np.random.randint(0, len(df) - 24*2)  # 24 hours of 30-minute intervals
    df.iloc[start_idx:start_idx+24*2, :] = np.nan # <-- Example usage of iloc

# Add noise to timestamps
# This simulates delays or inconsistencies in data transmission common in IoT systems
#
# In real-world IoT deployments, several factors can cause timestamp inconsistencies:
# 1. Network latency: Data packets may take varying amounts of time to reach the central system.
# 2. Device clock drift: IoT sensors' internal clocks may drift over time, causing timestamp inaccuracies.
# 3. Processing delays: Time taken for data processing and aggregation can vary.
# 4. Batch uploads: Some systems might store data locally and upload in batches, causing delays.

# We use a normal distribution to generate the noise:
# - np.random.normal(0, 30, size=len(df)) generates random numbers from a normal distribution
#   with mean 0 and standard deviation 30 (seconds).
# - Mean of 0 ensures that delays are centered around the original timestamp.
# - Standard deviation of 30 seconds means about 68% of the noise will be within ±30 seconds,
#   95% within ±60 seconds, and 99.7% within ±90 seconds.

# pd.to_timedelta() converts these numbers to time deltas that can be added to timestamps.
# The 'unit='s'' parameter specifies that our random numbers represent seconds.
noisy_timestamps = df.index + pd.to_timedelta(np.random.normal(0, 30, size=len(df)), unit='s')

# We then replace the original timestamps with these noisy versions.
# This step simulates the real-world scenario where the recorded timestamp
# might differ slightly from the actual time of data collection.
df.index = noisy_timestamps

# Note: After adding this noise, the data may no longer be in strict chronological order.
# In real-world data analysis, this out-of-order data would require additional preprocessing
# steps, such as sorting or using time-window-based operations, to ensure accurate analysis.

# Sort the DataFrame by the new timestamps
# Ensures data is in chronological order after adding noise
df.sort_index(inplace=True)

# Reset index to make timestamp a column
# This makes the data easier to work with in some analysis scenarios
df.reset_index(inplace=True)
df.rename(columns={'index': 'timestamp'}, inplace=True)

# Save to CSV
# We save the data for later use in our analysis pipeline
df.to_csv('hvac_iot_data_small.csv', index=False)

# Print summary statistics
# This gives us a quick overview of the generated dataset
print("HVAC IoT data generated and saved to 'hvac_iot_data_small.csv'")
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Missing data points: {df.isnull().sum().sum()}")
print(f"Sample of the data:\n{df.head()}")