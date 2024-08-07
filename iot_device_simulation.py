import asyncio
import aiohttp
import uvicorn
from fastapi import FastAPI
import random
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel
import nest_asyncio

# Create FastAPI server to simulate IoT sensors
app = FastAPI()


# Define data model for sensor readings
class SensorData(BaseModel):
    id: int
    temperature: float
    humidity: float
    energy_consumption: float
    active: bool


# Endpoint to generate random sensor data
@app.get("/sensor/{sensor_id}", response_model=SensorData)
async def sensor_data(sensor_id: int):
    return SensorData(
        id=sensor_id,
        temperature=round(random.uniform(18, 28), 1),  # Simulate temperature between 18-28°C
        humidity=round(random.uniform(30, 70), 1),  # Simulate humidity between 30-70%
        energy_consumption=round(random.uniform(0.5, 5.0), 2),  # Simulate energy use between 0.5-5.0 units
        active=random.choice([True, False])  # Randomly set sensor as active or inactive
    )


# Function to fetch data from a single sensor
async def fetch_sensor_data(session: aiohttp.ClientSession, sensor_id: int) -> Dict:
    url = f"http://localhost:8000/sensor/{sensor_id}"
    async with session.get(url) as response:
        return await response.json()


# Collect data from all building sensors concurrently
async def collect_building_data(sensor_ids: List[int]) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_sensor_data(session, sensor_id) for sensor_id in sensor_ids]
        results = await asyncio.gather(*tasks)

    return pd.DataFrame(results)


# Process collected data to generate summary statistics
async def process_building_data(df: pd.DataFrame) -> Dict[str, float]:
    await asyncio.sleep(1)  # Simulate time-consuming data processing
    return {
        'avg_temperature': df['temperature'].mean(),
        'total_energy': df['energy_consumption'].sum(),
        'num_active_sensors': df['active'].sum()
    }


# Main function to orchestrate data collection and processing
async def main():
    sensor_ids = list(range(1, 11))  # Simulate 10 sensors

    df = await collect_building_data(sensor_ids)
    results = await process_building_data(df)

    print("Building Data Summary:")
    print(results)
    print("\nRaw Data:")
    print(df)


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()  # Allow asyncio to run in Jupyter notebooks

    # Run FastAPI server in a separate thread
    import threading

    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000}, daemon=True).start()

    # Brief pause to ensure server is running
    import time

    time.sleep(1)

    # Execute the main asyncio event loop
    asyncio.run(main())
