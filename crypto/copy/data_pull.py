import requests
import time
import pandas as pd
import os
from datetime import datetime
import shutil

def to_milliseconds(date_str):
    return int(time.mktime(time.strptime(date_str, "%Y-%m-%d %H:%M:%S"))) * 1000

# Define timeframe
start_time = to_milliseconds("2025-01-01 00:00:00")  # Start date
today_str = datetime.now().strftime("%Y-%m-%d 00:00:00")
end_time = to_milliseconds(today_str)    # End date, default is today

# Binance API URL
url = "https://api.binance.com/api/v3/klines"

# List of coins
coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


# Maximum limit per request
max_limit = 1000

# Get various intervals
intervals = ['1m', '5m','30m', '1h']
def get_binance_data(interval, coins=coins, url=url, start_time=start_time, end_time=end_time, max_limit=1000):
    all_coins_data =[] # to concat multiple dfs
    # Iterate over each coin
    for coin in coins:
        all_data = []
        current_start_time = start_time
        
        while current_start_time < end_time:
            params = {
                "symbol": coin,
                "interval": interval,  
                "startTime": current_start_time,
                "endTime": end_time,
                "limit": max_limit  # Max limit per request
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    break  # Stop if no more data
                all_data.extend(data)
                
                # Move the start time forward based on the last received timestamp
                current_start_time = data[-1][0] + 1  # Add 1ms to avoid duplicates
            else:
                print(f"Error fetching {coin} data: {response.status_code}, {response.text}")
                break
        
        # Convert response to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"
        ])
        
        # Convert timestamps to readable dates
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
        df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms")
        
        # Add coin column
        df["Coin"] = coin[:-4]  # Remove 'USDT' from symbol
        
        # Keep only relevant columns
        df = df[["Open Time", "Open", "High", "Low", "Close", "Volume", "Number of Trades", "Coin"]]
        # Append the current coin's data to the list
        all_coins_data.append(df)
    return all_coins_data

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output directory at the script's location
output_dir = os.path.join(script_dir, "data_raw")
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Get data for each interval
for interval in intervals:
    interval_data = get_binance_data(interval)
    interval_combined = pd.concat(interval_data, ignore_index=True)
    interval_combined = interval_combined.rename(columns={"Open Time": "Date"})
    
    filename = os.path.join(output_dir, f"crypto_prices_{interval}.csv")
    interval_combined.to_csv(filename, index=False)
    print(f"Saved data to {filename}")

# Define the directory containing the CSVs (now "data_raw")
directory = os.path.join(script_dir, "data_raw")

# After saving all CSVs, zip the folder
zip_filename = os.path.join(script_dir, "data_raw")
shutil.make_archive(zip_filename, 'zip', directory)
print(f"Zipped data to {zip_filename}.zip")

