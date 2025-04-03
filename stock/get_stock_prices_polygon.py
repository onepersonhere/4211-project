import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# === Config ===
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "META"]
API_KEYS = [
    "V9d86wEIFQFc5d5pgS6rMjjKBOlHxEWe",
    "z7YpQ57e94Kf1oibGS8W2YL17Ur0DRfQ",
    "a31nObEyV8EP6bTN2_o8ctNAbnb8Sc6k",
    "UmEbcsHpy3NdqzLRPx0_nuRxS3RHeMgQ",
    "GFF8Uj9VwSVcvv9YnFyoWwsy2c5Y8WIt"
]

START_DATE = datetime.strptime("2025-01-01", "%Y-%m-%d")
END_DATE = datetime.strptime("2025-03-31", "%Y-%m-%d")
CHUNK_DAYS = 30
MAX_ROWS = 50000
SAVE_PATH = "./stock/data_raw"

os.makedirs(SAVE_PATH, exist_ok=True)

# === Fetch from Polygon ===
def fetch_polygon_data(ticker, api_key, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": MAX_ROWS,
        "apiKey": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "results" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["results"])
        df["Datetime"] = pd.to_datetime(df["t"], unit='ms')
        df["Ticker"] = ticker
        df = df.rename(columns={
            "o": "Open", "h": "High", "l": "Low",
            "c": "Close", "v": "Volume"
        })[["Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker"]]
        return df
    except Exception as e:
        print(f"Error fetching {ticker} from {from_date} to {to_date}: {e}")
        return pd.DataFrame()

def fetch_chunk_for_ticker(ticker, api_key, start, end):
    from_date = start.strftime("%Y-%m-%d")
    to_date = end.strftime("%Y-%m-%d")
    print(f"Fetching {ticker} from {from_date} to {to_date}")
    return fetch_polygon_data(ticker, api_key, from_date, to_date)

def fetch_minute_data_all_tickers():
    all_data = []

    for i in range(0, len(TICKERS), 4):
        batch_tickers = TICKERS[i:i + 4]
        api_keys = API_KEYS[:len(batch_tickers)]
        pointers = {ticker: START_DATE for ticker in batch_tickers}
        done = {ticker: False for ticker in batch_tickers}

        while not all(done.values()):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                for j, ticker in enumerate(batch_tickers):
                    if done[ticker]:
                        continue

                    current_start = pointers[ticker]
                    current_end = min(current_start + timedelta(days=CHUNK_DAYS), END_DATE)

                    if current_start >= END_DATE:
                        done[ticker] = True
                        continue

                    futures.append(executor.submit(
                        fetch_chunk_for_ticker,
                        ticker, api_keys[j], current_start, current_end
                    ))

                    pointers[ticker] = current_end
                    if current_end >= END_DATE:
                        done[ticker] = True

                for future in futures:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)

            print(f"Sleeping 60 seconds after batch of {len(batch_tickers)} tickers...\n")
            time.sleep(60)

    final_df = pd.concat(all_data, ignore_index=True)
    minute_file = os.path.join(SAVE_PATH, f"stock_prices_minute.csv")
    final_df.to_csv(minute_file, index=False)
    print(f"✅ Saved full 'minute' dataset to {minute_file}")
    return final_df

# === Resample to lower frequency ===
def resample_data(df, frequency):
    resampled_list = []
    for ticker in df["Ticker"].unique():
        sub_df = df[df["Ticker"] == ticker].copy()
        sub_df.set_index("Datetime", inplace=True)
        resampled = sub_df.resample(frequency).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna().reset_index()
        resampled["Ticker"] = ticker
        resampled_list.append(resampled)
    return pd.concat(resampled_list, ignore_index=True)

# === Main ===
if __name__ == "__main__":
    minute_df = fetch_minute_data_all_tickers()

    resample_intervals = {
        "2minute": "2min",
        "5minute": "5min",
        "15minute": "15min"
    }

    for name, freq in resample_intervals.items():
        print(f"🔄 Resampling to {name}...")
        df_resampled = resample_data(minute_df, freq)
        file_path = os.path.join(SAVE_PATH, f"stock_prices_{name}.csv")
        df_resampled.to_csv(file_path, index=False)
        print(f"✅ Saved resampled {name} data to {file_path}")
