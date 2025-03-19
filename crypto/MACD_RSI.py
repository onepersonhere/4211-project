import pandas as pd
import talib
import os

def calculate_macd_rsi_and_save(input_folder, output_folder, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of CSV files in the input folder
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)

            # Ensure 'Close' column exists
            if "Close" not in df.columns:
                print(f"Skipping {file_path}: 'Close' column not found")
                continue
            # Compute 9-Period EMA
            df["EMA_9"] = talib.EMA(df["Close"], timeperiod=9)

            # Compute 21-Period EMA
            df["EMA_21"] = talib.EMA(df["Close"], timeperiod=21)
            # Compute MACD
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(
                df["Close"], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
            )

            # Compute RSI
            df["RSI"] = talib.RSI(df["Close"], timeperiod=rsi_period)

            # Fill MACD & Signal Line NaNs using EMA of Close Price
            df["MACD"] = df["MACD"].fillna(df["Close"].ewm(span=macd_fast, adjust=False).mean())
            df["MACD_Signal"] = df["MACD_Signal"].fillna(df["Close"].ewm(span=macd_signal, adjust=False).mean())

            # Recalculate MACD_Hist after filling NaNs
            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

            # Fix RSI missing values:
            # 1. Replace first 14 NaNs with SMA of Close Price for the first 14 periods
            # We use the very first Close price for the first RSI entry
            initial_rsi_sma = df["Close"].rolling(window=rsi_period, min_periods=1).mean()
            df["RSI"] = df["RSI"].combine_first(initial_rsi_sma)

            # 2. Forward-fill any remaining NaNs
            df["RSI"] = df["RSI"].ffill()

            # Generate output filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_folder, f"{base_name}_indicators.csv")

            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Set folder paths
input_folder = "market_data/"
output_folder = "processed_data/"

# Run function
calculate_macd_rsi_and_save(input_folder, output_folder)
