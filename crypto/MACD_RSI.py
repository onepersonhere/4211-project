import pandas as pd
import talib
import os

#install talib using pip3 install TA-Lib

def calculate_macd_rsi_and_save(input_folder, output_folder, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14):
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        if "Close" not in df.columns:
            print(f"Skipping {file_path}: 'Close' column not found")
            continue

        # Compute MACD
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(df["Close"], 
                                                                    fastperiod=macd_fast, 
                                                                    slowperiod=macd_slow, 
                                                                    signalperiod=macd_signal)

        # Compute RSI
        df["RSI"] = talib.RSI(df["Close"], timeperiod=rsi_period)

        # Generate output filename (e.g., BTC_1m_data_indicators.csv)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        output_file = os.path.join(output_folder, f"{base_name}_indicators.csv")

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

input_folder = "./market_data"
output_folder = "./processed_data"
file_paths = ["BTC_1m_data.csv", "ETH_1m_data.csv", "XRP_1m_data.csv", "SOL_1m_data.csv"]
calculate_macd_rsi_and_save(input_folder, output_folder)
