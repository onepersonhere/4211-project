import pandas as pd
import talib
import os

def calculate_macd_rsi_and_save(input_folder, output_folder, macd_fast=12, macd_slow=26, macd_signal=9, rsi_period=14):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of CSV files in the input folder
    file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file_path in file_paths:
        
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

            # Add KDJ 
            # Calculate %K and %D using TA-Lib's STOCH function
            slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=7, slowk_period=3, slowd_period=3)

            # Calculate the %J line manually
            slowj = 3 * slowk - 2 * slowd

            # Add the K, D, and J lines to the dfframe
            df['%K'] = slowk
            df['%D'] = slowd
            df['%J'] = slowj

            # Add BOLL
            upper_band, middle_band, lower_band = talib.BBANDS(df['Close'], timeperiod=14, nbdevup=2, nbdevdn=2, matype=0)

            # Add the bands to the dfFrame for visualization
            df['Upper_Band'] = upper_band
            df['Middle_Band'] = middle_band
            df['Lower_Band'] = lower_band

            # Add ADX
            high_prices = df['High'].values
            low_prices = df['Low'].values
            close_prices = df['Close'].values

            # Calculate ADX with a typical period of 14 (can be adjusted)
            df['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df['+DI'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            df['-DI'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
            # ADX is the main line that measures trend strength
            # +DI and -DI are the directional indicators

            # You can add these values to the dfframe or use them directly for analysis
            

            # Aroon 
            # Calculate Aroon with a typical period of 14 (can be adjusted)
            aroon_up, aroon_down = talib.AROON(high_prices, low_prices, timeperiod=14)

            # Aroon Up and Aroon Down values
            df['Aroon Up'] = aroon_up
            df['Aroon Down'] = aroon_down
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

        
# Set folder paths
input_folder = "market_data/"
output_folder = "processed_data/"

# Run function
calculate_macd_rsi_and_save(input_folder, output_folder)
