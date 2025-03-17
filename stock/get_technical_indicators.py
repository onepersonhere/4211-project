import pandas as pd
import numpy as np
import os

# Function to process the stock data and calculate the indicators
def process_stock_data(file_name):
    df = pd.read_csv(f'./stock/data_raw/{file_name}.csv')

    # If a Date column exists, parse it as datetime and sort by time
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    # 1. MACD Calculation
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']

    # 2. RSI Calculation (Period = 14)
    delta = df['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = loss.abs()

    window_length = 14
    avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. KDJ Calculation (Period = 9)
    period = 9
    df['Low_Min'] = df['Low'].rolling(window=period, min_periods=period).min()
    df['High_Max'] = df['High'].rolling(window=period, min_periods=period).max()
    df['RSV'] = (df['Close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min']) * 100

    df['K'] = np.nan
    df['D'] = np.nan
    df['J'] = np.nan

    for i in range(len(df)):
        if np.isnan(df.loc[i, 'RSV']):
            continue
        if i == df['RSV'].first_valid_index():
            df.loc[i, 'K'] = 50
            df.loc[i, 'D'] = 50
        else:
            df.loc[i, 'K'] = (2/3) * df.loc[i-1, 'K'] + (1/3) * df.loc[i, 'RSV']
            df.loc[i, 'D'] = (2/3) * df.loc[i-1, 'D'] + (1/3) * df.loc[i, 'K']
        df.loc[i, 'J'] = 3 * df.loc[i, 'K'] - 2 * df.loc[i, 'D']

    # 4. OSC Calculation: Price Oscillator
    sma_fast = df['Close'].rolling(window=5).mean()
    sma_slow = df['Close'].rolling(window=10).mean()
    df['OSC'] = (sma_fast - sma_slow) / sma_slow * 100

    # 5. BOLL Calculation: Bollinger Bands
    boll_period = 20
    df['BOLL_Mid'] = df['Close'].rolling(window=boll_period).mean()
    df['BOLL_STD'] = df['Close'].rolling(window=boll_period).std()
    df['BOLL_Upper'] = df['BOLL_Mid'] + 2 * df['BOLL_STD']
    df['BOLL_Lower'] = df['BOLL_Mid'] - 2 * df['BOLL_STD']

    # 6. BIAS Calculation: Price Bias Indicator
    bias_period = 6
    sma_bias = df['Close'].rolling(window=bias_period).mean()
    df['BIAS'] = (df['Close'] - sma_bias) / sma_bias * 100

    # Display or save results
    print(df.tail())

    output_file_path = f'./stock/data_with_indicators/{file_name}_with_indicators.csv'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save the DataFrame to the specified path
    df.to_csv(output_file_path, index=False)

    print(f"Processed data saved to {output_file_path}")

# Process all CSV files in the /data_raw directory
def process_all_files():
    input_dir = './stock/data_raw'

    # List all files in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            # Process each file
            process_stock_data(file_name.replace('.csv', ''))  # Remove '.csv' and pass the name

# Run the process
if __name__ == "__main__":
    process_all_files()
