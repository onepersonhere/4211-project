import pandas as pd
import numpy as np
import os

# Function to process the stock data and calculate the indicators
def process_stock_data(file_name):
    df = pd.read_csv(f'./data_raw/{file_name}.csv')

    # If a Date column exists, parse it as datetime and sort by time
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 1. MACD Calculation
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    df['MACD_Buy'] = df['MACD'] > df['Signal']
    df['MACD_Sell'] = df['MACD'] < df['Signal']

    # 2. RSI Calculation (Period = 14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    window_length = 14
    avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['RSI_Buy'] = df['RSI'] < 30
    df['RSI_Sell'] = df['RSI'] > 70
    
    # 3. KDJ Calculation (Period = 9)
    period = 9
    df['Low_Min'] = df['Low'].rolling(window=period, min_periods=period).min()
    df['High_Max'] = df['High'].rolling(window=period, min_periods=period).max()
    df['RSV'] = (df['Close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min']) * 100
    
    df['K'] = df['RSV'].ewm(alpha=1/3).mean()
    df['D'] = df['K'].ewm(alpha=1/3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 4. OSC Calculation: Price Oscillator
    sma_fast = df['Close'].rolling(window=5).mean()
    sma_slow = df['Close'].rolling(window=10).mean()
    df['OSC'] = (sma_fast - sma_slow) / sma_slow * 100
    
    df['OSC_Buy'] = df['OSC'] > 0
    df['OSC_Sell'] = df['OSC'] < 0
    
    # 5. BOLL Calculation: Bollinger Bands
    boll_period = 20
    df['BOLL_Mid'] = df['Close'].rolling(window=boll_period).mean()
    df['BOLL_STD'] = df['Close'].rolling(window=boll_period).std()
    df['BOLL_Upper'] = df['BOLL_Mid'] + 2 * df['BOLL_STD']
    df['BOLL_Lower'] = df['BOLL_Mid'] - 2 * df['BOLL_STD']
    
    df['BOLL_Buy'] = df['Close'] <= df['BOLL_Lower']
    df['BOLL_Sell'] = df['Close'] >= df['BOLL_Upper']
    
    # 6. BIAS Calculation: Price Bias Indicator
    bias_period = 6
    sma_bias = df['Close'].rolling(window=bias_period).mean()
    df['BIAS'] = (df['Close'] - sma_bias) / sma_bias * 100
    
    df['BIAS_Buy'] = df['BIAS'] < 0
    df['BIAS_Sell'] = df['BIAS'] > 0
    
    # 7. Stochastic Oscillator (Stochs)
    df['%K'] = df['RSV']
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    df['STOCHS_Buy'] = (df['%K'] > df['%D']) & (df['%K'] < 20)
    df['STOCHS_Sell'] = (df['%K'] < df['%D']) & (df['%K'] > 80)
    
    # 8. ADX Calculation
    adx_period = 14
    df['+DI'] = 100 * (df['High'] - df['Low']).rolling(window=adx_period).mean()
    df['-DI'] = 100 * (df['Low'] - df['High']).rolling(window=adx_period).mean()
    df['ADX'] = (df['+DI'] - df['-DI']).abs().rolling(window=adx_period).mean()
    
    df['ADX_Buy'] = df['ADX'] > 25
    df['ADX_Sell'] = df['ADX'] < 25
    
    # 9. Aroon Calculation
    aroon_period = 25
    df['Aroon_Up'] = 100 * df['High'].rolling(window=aroon_period).apply(lambda x: x.argmax()) / aroon_period
    df['Aroon_Down'] = 100 * df['Low'].rolling(window=aroon_period).apply(lambda x: x.argmin()) / aroon_period
    
    df['Aroon_Buy'] = df['Aroon_Up'] > df['Aroon_Down']
    df['Aroon_Sell'] = df['Aroon_Up'] < df['Aroon_Down']
    
    # Display or save results
    print(df.tail())

    output_file_path = f'./data_with_indicators/{file_name}_with_indicators.csv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"Processed data saved to {output_file_path}")

# Process all CSV files in the /data_raw directory
def process_all_files():
    input_dir = './data_raw'
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            process_stock_data(file_name.replace('.csv', ''))

# Run the process
if __name__ == "__main__":
    process_all_files()
