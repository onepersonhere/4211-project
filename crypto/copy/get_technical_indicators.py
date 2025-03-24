import pandas as pd
import numpy as np
import os
import shutil
import zipfile

# Function to process the stock data and calculate the indicators
def process_crypto_data(script_dir, file_name):

    raw_folder = os.path.join(script_dir, "data_raw")
    combined_df = pd.read_csv(os.path.join(raw_folder, f'{file_name}.csv'))

    # If a Date column exists, parse it as datetime and sort by time
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df.sort_values('Date', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    
    new_combined_df_lst = []
    coins = ['BTC', 'ETH', 'SOL', 'XRP']
    for coin in coins:
        df = combined_df[combined_df['Coin'] == coin]
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_Buy'] = df['EMA9'] > df['EMA21']
        df['EMA_Sell'] = df['EMA9'] < df['EMA21']

        df['MACD'] = df['EMA9'] - df['EMA21']
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
        new_combined_df_lst.append(df)
    
    new_combined_df = pd.concat(new_combined_df_lst, ignore_index=True)

    output_dir = os.path.join(script_dir, "data_with_indicators")
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'{file_name}_with_indicators.zip')
    # Save data directly into a ZIP file without creating an extra CSV file
    with zipfile.ZipFile(output_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        csv_data = new_combined_df.to_csv(index=False).encode('utf-8')
        zipf.writestr(f'{file_name}_with_indicators.csv', csv_data)

    print(f"Processed data saved as ZIP: {output_file_path}")



# Function to zip each cryptocurrency's data separately
def zip_and_split_crypto_files(output_dir):
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.csv'):
            crypto_file_path = os.path.join(output_dir, file_name)
            zip_file_path = os.path.join(output_dir, file_name.replace('.csv', '.zip'))
            
            # Create a ZIP file for each crypto CSV
            shutil.make_archive(zip_file_path[:-4], 'zip', output_dir, file_name)

# Process all CSV files in the /data_raw directory
def process_all_files():
    #Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Unzip folders if not unzipped already
    # Define the path to the zip file and the destination folder
    zip_file = os.path.join(script_dir, 'data_raw.zip')
    
    extract_to = os.path.join(script_dir, 'data_raw')
    # Ensure the extraction directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Ensure extraction only happens if the folder is missing or empty
    if not os.path.exists(extract_to) or not os.listdir(extract_to):  
        shutil.unpack_archive(zip_file, extract_to)
        print(f"Unzipped to: {extract_to}")
    else:
        print("Skipping Extraction...")
    
    input_dir = os.path.join(script_dir, "data_raw")

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            process_crypto_data(script_dir, file_name.replace('.csv', ''))

    
    # Define the directory containing the modified CSVs (now "data_with_indicators")
    directory = os.path.join(script_dir, "data_with_indicators")

    # After saving all CSVs, zip every csv
    zip_and_split_crypto_files(directory)
    print(f"Zipped data in {directory}")
# Run the process
if __name__ == "__main__":
    process_all_files()
