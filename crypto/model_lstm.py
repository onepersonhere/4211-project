import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import matplotlib.dates as mdates

# Function to download cryptocurrency data
def download_crypto_data(ticker='BTC-USD', period='2y'):
    """
    Download historical cryptocurrency data from Yahoo Finance
    
    Parameters:
    ticker (str): Ticker symbol (e.g., 'BTC-USD', 'ETH-USD')
    period (str): Time period to download (e.g., '1y', '2y', 'max')
    
    Returns:
    pd.DataFrame: DataFrame with historical data
    """
    data = yf.download(ticker, period=period)
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)
    return data

# Function to prepare data for LSTM model
def prepare_data(data, feature='Close', sequence_length=60, train_size=0.8):
    """
    Prepare data for LSTM model
    
    Parameters:
    data (pd.DataFrame): DataFrame with historical price data
    feature (str): Column name to use as feature (e.g., 'Close', 'Adj Close')
    sequence_length (int): Number of previous time steps to use for prediction
    train_size (float): Proportion of data to use for training
    
    Returns:
    tuple: (X_train, y_train, X_test, y_test, scaler, train_dates, test_dates)
    """
    # Extract feature column and convert to numpy array
    dataset = data[feature].values.reshape(-1, 1)
    
    # Store corresponding dates for later reference
    dates = data.index
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create sequences
    X, y, sequence_dates = [], [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length, 0])
        y.append(scaled_data[i+sequence_length, 0])
        # Store the date corresponding to the target (y) value
        sequence_dates.append(dates[i+sequence_length])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to 3D format expected by LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into training and testing sets
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    train_dates = sequence_dates[:train_size]
    test_dates = sequence_dates[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler, train_dates, test_dates

# Function to build LSTM model
def build_lstm_model(sequence_length, units=50, dropout=0.2, layers=1):
    """
    Build LSTM model
    
    Parameters:
    sequence_length (int): Input sequence length
    units (int): Number of LSTM units
    dropout (float): Dropout rate
    layers (int): Number of LSTM layers
    
    Returns:
    keras.models.Sequential: LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    if layers == 1:
        model.add(LSTM(units=units, input_shape=(sequence_length, 1)))
    else:
        model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(layers-2):
            model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(dropout))
        
        # Final LSTM layer
        model.add(LSTM(units=units))
    
    model.add(Dropout(dropout))
    model.add(Dense(units=1))  # Output layer
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Function to train model and make predictions
def train_and_predict(X_train, y_train, X_test, y_test, model, epochs=50, batch_size=32):
    """
    Train model and make predictions
    
    Parameters:
    X_train, y_train: Training data
    X_test, y_test: Testing data
    model: LSTM model
    epochs (int): Number of training epochs
    batch_size (int): Batch size for training
    
    Returns:
    tuple: (trained_model, train_predictions, test_predictions, history)
    """
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    return model, train_predictions, test_predictions, history

# Function to inverse transform predictions
def inverse_transform_predictions(train_predictions, test_predictions, y_train, y_test, scaler):
    """
    Inverse transform scaled predictions
    
    Parameters:
    train_predictions, test_predictions: Scaled predictions
    y_train, y_test: Scaled actual values
    scaler: Fitted MinMaxScaler
    
    Returns:
    tuple: (train_predictions, test_predictions, train_actual, test_actual)
    """
    # Reshape predictions for inverse transform
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Inverse transform
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    train_actual = scaler.inverse_transform(y_train)
    test_actual = scaler.inverse_transform(y_test)
    
    return train_predictions, test_predictions, train_actual, test_actual

# Function to evaluate model performance
def evaluate_model(train_actual, train_predictions, test_actual, test_predictions):
    """
    Evaluate model performance using various metrics
    
    Parameters:
    train_actual, train_predictions: Actual and predicted values for training data
    test_actual, test_predictions: Actual and predicted values for testing data
    
    Returns:
    dict: Dictionary of performance metrics
    """
    # Calculate metrics
    train_mse = mean_squared_error(train_actual, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_actual, train_predictions)
    
    test_mse = mean_squared_error(test_actual, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_actual, test_predictions)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    train_mape = np.mean(np.abs((train_actual - train_predictions) / train_actual)) * 100
    test_mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
    
    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape
    }

# Function to plot results using datetime
def plot_results(data, train_predictions, test_predictions, train_dates, test_dates, feature='Close'):
    """
    Plot actual vs predicted prices with proper datetime x-axis
    
    Parameters:
    data (pd.DataFrame): Original data with datetime index
    train_predictions, test_predictions: Predicted values
    train_dates, test_dates: Datetime values for predictions
    feature (str): Feature name for plot title
    """
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # Plot historical data
    plt.plot(data.index, data[feature], label='Actual Prices', color='blue')
    
    # Plot predictions
    plt.plot(train_dates, train_predictions, label='Training Predictions', color='green', alpha=0.8)
    plt.plot(test_dates, test_predictions, label='Testing Predictions', color='red', alpha=0.8)
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()  # Rotate date labels
    
    # Add grid, labels and legend
    plt.grid(True, alpha=0.3)
    plt.title(f'Cryptocurrency {feature} Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'{feature} Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Function to predict future prices
def predict_future(model, data, scaler, feature='Close', sequence_length=60, future_days=30):
    """
    Predict future prices
    
    Parameters:
    model: Trained LSTM model
    data (pd.DataFrame): Historical data with datetime index
    scaler: Fitted MinMaxScaler
    feature (str): Feature to predict
    sequence_length (int): Input sequence length used for model
    future_days (int): Number of days to predict into the future
    
    Returns:
    pd.DataFrame: DataFrame with future dates and predicted prices
    """
    # Extract the last sequence_length days of data
    last_sequence = data[feature].values[-sequence_length:].reshape(-1, 1)
    last_sequence = scaler.transform(last_sequence)
    
    # Initialize array for future predictions
    future_predictions = []
    
    # Create the initial input sequence
    current_sequence = last_sequence.reshape(1, sequence_length, 1)
    
    # Predict future days one by one
    for _ in range(future_days):
        # Get prediction for next day
        next_day_pred = model.predict(current_sequence)[0, 0]
        
        # Add prediction to future predictions list
        future_predictions.append(next_day_pred)
        
        # Update sequence: remove first value and add the prediction
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_day_pred]], axis=1)
    
    # Reshape and inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    # Create future dates
    last_date = data.index[-1]
    
    # Generate business days (excluding weekends) if needed
    # future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')
    
    # Or simply generate consecutive days
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    
    # Create DataFrame for future predictions
    future_df = pd.DataFrame({
        f'Predicted {feature}': future_predictions.flatten()
    }, index=future_dates)
    
    return future_df

# Function to plot future predictions with datetime
def plot_future_predictions(data, future_df, feature='Close', ticker='BTC-USD'):
    """
    Plot historical and future predicted prices with proper datetime formatting
    
    Parameters:
    data (pd.DataFrame): Historical data with datetime index
    future_df (pd.DataFrame): Future predictions with datetime index
    feature (str): Feature name for plot title
    ticker (str): Ticker symbol for plot title
    """
    plt.figure(figsize=(16, 8))
    
    # Plot historical data
    plt.plot(data.index, data[feature], label=f'Historical {feature}', color='blue')
    
    # Plot future predictions
    plt.plot(future_df.index, future_df[f'Predicted {feature}'], 
            label=f'Predicted {feature}', color='red', linestyle='--')
    
    # Add vertical line to separate historical from predicted data
    plt.axvline(x=data.index[-1], color='black', linestyle='-', alpha=0.7, 
                label='Prediction Start')
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()  # Rotate date labels
    
    # Add grid, labels and legend
    plt.grid(True, alpha=0.3)
    plt.title(f'{ticker} {feature} Price Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'{feature} Price (USD)', fontsize=14)
    plt.legend(fontsize=12)
    
    # Add annotations for last actual price and first predicted price
    last_actual_price = data[feature].iloc[-1]
    first_pred_price = future_df[f'Predicted {feature}'].iloc[0]
    
    plt.annotate(f'Last actual: ${last_actual_price:.2f}',
                xy=(data.index[-1], last_actual_price),
                xytext=(10, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7))
    
    plt.annotate(f'First prediction: ${first_pred_price:.2f}',
                xy=(future_df.index[0], first_pred_price),
                xytext=(10, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->'),
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# Main function to run the entire pipeline
def run_crypto_prediction(ticker='BTC-USD', period='2y', feature='Close', 
                            sequence_length=60, train_size=0.8,
                            lstm_units=50, dropout=0.2, lstm_layers=2,
                            epochs=50, batch_size=32, future_days=30):
    """
    Run the entire cryptocurrency price prediction pipeline
    
    Parameters:
    ticker (str): Cryptocurrency ticker symbol
    period (str): Time period for historical data
    feature (str): Price feature to predict
    sequence_length (int): Number of previous time steps to use
    train_size (float): Proportion of data for training
    lstm_units (int): Number of LSTM units
    dropout (float): Dropout rate
    lstm_layers (int): Number of LSTM layers
    epochs (int): Training epochs
    batch_size (int): Training batch size
    future_days (int): Number of days to predict into future
    
    Returns:
    tuple: (model, metrics, future_predictions)
    """
    # Download data
    print(f"Downloading {ticker} data for the past {period}...")
    data = download_crypto_data(ticker, period)
    
    # Prepare data
    print("Preparing data for LSTM model...")
    X_train, y_train, X_test, y_test, scaler, train_dates, test_dates = prepare_data(
        data, feature, sequence_length, train_size
    )
    
    # Build model
    print("Building LSTM model...")
    model = build_lstm_model(
        sequence_length, lstm_units, dropout, lstm_layers
    )
    model.summary()
    
    # Train model and make predictions
    print("Training model and making predictions...")
    model, train_predictions, test_predictions, history = train_and_predict(
        X_train, y_train, X_test, y_test, model, epochs, batch_size
    )
    
    # Inverse transform predictions
    train_predictions, test_predictions, train_actual, test_actual = inverse_transform_predictions(
        train_predictions, test_predictions, y_train, y_test, scaler
    )
    
    # Evaluate model
    print("Evaluating model performance...")
    metrics = evaluate_model(train_actual, train_predictions, test_actual, test_predictions)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"Training RMSE: ${metrics['train_rmse']:.2f}")
    print(f"Training MAE: ${metrics['train_mae']:.2f}")
    print(f"Training MAPE: {metrics['train_mape']:.2f}%")
    print(f"Testing RMSE: ${metrics['test_rmse']:.2f}")
    print(f"Testing MAE: ${metrics['test_mae']:.2f}")
    print(f"Testing MAPE: {metrics['test_mape']:.2f}%")
    
    # Plot results with datetime x-axis
    plot_results(data, train_predictions, test_predictions, train_dates, test_dates, feature)
    
    # Predict future prices
    print(f"\nPredicting prices for the next {future_days} days...")
    future_df = predict_future(
        model, data, scaler, feature, sequence_length, future_days
    )
    
    # Plot future predictions with datetime x-axis
    plot_future_predictions(data, future_df, feature, ticker)
    
    # Display future predictions table with formatted dates
    print("\nFuture price predictions:")
    pd.set_option('display.float_format', '${:.2f}'.format)
    print(future_df)
    
    # Save results to CSV
    results_file = f"{ticker.replace('-', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_predictions.csv"
    future_df.to_csv(results_file)
    print(f"\nFuture predictions saved to {results_file}")
    
    return model, metrics, future_df

# Example usage:
if __name__ == "__main__":
    # Run prediction for Bitcoin
    model, metrics, future_predictions = run_crypto_prediction(
        ticker='BTC-USD',
        period='2y',
        feature='Close',
        sequence_length=60,
        train_size=0.8,
        lstm_units=100,
        dropout=0.2,
        lstm_layers=2,
        epochs=50,
        batch_size=32,
        future_days=30
    )