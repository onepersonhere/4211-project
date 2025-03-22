import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Download Bitcoin data using yfinance
def download_crypto_data(crypto_symbol='BTC-USD', period="2y"):
    """Download cryptocurrency data using Yahoo Finance API"""
    print(f"Downloading {crypto_symbol} data...")
    data = yf.download(crypto_symbol, period=period)
    return data

# Data preprocessing
def preprocess_data(data, feature_columns=['Close'], target_column='Close', 
                    sequence_length=60, train_split=0.8):
    """
    Preprocess the data for time series prediction
    - Normalize the data
    - Create sequences
    - Split into train and test sets
    """
    # Select features
    data_featured = data[feature_columns].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_featured)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length, feature_columns.index(target_column)])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Get corresponding dates for visualization
    dates = data.index[sequence_length:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler, train_dates, test_dates

# Define Transformer model
def create_transformer_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, dropout=0.25):
    """Create a Transformer model for time series prediction"""
    
    # Define input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Transformer layers
    x = inputs
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # Feed forward network
        ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(inputs.shape[-1]),
        ])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
    
    # Global average pooling and output layer
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Train the model
def train_model(X_train, y_train, input_shape, epochs=50, batch_size=32):
    """Train the transformer model"""
    model = create_transformer_model(input_shape)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='mean_squared_error')
    
    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

# Make predictions
def make_predictions(model, X_train, X_test, scaler, feature_columns, target_column):
    """Make predictions using the trained model"""
    # Predict on training data
    train_predictions = model.predict(X_train)
    
    # Predict on test data
    test_predictions = model.predict(X_test)
    
    # Inverse transform the predictions
    # Create dummy arrays for inverse transform
    train_predictions_dummy = np.zeros((len(train_predictions), len(feature_columns)))
    test_predictions_dummy = np.zeros((len(test_predictions), len(feature_columns)))
    
    # Put predictions in the target column position
    train_predictions_dummy[:, feature_columns.index(target_column)] = train_predictions.flatten()
    test_predictions_dummy[:, feature_columns.index(target_column)] = test_predictions.flatten()
    
    # Inverse transform
    train_predictions = scaler.inverse_transform(train_predictions_dummy)[:, feature_columns.index(target_column)]
    test_predictions = scaler.inverse_transform(test_predictions_dummy)[:, feature_columns.index(target_column)]
    
    return train_predictions, test_predictions

# Visualize predictions
def visualize_predictions(data, train_dates, test_dates, train_predictions, test_predictions, 
                         feature_columns, target_column, crypto_symbol):
    """Visualize the predictions against actual values"""
    # Get actual values
    actual_values = data[target_column].values
    
    # Create dates for the entire dataset
    sequence_length = 60  # Should match the sequence_length in preprocessing
    actual_dates = data.index[sequence_length:]
    
    plt.figure(figsize=(14, 7))
    
    # Plot training data
    plt.plot(train_dates, actual_values[:len(train_dates)], 'b-', 
             label=f'Actual {crypto_symbol} Price', linewidth=1)
    plt.plot(train_dates, train_predictions, 'g-', 
             label='Training Predictions', linewidth=1, alpha=0.8)
    
    # Plot test data
    plt.plot(test_dates, actual_values[len(train_dates):len(train_dates)+len(test_dates)], 'b-', 
             linewidth=1)
    plt.plot(test_dates, test_predictions, 'r-', 
             label='Test Predictions', linewidth=2)
    
    # Add vertical line to separate training and test data
    plt.axvline(x=train_dates[-1], color='k', linestyle='--', 
                label='Train/Test Split')
    
    # Customize plot
    plt.title(f'Transformer Model Predictions for {crypto_symbol} Price ({target_column})')
    plt.xlabel('Date')
    plt.ylabel(f'{target_column} Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    return plt

# Evaluate model performance
def evaluate_model(actual_values, predictions):
    """Calculate evaluation metrics for the model"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Calculate MSE, RMSE, MAE
    mse = mean_squared_error(actual_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, rmse, mae, r2

# Main function
def main():
    # Set parameters
    crypto_symbol = 'BTC-USD'  # Can be changed to any crypto symbol available on Yahoo Finance
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_column = 'Close'
    sequence_length = 60  # 60 days of historical data for prediction
    
    # Download data
    data = download_crypto_data(crypto_symbol=crypto_symbol, period="2y")
    print(f"Downloaded {len(data)} days of {crypto_symbol} data")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, train_dates, test_dates = preprocess_data(
        data, feature_columns, target_column, sequence_length)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model, history = train_model(X_train, y_train, input_shape, epochs=50)
    
    # Make predictions
    train_predictions, test_predictions = make_predictions(
        model, X_train, X_test, scaler, feature_columns, target_column)
    
    # Get actual values for evaluation
    target_idx = feature_columns.index(target_column)
    y_train_actual = scaler.inverse_transform(
        np.hstack([np.zeros((len(y_train), target_idx)), 
                  y_train.reshape(-1, 1), 
                  np.zeros((len(y_train), len(feature_columns)-target_idx-1))])
    )[:, target_idx]
    
    y_test_actual = scaler.inverse_transform(
        np.hstack([np.zeros((len(y_test), target_idx)), 
                  y_test.reshape(-1, 1), 
                  np.zeros((len(y_test), len(feature_columns)-target_idx-1))])
    )[:, target_idx]
    
    # Evaluate model
    print("\nTraining Data Evaluation:")
    evaluate_model(y_train_actual, train_predictions)
    
    print("\nTest Data Evaluation:")
    test_metrics = evaluate_model(y_test_actual, test_predictions)
    
    # Visualize predictions
    plt = visualize_predictions(data, train_dates, test_dates, 
                              train_predictions, test_predictions, 
                              feature_columns, target_column, crypto_symbol)
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Return important objects for further analysis if needed
    return model, history, data, test_metrics, train_predictions, test_predictions

# Run the main function
if __name__ == "__main__":
    model, history, data, test_metrics, train_preds, test_preds = main()