import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM training.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_arima_model(train_data):
    """
    Trains an ARIMA model using auto_arima.
    """
    print("Finding optimal ARIMA parameters...")
    model_arima_auto = pm.auto_arima(train_data, seasonal=False, suppress_warnings=True, stepwise=True, trace=False)
    p, d, q = model_arima_auto.order
    
    print(f"Optimal ARIMA parameters: {model_arima_auto.order}")
    model_arima = ARIMA(train_data, order=(p, d, q))
    model_arima_fit = model_arima.fit()
    
    return model_arima_fit

def train_lstm_model(train_data, seq_length, epochs=50, batch_size=32):
    """
    Builds, trains, and saves an LSTM model.
    """
    print("Training LSTM model...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    X_train, y_train = create_sequences(scaled_train_data, seq_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping], verbose=0)
    
    model.save('../data/processed/lstm_model.keras')
    print("LSTM model trained and saved successfully.")
    
    return model, scaler

def forecast_lstm(model, last_sequence, scaler, forecast_days):
    """
    Generates a multi-step forecast using a trained LSTM model.
    """
    forecast_list = []
    current_sequence = last_sequence.copy()
    
    for _ in range(forecast_days):
        prediction = model.predict(current_sequence.reshape(1, len(current_sequence), 1), verbose=0)[0][0]
        forecast_list.append(prediction)
        current_sequence = np.append(current_sequence[1:], [[prediction]], axis=0)

    forecast_prices = scaler.inverse_transform(np.array(forecast_list).reshape(-1, 1))
    return forecast_prices