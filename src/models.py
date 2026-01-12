import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import streamlit as st

# Optional: Deep Learning
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except ImportError:
    HAS_TF = False

def prepare_ml_data(df, target_col='Close', horizon=1, lookback=10):
    """
    Prepare data for ML.
    Creates 'Target' column which is the future price (Regression) or Direction (Classification).
    Features include Lags and Indicators.
    """
    data = df.copy()
    
    # 1. Feature Engineering: Lags
    for i in range(1, lookback + 1):
        data[f'Lag_{i}'] = data['Close'].shift(i)
        
    # Drop rows with NaNs created by indicators or lags
    data.dropna(inplace=True)
    
    # 2. Target Creation
    # Regression Target: Price in 'horizon' days
    data['Target_Price'] = data['Close'].shift(-horizon)
    # Classification Target: 1 if Price goes up, 0 otherwise
    data['Target_Direction'] = (data['Target_Price'] > data['Close']).astype(int)
    
    data.dropna(inplace=True)
    
    feature_cols = [c for c in data.columns if c not in ['Date', 'Target_Price', 'Target_Direction', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    # Include some price columns if needed, but usually we use returns or indicators. 
    # For this simple exam, we can use indicators + lags.
    # Let's verify what columns we have. We should use numeric columns only.
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
    
    return data, feature_cols

def train_eval_models(df, model_type='Random Forest', task_type='Regression', test_size=0.2):
    """
    Train and evaluate models.
    """
    data, feature_cols = prepare_ml_data(df)
    
    # Time Series Split
    split_idx = int(len(data) * (1 - test_size))
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    if task_type == 'Regression':
        y_train = train_df['Target_Price']
        y_test = test_df['Target_Price']
    else: # Classification
        y_train = train_df['Target_Direction']
        y_test = test_df['Target_Direction']
        
    model = None
    preds = []
    metrics = {}
    
    if model_type == 'Random Forest':
        if task_type == 'Regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics['Accuracy'] = accuracy_score(y_test, preds)
            
    elif model_type == 'XGBoost':
        if task_type == 'Regression':
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, preds))
        else:
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics['Accuracy'] = accuracy_score(y_test, preds)

    return model, metrics, preds, y_test.index, y_test

def train_lstm_model(df, lookback=60, epochs=20, batch_size=32):
    """
    Train LSTM model. 
    LSTMs require specific 3D shape (Samples, Time Steps, Features).
    """
    if not HAS_TF:
        st.error("TensorFlow/Keras not installed. Cannot run LSTM.")
        return None, {}, None, None

    # Use only Close price for simplicity in this bonus section, or strictly scaled features.
    # To do it right, we scale data.
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # (Samples, Lookback, 1)
    
    # Split
    dataset_size = len(X)
    train_size = int(dataset_size * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Predict next close price scaled
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Predict
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    
    # Get corresponding dates
    test_dates = df.index[lookback:][train_size:]
    
    return model, {'LSTM RMSE': rmse}, predicted_stock_price, test_dates, real_stock_price
