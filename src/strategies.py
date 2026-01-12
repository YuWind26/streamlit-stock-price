import pandas as pd
import numpy as np

def apply_rsi_strategy(df, buy_threshold=30, sell_threshold=70):
    """
    Apply RSI Strategy:
    Buy when RSI < buy_threshold
    Sell when RSI > sell_threshold
    """
    if 'RSI' not in df.columns:
        return df
        
    df['Signal_RSI'] = 0
    # 1 for Buy, -1 for Sell
    df.loc[df['RSI'] < buy_threshold, 'Signal_RSI'] = 1
    df.loc[df['RSI'] > sell_threshold, 'Signal_RSI'] = -1
    return df

def apply_ma_crossover_strategy(df, short_window=20, long_window=50):
    """
    Apply Moving Average Crossover Strategy using SMA by default.
    Buy when Short MA crosses above Long MA (Golden Cross).
    Sell when Short MA crosses below Long MA (Death Cross).
    """
    short_col = f'SMA_{short_window}'
    long_col = f'SMA_{long_window}'
    
    if short_col not in df.columns or long_col not in df.columns:
        return df
        
    df['Signal_MA'] = 0
    df['Position'] = 0.0
    
    # Create signals
    # Use numpy where to detect crossover
    # 1 if Short > Long, 0 otherwise
    df['MA_Signal_Cond'] = np.where(df[short_col] > df[long_col], 1.0, 0.0)
    
    # Take difference to find crossover points
    # 1.0 - 0.0 = 1.0 (Buy)
    # 0.0 - 1.0 = -1.0 (Sell)
    df['Signal_MA'] = df['MA_Signal_Cond'].diff()
    
    return df
