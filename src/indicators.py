import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("Warning: pandas_ta not found. Using manual calculation.")

def add_technical_indicators(df):
    """
    Add technical indicators.
    Uses pandas_ta if available (preferred), otherwise manual pandas.
    """
    if df is None or df.empty:
        return df
    
    # Ensure Close is float
    df = df.copy()
    df['Close'] = df['Close'].astype(float)
    
    if ta is not None:
        # Use pandas_ta
        # It appends to df usually if append=True, or returns new df.
        # ta.sma(df['Close'], length=20) returns Series.
        # Strategy: Use ta functions directly assignment
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_100'] = ta.sma(df['Close'], length=100)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            # macd columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            df = pd.concat([df, macd], axis=1)
            # Ensure naming consistency if needed for manual fallback compatibility?
            # Manual uses: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9.
            # pandas_ta uses exactly these usually.
            
        return df

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Better RSI calculation (Wilder's Smoothing) - Standard in TA
    # However, simple rolling mean is often used in simple implementations. 
    # Let's stick to standard Wilder's:
    # avg_gain = (prev_avg_gain * 13 + current_gain) / 14
    # pandas ewm calc with com=13 (roughly) or alpha=1/14
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    # MACD Line = EMA_12 - EMA_26
    df['MACD_12_26_9'] = df['EMA_12'] - df['EMA_26']
    # Signal Line = EMA_9 of MACD Line
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    # Histogram = MACD Line - Signal Line
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    # Map to simpler names if expected by app
    # (The app uses default pandas_ta names or simpler ones? 
    # Checking app.py: it just calls add_technical_indicators.
    # But checking if we use specific column names elsewhere.)
    # In strategies.py: if 'RSI' in... - yes.
    # In app.py plotting: df['RSI'].
    # The MACD plotting in app.py was not fully detailed or used default result. 
    # Let's ensure column names match what pandas_ta produced mostly or update app if needed.
    
    return df
    
def calculate_volatility(df, window=20):
    """
    Calculate rolling volatility (std dev of daily returns).
    """
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=window).std()
    return df
