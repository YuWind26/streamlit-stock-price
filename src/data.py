import yfinance as yf
import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

def fetch_yfinance_data(ticker, start_date, end_date):
    """
    Fetch historical data from Yahoo Finance.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_finmind_data(ticker, start_date, end_date):
    """
    Fetch historical data from FinMind (Taiwan Stocks).
    """
    try:
        # Convert dates to string YYYY-MM-DD
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        dl = DataLoader()
        # FinMind expects dataset='TaiwanStockPrice' for stock prices
        df = dl.taiwan_stock_daily(
            stock_id=ticker,
            start_date=start_str,
            end_date=end_str
        )
        
        if df.empty:
            return None
            
        # Standardize columns to match yfinance: Date, Open, High, Low, Close, Volume
        # FinMind returns: date, stock_id, Trading_Volume, Trading_Money, open, max, min, close, spread, ...
        # Rename:
        # date -> Date, open -> Open, max -> High, min -> Low, close -> Close, Trading_Volume -> Volume
        
        df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'max': 'High',
            'min': 'Low',
            'close': 'Close',
            'Trading_Volume': 'Volume'
        }, inplace=True)
        
        # Ensure date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Keep necessary columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching FinMind data for {ticker}: {e}")
        return None
