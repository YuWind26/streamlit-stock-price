import pandas as pd

def calculate_basic_stats(df):
    """
    Calculate basic statistics:
    - Daily Returns
    - Rolling Mean/Std (20 days)
    """
    stats_df = df.copy()
    stats_df['Daily_Return'] = stats_df['Close'].pct_change()
    stats_df['Rolling_Mean_20'] = stats_df['Close'].rolling(window=20).mean()
    stats_df['Rolling_Std_20'] = stats_df['Close'].rolling(window=20).std()
    
    # Summary Table
    summary = {
        'Latest Price': stats_df['Close'].iloc[-1],
        'Mean Return': stats_df['Daily_Return'].mean(),
        'Volatility (Std)': stats_df['Daily_Return'].std(),
        'Max Drawdown': (stats_df['Close'] / stats_df['Close'].cummax() - 1).min()
    }
    return stats_df, summary

def calculate_correlation(tickers_data):
    """
    Calculate correlation matrix if multiple assets are loaded.
    tickers_data: dict of {ticker: dataframe}
    """
    close_prices = pd.DataFrame()
    for ticker, df in tickers_data.items():
        if df is not None and not df.empty:
            close_prices[ticker] = df['Close']
            
    return close_prices.corr()
