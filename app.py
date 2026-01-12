import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.metrics import confusion_matrix, classification_report

# Import custom modules
from src.data import fetch_yfinance_data, fetch_finmind_data
from src.indicators import add_technical_indicators
from src.strategies import apply_rsi_strategy, apply_ma_crossover_strategy
from src.analysis import calculate_basic_stats
from src.models import train_eval_models, train_lstm_model
import src.models

# Page Config
st.set_page_config(page_title="Final Exam: Financial Time Series", layout="wide")

st.title("Final Exam: Financial Time Series Analysis & Prediction")

# Sidebar
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["Yahoo Finance", "FinMind (Taiwan Stocks)"])

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'ticker_name' not in st.session_state:
    st.session_state['ticker_name'] = "Asset"

# Data Loading logic
df = None

if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")
    start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365*2))
    end_date = st.sidebar.date_input("End Date", value=date.today())
    
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            fetched_df = fetch_yfinance_data(ticker, start_date, end_date)
            if fetched_df is not None:
                st.session_state['data'] = fetched_df
                st.session_state['ticker_name'] = ticker
            else:
                st.error("No data found.")

elif data_source == "FinMind (Taiwan Stocks)":
    ticker = st.sidebar.text_input("Stock ID", value="2330") # TSMC default
    start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365*2))
    end_date = st.sidebar.date_input("End Date", value=date.today())
    
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching FinMind data..."):
            fetched_df = fetch_finmind_data(ticker, start_date, end_date)
            if fetched_df is not None:
                st.session_state['data'] = fetched_df
                st.session_state['ticker_name'] = f"TW-{ticker}"
            else:
                st.error("No data found.")

# Retrieve from session state
df = st.session_state['data']
ticker_name = st.session_state['ticker_name']

# Main Logic
if df is not None:
    # ensure Date is datetime and set as index or column
    # If using mplfinance, it often needs Date index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        # For session state persistence, sometimes index is lost or reset.
        # It's safer to ensure we have Date column then set index.
        df.set_index('Date', inplace=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Chart & Indicators", "Strategies", "Analysis", "ML/DL Prediction"])
    
    # Add Indicators for all tabs
    df = add_technical_indicators(df)
    
    with tab1:
        st.subheader(f"{ticker_name} Price Chart")
        
        chart_type = st.radio("Chart Type", ["Plotly (Interactive)", "mplfinance (Detailed T.A.)"], horizontal=True)
        
        if chart_type == "Plotly (Interactive)":
            # Interactive Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index,
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'], name='OHLC'))
            
            # User selects indicators to show
            show_sma = st.checkbox("Show SMA 20/50")
            if show_sma:
                if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
                if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
                
            st.plotly_chart(fig, use_container_width=True)
            
            # Indicator Subplots (Plotly)
            st.subheader("Indicators")
            ind_fig = go.Figure()
            if 'RSI' in df.columns:
                ind_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
                ind_fig.add_hline(y=70, line_dash="dash", line_color="red")
                ind_fig.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(ind_fig, use_container_width=True)
            
        else: # mplfinance
            # mplfinance requires specific columns and distinct index
            # df index is already Date
            # Create a style
            st.write("Generating mplfinance chart...")
            
            # We need to pass data to mplfinance
            # Filter last N days to make it readable if too long
            # Let's show last 150 days by default or slider
            days_to_show = st.slider("Days to show", 30, 365, 100)
            plot_data = df.tail(days_to_show)
            
            # Add plots (SMA)
            ap = []
            if 'SMA_20' in plot_data.columns:
                ap.append(mpf.make_addplot(plot_data['SMA_20'], color='blue'))
            if 'SMA_50' in plot_data.columns:
                ap.append(mpf.make_addplot(plot_data['SMA_50'], color='orange'))
                
            fig, ax = mpf.plot(plot_data, type='candle', style='yahoo', 
                             addplot=ap,
                             volume=True, 
                             returnfig=True,
                             title=f"{ticker_name} - Last {days_to_show} Days")
            st.pyplot(fig)

    with tab2:
        st.subheader("Rule-Based Strategies")
        
        strat = st.selectbox("Select Strategy", ["RSI Strategy", "MA Crossover"])
        
        if strat == "RSI Strategy":
            df = apply_rsi_strategy(df)
            st.write("Buy Signal: RSI < 30 | Sell Signal: RSI > 70")
            if 'Signal_RSI' in df.columns:
                buy_signals = df[df['Signal_RSI'] == 1]
                sell_signals = df[df['Signal_RSI'] == -1]
                
                st.write(f"Total Buy Signals: {len(buy_signals)}")
                st.write(f"Total Sell Signals: {len(sell_signals)}")
                
                # Visualize Signals (Plotly for interactivity)
                sig_fig = go.Figure()
                sig_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
                sig_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=10, name='Buy'))
                sig_fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10, name='Sell'))
                st.plotly_chart(sig_fig, use_container_width=True)
            
        elif strat == "MA Crossover":
            df = apply_ma_crossover_strategy(df)
            st.write("Golden Cross (Short > Long) = Buy | Death Cross (Short < Long) = Sell")
            
            if 'Signal_MA' in df.columns:
                buy_signals = df[df['Signal_MA'] == 1.0]
                sell_signals = df[df['Signal_MA'] == -1.0]
                
                st.write(f"Total Buy Signals: {len(buy_signals)}")
                st.write(f"Total Sell Signals: {len(sell_signals)}")
                
                sig_fig = go.Figure()
                sig_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
                if 'SMA_20' in df.columns: sig_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(dash='dot')))
                if 'SMA_50' in df.columns: sig_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(dash='dot')))
                
                sig_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=10, name='Buy'))
                sig_fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10, name='Sell'))
                st.plotly_chart(sig_fig, use_container_width=True)

    with tab3:
        st.subheader("Statistical Analysis")
        stats_df, summary = calculate_basic_stats(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Price", f"{summary['Latest Price']:.2f}")
        col2.metric("Mean Daily Return", f"{summary['Mean Return']:.4f}")
        col3.metric("Volatility", f"{summary['Volatility (Std)']:.4f}")
        col4.metric("Max Drawdown", f"{summary['Max Drawdown']:.4f}")
        
        st.write("### Rolling Statistics (20 Days)")
        roll_fig = go.Figure()
        if 'Rolling_Mean_20' in stats_df.columns:
            roll_fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df['Rolling_Mean_20'], name='Rolling Mean'))
            roll_fig.add_trace(go.Scatter(x=stats_df.index, y=stats_df['Rolling_Std_20'], name='Rolling Std', yaxis='y2'))
        
        roll_fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volatility", overlaying='y', side='right')
        )
        st.plotly_chart(roll_fig, use_container_width=True)

    with tab4:
        st.subheader("Machine Learning Prediction")
        
        col_m1, col_m2 = st.columns(2)
        model_choice = col_m1.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM (Deep Learning)"])
        task_choice = col_m2.selectbox("Prediction Task", ["Regression (Next Day Price)", "Classification (Up/Down)"])
        
        if st.button("Train & Predict"):
            with st.spinner(f"Training {model_choice}..."):
                if model_choice == "LSTM (Deep Learning)":
                    if "Classification" in task_choice:
                        st.warning("LSTM implemented for Regression (Price) only in this demo.")
                    else:
                        model, metrics, preds, test_dates, real_price = train_lstm_model(df)
                        if model:
                            st.success("Training Complete!")
                            st.json(metrics)
                            
                            pred_fig = go.Figure()
                            pred_fig.add_trace(go.Scatter(x=test_dates, y=real_price.flatten(), name='Actual'))
                            pred_fig.add_trace(go.Scatter(x=test_dates, y=preds.flatten(), name='Predicted'))
                            st.plotly_chart(pred_fig, use_container_width=True)
                else:
                    # ML Models
                    task_type = task_choice.split()[0]
                    # train_eval_models returns: model, metrics, preds, test_index, y_test
                    model, metrics, preds, test_index, y_test_vals = train_eval_models(df, model_type=model_choice, task_type=task_type)
                    
                    st.success("Training Complete!")
                    st.json(metrics)
                    
                    if "Regression" in task_choice:
                        pred_fig = go.Figure()
                        pred_fig.add_trace(go.Scatter(x=test_index, y=y_test_vals, name='Actual Target'))
                        pred_fig.add_trace(go.Scatter(x=test_index, y=preds, name='Predicted Target'))
                        st.plotly_chart(pred_fig, use_container_width=True)
                    else:
                        # Classification visualization or metrics
                        st.subheader("Classification Results")
                        # Confusion Matrix
                        cm = confusion_matrix(y_test_vals, preds)
                        st.write("Confusion Matrix:")
                        st.write(cm)
                        
                        # Classification Report
                        st.write("Classification Report:")
                        report = classification_report(y_test_vals, preds, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())

import src.models # Late import helper if needed
