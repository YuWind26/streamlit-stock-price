import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.indicators import add_technical_indicators
from src.strategies import apply_rsi_strategy
from src.analysis import calculate_basic_stats
from src.models import prepare_ml_data, train_eval_models

def run_verification():
    print("Generating Mock Data...")
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(100, 1000, 100)
    })
    df.set_index('Date', inplace=True)
    
    print("Testing Indicators...")
    df = add_technical_indicators(df)
    if 'RSI' in df.columns:
        print("PASS: RSI calculated")
    else:
        print("FAIL: RSI missing")
        
    print("Testing Strategy...")
    df = apply_rsi_strategy(df)
    if 'Signal_RSI' in df.columns:
        print("PASS: RSI Strategy applied")
        
    print("Testing Analysis...")
    _, summary = calculate_basic_stats(df)
    print("PASS: Statistics calculated:", summary)
    
    print("Testing ML Model (Random Forest)...")
    try:
        model, metrics, preds, _, _ = train_eval_models(df, model_type='Random Forest', task_type='Regression')
        print("PASS: Random Forest trained. Metrics:", metrics)
    except Exception as e:
        print("FAIL: ML Training error:", e)
        
    print("Verification Complete.")

if __name__ == "__main__":
    run_verification()
