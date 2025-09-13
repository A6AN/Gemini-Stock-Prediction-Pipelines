# pipeline_control_autogluon.py

import pandas as pd
import numpy as np
import argparse
import os
import json
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta
import pandas_market_calendars as mcal

# --- Helper Functions ---

def load_csv(file_path, required_cols=['Date', 'Close']):
    if not file_path or not os.path.exists(file_path):
        print(f"Info: File {file_path} not found. Skipping.")
        return pd.DataFrame()
    # Try reading with no skip first
    df = pd.read_csv(file_path)
    df.columns = [col.capitalize() for col in df.columns]
    if 'Date' not in df.columns:
        # Try skipping first 2 rows (for VIX files with extra info)
        print(f"'Date' column not found in {file_path}. Trying skiprows=2...")
        df = pd.read_csv(file_path, skiprows=2)
        df.columns = [col.capitalize() for col in df.columns]
        if 'Date' not in df.columns:
            raise ValueError(f"'Date' column not found in {file_path} even after skipping 2 rows.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df

def merge_external_feature(main_df, ext_df, col_name, prefix):
    if ext_df.empty or col_name not in ext_df.columns:
        main_df[f"{prefix}_close"] = 0
        main_df[f"{prefix}_log_return"] = 0
        return main_df
    ext_df = ext_df[[col_name]].rename(columns={col_name: f"{prefix}_close"})
    main_df = main_df.merge(ext_df, left_index=True, right_index=True, how='left')
    # FIX: Avoid chained assignment warning by assigning the result
    main_df[f"{prefix}_close"] = main_df[f"{prefix}_close"].ffill()
    main_df[f"{prefix}_close"] = main_df[f"{prefix}_close"].bfill()
    main_df[f"{prefix}_log_return"] = np.log(main_df[f"{prefix}_close"] / main_df[f"{prefix}_close"].shift(1)).fillna(0)
    return main_df

def add_calendar_features(df):
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def get_next_trading_day(last_date, calendar_name='NYSE'):
    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=last_date + timedelta(days=1), end_date=last_date + timedelta(days=10))
    if not schedule.empty:
        return schedule.index[0].date()
    else:
        return (last_date + pd.offsets.BDay(1)).date()

# --- Main Script ---

def main(args):
    # Load main stock data
    train_df = load_csv(args.train_file, required_cols=['Open','High','Low','Close','Volume'])
    test_df = load_csv(args.test_file, required_cols=['Open','High','Low','Close','Volume'])
    if train_df.empty or test_df.empty:
        print("Error: train or test data missing or empty.")
        return

    # Merge all data
    all_df = pd.concat([train_df, test_df]).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep='first')]

    # Add log return as target
    all_df['Target'] = all_df['Close'].shift(-1)
    all_df['Target_Log_Return'] = np.log(all_df['Target'] / all_df['Close'])
    all_df['Close_Log_Return'] = np.log(all_df['Close'] / all_df['Close'].shift(1)).fillna(0)

    # Add calendar features
    all_df = add_calendar_features(all_df)

    # Merge market index
    market_idx_df = load_csv(args.market_index_file)
    all_df = merge_external_feature(all_df, market_idx_df, 'Close', 'market_index')

    # Merge VIX
    vix_df = load_csv(args.vix_file)
    all_df = merge_external_feature(all_df, vix_df, 'Close', 'vix')

    # Merge EPU (optional)
    if args.epu_file:
        epu_df = load_csv(args.epu_file, required_cols=['Close','Value'])
        col = 'Value' if 'Value' in epu_df.columns else 'Close'
        all_df = merge_external_feature(all_df, epu_df, col, 'epu')
    else:
        all_df['epu_close'] = 0
        all_df['epu_log_return'] = 0

    # Merge Commodity (optional)
    if args.commodity_file:
        commodity_df = load_csv(args.commodity_file)
        all_df = merge_external_feature(all_df, commodity_df, 'Close', 'commodity')
    else:
        all_df['commodity_close'] = 0
        all_df['commodity_log_return'] = 0

    # Drop rows with NaN in target
    all_df.dropna(subset=['Target_Log_Return'], inplace=True)

    # Split back to train/test
    test_start = test_df.index.min()
    train_data = all_df[all_df.index < test_start].copy()
    test_data = all_df[all_df.index >= test_start].copy()

    # Prepare features
    feature_cols = [col for col in all_df.columns if col not in ['Target','Target_Log_Return']]
    label_col = 'Target_Log_Return'

    # AutoGluon expects a DataFrame with features + label
    train_ag = train_data[feature_cols + [label_col]].copy()
    test_ag = test_data[feature_cols + [label_col]].copy()

    # Reset index for AutoGluon
    train_ag.reset_index(drop=True, inplace=True)
    test_ag.reset_index(drop=True, inplace=True)

    # Train AutoGluon
    print("Training AutoGluon TabularPredictor...")
    predictor = TabularPredictor(label=label_col, eval_metric='root_mean_squared_error').fit(
        train_ag, time_limit=600, presets='best_quality'
    )

    # Predict on test set
    print("Predicting on test set...")
    y_pred_log_return = predictor.predict(test_ag[feature_cols])
    y_true_log_return = test_ag[label_col].values

    # Convert log returns to price
    test_data = test_data.iloc[:len(y_pred_log_return)].copy()
    test_data['Pred_Log_Return'] = y_pred_log_return.values
    test_data['Pred_Close'] = test_data['Close'] * np.exp(test_data['Pred_Log_Return'])

    # Metrics
    y_true_price = test_data['Target'].values
    y_pred_price = test_data['Pred_Close'].values
    close_t = test_data['Close'].values

    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae = mean_absolute_error(y_true_price, y_pred_price)
    actual_direction = np.sign(y_true_price - close_t)
    pred_direction = np.sign(y_pred_price - close_t)
    actual_direction[actual_direction == 0] = 1
    pred_direction[pred_direction == 0] = 1
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    print(f"\n--- Test Eval (AutoGluon) ---")
    print(f"RMSE: {rmse:.4f} MAE: {mae:.4f} DirAcc: {directional_accuracy:.2f}%")

    # Save metrics
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "directional_accuracy": float(directional_accuracy)
    }
    with open("pipeline_control_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved: pipeline_control_metrics.json")

    # Save predictions
    out_df = test_data[['Target','Pred_Close','Close']].copy()
    out_df.columns = ['Actual_Close','Predicted_Close','Close_at_T']
    out_df.to_csv("pipeline_control_output.csv", index=False)
    print("Predictions saved: pipeline_control_output.csv")

    # Next day prediction
    last_row = all_df.iloc[[-1]]
    next_trading_day = get_next_trading_day(last_row.index[-1])
    last_features = last_row[feature_cols]
    next_log_return = predictor.predict(last_features)[0]
    next_pred_price = last_row['Close'].values[0] * np.exp(next_log_return)
    print(f"\nPredicted Close for {next_trading_day}: {next_pred_price:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control Pipeline: AutoGluon Baseline")
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data CSV.')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test data CSV.')
    parser.add_argument('--stock_symbol', type=str, default="STOCK", help='Stock symbol for context.')
    parser.add_argument('--market_index_file', type=str, required=True, help='Path to local Market Index data CSV (e.g., spy.csv). Needs Date,Close columns.')
    parser.add_argument('--vix_file', type=str, required=True, help='Path to local VIX data CSV. Needs Date,Close (or Value) columns.')
    parser.add_argument('--epu_file', type=str, default=None, help='Optional path to local EPU data CSV. Needs Date and a value column.')
    parser.add_argument('--commodity_file', type=str, default=None, help='Optional path to local Commodity data CSV. Needs Date,Close columns.')
    args = parser.parse_args()
    main(args)