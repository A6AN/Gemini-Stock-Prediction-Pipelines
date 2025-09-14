import warnings
import os

# --- Stability Fix for TensorFlow on some systems (e.g., macOS) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
# --- End of Stability Fix ---

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import mlflow
import yaml
import json
import ta

warnings.filterwarnings('ignore')

def get_price_column_name(df):
    """Finds the correct price column ('Close' or 'Price') in a DataFrame."""
    possible_cols = ['Close', 'Price']
    for col in possible_cols:
        if col in df.columns:
            return col
    raise KeyError(f"None of the expected price columns {possible_cols} found in DataFrame.")

def load_data():
    """Loads all required data from the local 'data/' directory."""
    print("Loading data from local CSV files...")
    data_path = 'data/'
    
    required_files = {
        'stock_train': 'stock_train.csv', 'stock_test': 'stock_test.csv',
        'local_index': 'local_index_data.csv', 'vol_index': 'vol_index_data.csv',
    }

    dfs = {}
    for name, filename in required_files.items():
        path = os.path.join(data_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Required data file '{path}' not found.")
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        df.index = df.index.normalize()
        dfs[name] = df
    
    stock_full = pd.concat([dfs['stock_train'], dfs['stock_test']])
    
    external_data_series = {}
    for name, df in dfs.items():
        if name not in ['stock_train', 'stock_test']:
            price_col = get_price_column_name(df)
            external_data_series[name] = df[price_col].rename(name)
            
    external_data = pd.concat(external_data_series.values(), axis=1)
    for col in external_data.columns:
        external_data[col] = pd.to_numeric(external_data[col], errors='coerce')
    
    external_data = external_data.bfill().ffill()
    combined_df = stock_full.join(external_data, how='left').bfill().ffill()
    test_start_date = dfs['stock_test'].index.min()
    return combined_df, test_start_date

def prepare_features(df):
    """Creates a richer feature set for the GRU model."""
    print("Preparing rich feature set for GRU...")
    features_df = pd.DataFrame(index=df.index)
    
    # Core features
    features_df['return'] = df['Close'].pct_change()
    features_df['index_return'] = df['local_index'].pct_change()
    features_df['vol_index_change'] = df['vol_index'].pct_change()
    
    # Technical indicators from Pipeline 1 to add more context
    features_df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    features_df['bollinger_width'] = ta.volatility.BollingerBands(df['Close']).bollinger_wband()
    
    features_df.replace([np.inf, -np.inf], 0, inplace=True)
    features_df.dropna(inplace=True)
    return features_df

def create_sequences(data, target, sequence_length):
    """Converts a time series DataFrame into sequences for GRU training."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def build_gru_model(seq_len, num_features, params):
    """Builds and compiles the Keras GRU model."""
    model = Sequential([
        GRU(params['gru_units_layer_1'], return_sequences=True, input_shape=(seq_len, num_features)),
        Dropout(params['dropout_rate']),
        GRU(params['gru_units_layer_2'], return_sequences=False),
        Dropout(params['dropout_rate']),
        Dense(params['dense_units'], activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    mlflow.set_experiment("GRU Stock Forecasting - Pipeline 2")
    with mlflow.start_run():
        print("MLflow Run Started for Pipeline 2 (GRU)...")
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        mlflow.log_params(params['gru_model'])
        mlflow.log_params(params['training'])
        mlflow.log_params(params['early_stopping'])
        print("Parameters logged to MLflow.")

        combined_df, test_start_date = load_data()
        features_df = prepare_features(combined_df)

        test_split_index = features_df.index.searchsorted(test_start_date)
        actual_test_size = len(features_df) - test_split_index
        
        predictions = []
        model = None
        scaler = None
        seq_len = params['gru_model']['sequence_length']
        retrain_interval = params['training']['retrain_every_n_days']
        
        for i in tqdm(range(actual_test_size), desc="Optimized Walk-Forward Validation (GRU)"):
            current_day_index = test_split_index + i
            
            if i % retrain_interval == 0:
                print(f"\nRetraining model at step {i}...")
                train_data = features_df.iloc[:current_day_index]
                
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train_data = scaler.fit_transform(train_data)
                
                X_train_seq, y_train_seq = create_sequences(scaled_train_data, scaled_train_data[:, 0], seq_len)
                
                if X_train_seq.shape[0] == 0:
                    raise ValueError("Training sequence is empty. Check data length and sequence length.")

                model = build_gru_model(seq_len, X_train_seq.shape[2], params['gru_model'])
                
                es_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', patience=params['early_stopping']['patience'], restore_best_weights=True
                )
                
                model.fit(X_train_seq, y_train_seq, 
                          epochs=params['training']['epochs'], 
                          batch_size=params['training']['batch_size'], 
                          callbacks=[es_callback], verbose=0)
            
            input_data = features_df.iloc[current_day_index - seq_len : current_day_index]
            scaled_input = scaler.transform(input_data)
            X_pred = np.reshape(scaled_input, (1, seq_len, features_df.shape[1]))
            
            predicted_scaled_return = model.predict(X_pred, verbose=0)[0][0]
            
            dummy_row = np.zeros((1, features_df.shape[1]))
            dummy_row[0, 0] = predicted_scaled_return
            predicted_return = scaler.inverse_transform(dummy_row)[0, 0]
            
            predictions.append(predicted_return)

        results_df = combined_df[combined_df.index.isin(features_df.index[-actual_test_size:])].copy()
        results_df['predicted_return'] = predictions
        results_df['prediction'] = results_df['Close'] * (1 + results_df['predicted_return'])
        results_df['actual_target_price'] = results_df['Close'].shift(-1)
        results_df.dropna(subset=['actual_target_price'], inplace=True)

        rmse = np.sqrt(mean_squared_error(results_df['actual_target_price'], results_df['prediction']))
        mae = mean_absolute_error(results_df['actual_target_price'], results_df['prediction'])
        results_df['actual_direction'] = np.sign(results_df['actual_target_price'] - results_df['Close'])
        results_df['predicted_direction'] = np.sign(results_df['prediction'] - results_df['Close'])
        results_df_filtered = results_df[results_df['actual_direction'] != 0]
        dir_accuracy = np.mean(results_df_filtered['predicted_direction'] == results_df_filtered['actual_direction']) * 100

        print("\n--- Walk-Forward Validation Results (GRU) ---")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Directional Accuracy: {dir_accuracy:.2f}%")
        
        metrics = {"rmse": rmse, "mae": mae, "directional_accuracy": dir_accuracy}
        mlflow.log_metrics(metrics)
        
        with open("metrics.json", "w") as f: json.dump(metrics, f, indent=4)
        print("DVC metrics file saved to metrics.json")

        plot_path = "plots/predictions_vs_actuals.png"
        os.makedirs("plots", exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(15, 7))
        plt.plot(results_df.index, results_df['actual_target_price'], label='Actual Price')
        plt.plot(results_df.index, results_df['prediction'], label='Predicted Price', linestyle='--')
        plt.title('GRU Walk-Forward Validation'); plt.xlabel('Date'); plt.ylabel('Price'); plt.legend()
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)
        results_df[['actual_target_price', 'prediction']].to_csv("prediction_results.csv")
        mlflow.log_artifact("prediction_results.csv")
        print("Artifacts logged to MLflow.")

if __name__ == '__main__':
    main()