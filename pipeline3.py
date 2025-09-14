import warnings
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import mlflow
import yaml
import joblib
import ta

# --- Suppress Warnings ---
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Transformer Components ---
class PositionalEncoding(Layer):
    def __init__(self, max_position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.calculate_positional_encoding(max_position, d_model)
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    def calculate_positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]); angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder_block(inputs, d_model, num_heads, ff_dim, dropout_rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# --- Data and Feature Functions ---
def load_trends_data():
    data_path = 'data/'
    trends_files = {
        'trend_apple_stock': 'apple_stock_trend.csv', 'trend_iphone': 'iphone_trend.csv',
        'trend_aapl_earnings': 'AAPL_earnings_trend.csv', 'trend_tim_cook': 'tim_cook_trends.csv',
        'trend_macbook': 'macbook_trends.csv'
    }
    all_trends_df = None
    for name, filename in trends_files.items():
        path = os.path.join(data_path, filename)
        if os.path.exists(path):
            try:
                trend_df = pd.read_csv(path, skiprows=2)
                trend_df.columns = ['Date', name]
                trend_df['Date'] = pd.to_datetime(trend_df['Date'])
                trend_df.set_index('Date', inplace=True)
                if pd.infer_freq(trend_df.index) in ['M', 'MS']:
                    trend_df = trend_df.resample('D').ffill()
                all_trends_df = trend_df if all_trends_df is None else all_trends_df.join(trend_df, how='outer')
                print(f"Successfully loaded trends file: {filename}")
            except Exception as e:
                print(f"Warning: Could not process trends file {filename}. Error: {e}")
    if all_trends_df is not None:
        all_trends_df.ffill(inplace=True); all_trends_df.bfill(inplace=True)
    return all_trends_df

def prepare_features(stock_df, spy_df, trends_df, params):
    print("Engineering enhanced feature set...")
    data = stock_df.copy()
    data['return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['volatility'] = data['return'].rolling(window=20).std()
    data['rsi'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
    # --- New Features ---
    macd = ta.trend.MACD(close=data['Close'])
    data['macd_diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(close=data['Close'])
    data['bollinger_wband'] = bollinger.bollinger_wband()

    data = data.merge(spy_df.rename(columns={'Close': 'SPY_Close'}), left_index=True, right_index=True, how='left')
    data['spy_return'] = np.log(data['SPY_Close'] / data['SPY_Close'].shift(1))
    data['relative_strength'] = data['return'] - data['spy_return']
    
    if trends_df is not None:
        data = data.join(trends_df, how='left')
        trend_window = params['feature_params']['trend_pct_change_window']
        for col in trends_df.columns:
            data[f'{col}_change'] = data[col].pct_change(trend_window)
    
    data['target'] = data['Close'].shift(-1)
    
    features_to_use = [
        'return', 'volatility', 'rsi', 'macd_diff', 'bollinger_wband',
        'relative_strength', 'spy_return'
    ] + [col for col in data.columns if col.startswith('trend_')]
    
    X_candidate, y_candidate = data[features_to_use], data['target']
    
    # --- More Aggressive Outlier Clipping ---
    lower_q = params['feature_params']['outlier_clip_lower_quantile']
    upper_q = params['feature_params']['outlier_clip_upper_quantile']
    for col in X_candidate.columns:
        if col in X_candidate.columns:
            lower_bound, upper_bound = X_candidate[col].quantile(lower_q), X_candidate[col].quantile(upper_q)
            X_candidate[col] = np.clip(X_candidate[col], lower_bound, upper_bound)

    X_candidate.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_candidate.ffill(inplace=True); X_candidate.bfill(inplace=True); X_candidate.fillna(0, inplace=True)
    common_index = y_candidate.dropna().index.intersection(X_candidate.index)
    X, y = X_candidate.loc[common_index], y_candidate.loc[common_index]
    return X, y

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X.iloc[i:(i + seq_len)].values)
        ys.append(y.iloc[i + seq_len - 1])
    return np.array(Xs), np.array(ys)

# --- Model Building ---
X_train_seq_shape_ref, HPS_ref = None, None
def build_transformer_hypermodel(hp):
    global X_train_seq_shape_ref, HPS_ref
    input_shape = (X_train_seq_shape_ref[1], X_train_seq_shape_ref[2])
    
    d_model = hp.Choice('d_model', values=HPS_ref['d_model'])
    num_heads = hp.Choice('num_heads', values=HPS_ref['num_heads'])
    num_blocks = hp.Int('num_transformer_blocks', min_value=min(HPS_ref['num_transformer_blocks']), max_value=max(HPS_ref['num_transformer_blocks']))
    ff_factor = hp.Choice('ff_dim_factor', values=HPS_ref['ff_dim_factor'])
    ff_dim = d_model * ff_factor
    dropout = hp.Float('transformer_dropout', min_value=min(HPS_ref['transformer_dropout']), max_value=max(HPS_ref['transformer_dropout']), step=0.05)
    lr = hp.Choice('learning_rate', values=HPS_ref['learning_rate'])
    
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs); x = PositionalEncoding(input_shape[0], d_model)(x)
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x); outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# --- Main Execution ---
def main():
    global HPS_ref, X_train_seq_shape_ref
    mlflow.set_experiment("Transformer Stock Forecasting - Pipeline 3")
    with mlflow.start_run() as run:
        print(f"MLflow Run Started: {run.info.run_name}")
        with open("params.yaml") as f: params = yaml.safe_load(f)
        mlflow.log_params(params)
        
        SEQ_LEN = params['feature_params']['sequence_length']
        STOCK_SYMBOL = params['feature_params']['stock_symbol']
        HPS_ref = params['model_params']['hyperparameters']
        
        train_df = pd.read_csv('data/stock_train.csv', index_col='Date', parse_dates=True)
        test_df = pd.read_csv('data/stock_test.csv', index_col='Date', parse_dates=True)
        spy_df = pd.read_csv('data/spy.csv', index_col='Date', parse_dates=True)
        trends_df = load_trends_data()
        
        combined_df = pd.concat([train_df, test_df]).sort_index()
        X_all, y_all = prepare_features(combined_df, spy_df, trends_df, params)
        
        train_mask = X_all.index < test_df.index.min(); test_mask = X_all.index >= test_df.index.min()
        X_train, y_train = X_all[train_mask], y_all[train_mask]; X_test, y_test = X_all[test_mask], y_all[test_mask]

        if X_train.empty: raise ValueError("Training set is empty after feature engineering.")

        scaler_X = MinMaxScaler((-1, 1)); scaler_y = MinMaxScaler((-1, 1))
        X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        y_train_scaled = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1,1)).flatten(), index=y_train.index)
        
        joblib.dump(scaler_X, "scaler_X.joblib"); joblib.dump(scaler_y, "scaler_y.joblib")
        mlflow.log_artifact("scaler_X.joblib"); mlflow.log_artifact("scaler_y.joblib")

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
        X_train_seq_shape_ref = X_train_seq.shape

        tuner_type = params['model_params']['tuner'].get('tuner_type', 'random').lower()
        TunerClass = kt.BayesianOptimization if tuner_type == 'bayesian' else kt.RandomSearch

        tuner = TunerClass(build_transformer_hypermodel, objective='val_loss', max_trials=params['model_params']['tuner']['max_trials'], executions_per_trial=params['model_params']['tuner']['executions_per_trial'], directory='keras_tuner', project_name=f'p3_{STOCK_SYMBOL}_advanced')
        print(f"Starting hyperparameter tuning with {tuner_type.capitalize()}Search...")
        tuner.search(X_train_seq, y_train_seq, epochs=params['model_params']['tuner']['epochs'], validation_split=0.2, callbacks=[EarlyStopping('val_loss', patience=7)])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        mlflow.log_params({f"best_hp_{k}": v for k, v in best_hps.values.items()})

        print("Training initial model on full training data...")
        model = tuner.hypermodel.build(best_hps)
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=params['training_params']['lr_scheduler_patience'], min_lr=1e-6)
        es_callback = EarlyStopping(monitor='loss', patience=params['training_params']['early_stopping_patience'], restore_best_weights=True)
        model.fit(X_train_seq, y_train_seq, epochs=params['training_params']['epochs'], batch_size=params['training_params']['batch_size'], callbacks=[es_callback, lr_scheduler], verbose=1)

        full_history_X_scaled = X_train_scaled.copy(); predictions = []
        
        for i in tqdm(range(len(X_test)), desc="Walk-Forward Validation"):
            if i > 0 and i % params['training_params']['retrain_every_n_days'] == 0:
                print(f"\nRetraining model at step {i}...")
                y_history_scaled = pd.Series(scaler_y.transform(y_all[y_all.index.isin(full_history_X_scaled.index)].values.reshape(-1,1)).flatten(), index=full_history_X_scaled.index)
                X_retrain_seq, y_retrain_seq = create_sequences(full_history_X_scaled, y_history_scaled, SEQ_LEN)
                model.fit(X_retrain_seq, y_retrain_seq, epochs=params['training_params']['epochs'], batch_size=params['training_params']['batch_size'], callbacks=[es_callback, lr_scheduler], verbose=0)
            
            input_seq = full_history_X_scaled.tail(SEQ_LEN).values.reshape(1, SEQ_LEN, X_train.shape[1])
            pred_scaled = model.predict(input_seq, verbose=0)
            pred_unscaled = scaler_y.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred_unscaled)
            
            current_test_X_row_scaled = pd.DataFrame(scaler_X.transform(X_test.iloc[[i]]), index=X_test.iloc[[i]].index, columns=X_test.columns)
            full_history_X_scaled = pd.concat([full_history_X_scaled, current_test_X_row_scaled])

        predictions = np.array(predictions); rmse = np.sqrt(mean_squared_error(y_test, predictions)); mae = mean_absolute_error(y_test, predictions)
        close_ref = combined_df['Close'].reindex(y_test.index).values
        actual_dir = np.sign(y_test - close_ref); pred_dir = np.sign(predictions - close_ref)
        dir_accuracy = np.mean(actual_dir == pred_dir) * 100
        
        print(f"\n--- Results ---\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, DirAcc: {dir_accuracy:.2f}%")
        metrics = {"rmse": rmse, "mae": mae, "directional_accuracy": dir_accuracy}; mlflow.log_metrics(metrics)
        with open("metrics.json", 'w') as f: json.dump(metrics, f)
        
        plot_path = "plots/predictions_vs_actuals.png"; os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(15, 7)); plt.plot(y_test.index, y_test, label='Actual Price'); plt.plot(y_test.index, predictions, label='Predicted Price', linestyle='--')
        plt.title('Transformer Walk-Forward Validation'); plt.xlabel('Date'); plt.ylabel('Price'); plt.legend(); plt.savefig(plot_path); plt.close()
        mlflow.log_artifact(plot_path)

if __name__ == '__main__':
    main()