import pandas as pd
import numpy as np
import lightgbm as lgb
import ta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import RFE
from tqdm import tqdm
import warnings
import os
import mlflow
import yaml
import json
import optuna

# Suppress Optuna's informational messages during the study
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

def get_price_column_name(df):
    """Finds the correct price column ('Close' or 'Price') in a DataFrame."""
    possible_cols = ['Close', 'Price']
    for col in possible_cols:
        if col in df.columns:
            return col
    raise KeyError(f"None of the expected price columns {possible_cols} found in DataFrame.")

def load_all_data():
    """Loads all required and optional data for the V5 pipeline."""
    print("Loading data from local CSV files...")
    data_path = 'data/'
    
    required_files = {
        'stock_train': 'stock_train.csv', 'stock_test': 'stock_test.csv',
        'local_index': 'local_index_data.csv', 'vol_index': 'vol_index_data.csv',
        'crude_oil': 'crude_oil_data.csv', 'gold': 'gold_data.csv',
        'us_10y_treasury': 'us_10y_treasury_data.csv'
    }
    
    optional_files = {
        'currency_exchange_rate': 'currency_exchange_rate_data.csv'
    }

    dfs = {}
    for name, filename in required_files.items():
        path = os.path.join(data_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: Required data file '{path}' not found. Please create it as per MANUAL_DATA.md")
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        df.index = df.index.normalize()
        dfs[name] = df
    
    for name, filename in optional_files.items():
        path = os.path.join(data_path, filename)
        if os.path.exists(path):
            print(f"Found optional data file '{filename}'.")
            df = pd.read_csv(path, index_col='Date', parse_dates=True)
            df.index = df.index.normalize()
            dfs[name] = df
        else:
            print(f"Warning: Optional data file '{filename}' not found. Skipping related features.")

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

def create_features(df):
    """Engineers the feature set for the V5 Price Expert model."""
    print("Engineering features...")
    
    core_cols = ['High', 'Low', 'Close', 'Volume']
    for col in core_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=core_cols, inplace=True)

    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['bollinger_width'] = ta.volatility.BollingerBands(df['Close']).bollinger_wband()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_21d'] = df['Close'].pct_change(21)
    df['rsi_rol_mean_10d'] = df['rsi'].rolling(10).mean()
    df['bollinger_width_rol_mean_10d'] = df['bollinger_width'].rolling(10).mean()
    df['return_rol_std_10d'] = df['return_5d'].rolling(10).std()
    df['vol_of_vol'] = df['vol_index'].rolling(10).std()
    df['rsi_x_bb_width'] = df['rsi'] * df['bollinger_width']
    df['atr_x_vol_of_vol'] = df['atr'] * df['vol_of_vol']

    if 'currency_exchange_rate' in df.columns:
        print("Creating currency-based features...")
        df['currency_return'] = df['currency_exchange_rate'].pct_change()
        df['stock_currency_corr_30d'] = df['return_5d'].rolling(30).corr(df['currency_return'])
    
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df['close_vs_index'] = df['Close'] / df['local_index']
    
    for lag in [1, 2, 3]:
        df[f'return_lag_{lag}'] = df['Close'].pct_change(lag)
        df[f'vol_change_lag_{lag}'] = df['vol_index'].pct_change(lag)

    df['target'] = df['Close'].shift(-1) - df['Close']
    df['target_volatility_10d'] = df['target'].shift(1).rolling(10).std()

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def select_features(df, n_features, test_split_index):
    """Selects the best features using Recursive Feature Elimination (RFE)."""
    print(f"Performing feature selection to find the best {n_features} features...")
    train_df = df.iloc[:test_split_index]
    all_features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target']]
    X_train = train_df[all_features]
    y_train = train_df['target']

    estimator = lgb.LGBMRegressor(n_jobs=-1, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.support_].tolist()
    print(f"Selected Features: {selected_features}")
    return selected_features

def optimize_hyperparameters(df, features, test_split_index, n_trials):
    """Finds the best hyperparameters using Optuna."""
    if n_trials <= 0:
        print("Skipping hyperparameter optimization.")
        return {}
        
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    train_val_df = df.iloc[:test_split_index]
    train_opt_df = train_val_df.iloc[:-252]
    val_opt_df = train_val_df.iloc[-252:]
    
    X_train_opt, y_train_opt = train_opt_df[features], train_opt_df['target']
    X_val_opt, y_val_opt = val_opt_df[features], val_opt_df['target']
    
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'verbose': -1, 'n_jobs': -1, 'seed': 42
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_opt, y_train_opt, eval_set=[(X_val_opt, y_val_opt)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val_opt)
        mae = mean_absolute_error(y_val_opt, preds)
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Optimization finished. Best MAE: {study.best_value:.4f}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    return study.best_params

def walk_forward_validation(df, test_set_size, features, lgbm_params, early_stopping_rounds, train_window_size, retrain_every_n_days):
    """Performs walk-forward validation with a rolling window and periodic retraining."""
    print(f"Starting walk-forward validation. Retraining every {retrain_every_n_days} days.")
    predictions = []
    model = None
    for i in tqdm(range(test_set_size), desc="Walk-Forward Validation"):
        train_end_index = len(df) - test_set_size + i
        if model is None or i % retrain_every_n_days == 0:
            train_start_index = max(0, train_end_index - train_window_size)
            current_train_data = df.iloc[train_start_index:train_end_index]
            X_train, y_train = current_train_data[features], current_train_data['target']
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)],
                      callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        current_test_row = df.iloc[[train_end_index]]
        X_test = current_test_row[features]
        pred = model.predict(X_test)[0]
        predictions.append(pred)
    return df.iloc[-test_set_size:], np.array(predictions)

def main():
    mlflow.set_experiment("Advanced Stock Forecasting")
    with mlflow.start_run():
        print("MLflow Run Started...")
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        lgbm_params = params.get('lgbm', {})
        training_params = params.get('training', {})
        feature_params = params.get('feature_selection', {})
        opt_params = params.get('optimization', {})

        early_stopping_rounds = params.get('early_stopping', {}).get('rounds', 100)
        train_window_size = training_params.get('train_window_size_days', 1260)
        retrain_every_n_days = training_params.get('retrain_every_n_days', 5)
        n_features_to_select = feature_params.get('n_features_to_select', 15)
        
        run_optimization = opt_params.get('enabled', False)
        optuna_trials = opt_params.get('n_trials', 0)
        
        mlflow.log_params(lgbm_params)
        mlflow.log_params(training_params)
        mlflow.log_params(feature_params)
        mlflow.log_params(opt_params)
        print("Parameters logged to MLflow.")

        combined_df, test_start_date = load_all_data()
        featured_df = create_features(combined_df.copy())
        
        test_split_index = featured_df.index.searchsorted(test_start_date)
        actual_test_size = len(featured_df) - test_split_index
        
        if actual_test_size <= 0:
            raise ValueError("Test set is empty after feature engineering.")

        selected_features = select_features(featured_df, n_features_to_select, test_split_index)
        mlflow.log_param("selected_features", selected_features)

        best_params = {}
        if run_optimization:
            best_params = optimize_hyperparameters(featured_df, selected_features, test_split_index, optuna_trials)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        final_lgbm_params = lgbm_params.copy()
        final_lgbm_params.update(best_params)

        results_df, predicted_changes = walk_forward_validation(
            featured_df, actual_test_size, selected_features, final_lgbm_params, 
            early_stopping_rounds, train_window_size, retrain_every_n_days
        )
        
        results_df['predicted_change'] = predicted_changes
        results_df['prediction'] = results_df['Close'] + results_df['predicted_change']
        
        results_df['actual_target_price'] = results_df['Close'].shift(-1)
        results_df.dropna(subset=['actual_target_price'], inplace=True)

        rmse = np.sqrt(mean_squared_error(results_df['actual_target_price'], results_df['prediction']))
        mae = mean_absolute_error(results_df['actual_target_price'], results_df['prediction'])
        
        results_df['actual_direction'] = np.sign(results_df['actual_target_price'] - results_df['Close'])
        results_df['predicted_direction'] = np.sign(results_df['prediction'] - results_df['Close'])
        results_df_filtered = results_df[results_df['actual_direction'] != 0]
        dir_accuracy = np.mean(results_df_filtered['predicted_direction'] == results_df_filtered['actual_direction']) * 100

        print("\n--- Walk-Forward Validation Results ---")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Directional Accuracy: {dir_accuracy:.2f}%")
        
        metrics = {"rmse": rmse, "mae": mae, "directional_accuracy": dir_accuracy}
        mlflow.log_metrics(metrics)
        print("Metrics logged to MLflow.")
        
        os.makedirs("plots", exist_ok=True)
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("DVC metrics file saved to metrics.json")

        plot_path = "plots/predictions_vs_actuals.png"
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(15, 7))
        plt.plot(results_df.index, results_df['actual_target_price'], label='Actual Price', color='royalblue')
        plt.plot(results_df.index, results_df['prediction'], label='Predicted Price', color='darkorange', linestyle='--')
        plt.title('Walk-Forward Validation: Actual vs. Predicted Prices'); plt.xlabel('Date'); plt.ylabel('Price'); plt.legend()
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

        mlflow.log_artifact(plot_path)
        
        results_df_to_log = results_df[['actual_target_price', 'prediction', 'predicted_change', 'actual_direction', 'predicted_direction']]
        results_df_to_log.to_csv("prediction_results.csv")
        mlflow.log_artifact("prediction_results.csv")
        print("Artifacts (plot, results CSV) logged to MLflow.")

if __name__ == '__main__':
    main()