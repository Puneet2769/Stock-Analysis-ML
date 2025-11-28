# region Imports & Config

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

pd.set_option("display.float_format", lambda x: f"{x:.2f}")

TICKER = "RELIANCE.NS"
PERIOD = "5y"
INTERVAL = "1d"
TRAIN_FRACTION = 0.8
INITIAL_CAPITAL = 100_000

# endregion


# region Feature Engineering

def download_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval)
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def add_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw OHLCV data and add all technical features.
    Returns df with:
      Close, Volume, High, Low, Open,
      SMA_10, SMA_20, RSI_14,
      Return_1d, Return_5d, Volatility_10d, Price_vs_SMA10,
      SMA_Diff, SMA_Ratio, Return_3d_mean, Return_7d_mean,
      Volatility_3d, Volatility_7d, Range_HL, Range_OC,
      MACD_Line, MACD_Signal, MACD_Hist,
      BB_Width, BB_Position,
      ADX_14
    """
    df = raw[["Close", "Volume", "High", "Low", "Open"]].copy()
    df = df.dropna().reset_index(drop=True)

    # ----- Moving averages -----
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # ----- RSI 14 -----
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    window = 14
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["RSI_14"] = rsi

    # ----- Returns & volatility -----
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Volatility_10d"] = df["Return_1d"].rolling(window=10).std()

    # ----- Relative to MA -----
    df["Price_vs_SMA10"] = df["Close"] / df["SMA_10"]

    # ----- Trend strength -----
    df["SMA_Diff"] = df["SMA_10"] - df["SMA_20"]
    df["SMA_Ratio"] = df["SMA_10"] / df["SMA_20"]

    # ----- Rolling return means -----
    df["Return_3d_mean"] = df["Return_1d"].rolling(window=3).mean()
    df["Return_7d_mean"] = df["Return_1d"].rolling(window=7).mean()

    # ----- Volatility clustering -----
    df["Volatility_3d"] = df["Return_1d"].rolling(window=3).std()
    df["Volatility_7d"] = df["Return_1d"].rolling(window=7).std()

    # ----- Ranges -----
    df["Range_HL"] = df["High"] - df["Low"]
    df["Range_OC"] = df["Close"] - df["Open"]

    # ================== NEW: MACD (12, 26, 9) ==================
    # EMA helpers
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    df["MACD_Line"] = macd_line
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    # ================== NEW: Bollinger Bands (20, 2σ) ==================
    bb_window = 20
    bb_mid = df["Close"].rolling(window=bb_window).mean()
    bb_std = df["Close"].rolling(window=bb_window).std()

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Band width (normalized by mid)
    df["BB_Width"] = (bb_upper - bb_lower) / bb_mid

    # Position of price within the bands: 0 = lower band, 1 = upper band
    band_range = (bb_upper - bb_lower)
    df["BB_Position"] = (df["Close"] - bb_lower) / band_range

    # Avoid division issues where band_range is 0
    df["BB_Position"] = df["BB_Position"].clip(lower=0, upper=1)

    # ================== NEW: ADX (14) ==================
    # True Range
    prev_close = df["Close"].shift(1)
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - prev_close).abs()
    low_prev_close = (df["Low"] - prev_close).abs()

    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    # Directional Movement
    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)

    plus_dm = df["High"] - prev_high
    minus_dm = prev_low - df["Low"]

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Wilder's smoothing (using EMA approximation)
    alpha = 1 / 14
    tr_smoothed = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smoothed = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
    minus_di = 100 * (minus_dm_smoothed / tr_smoothed)

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di) ) * 100
    adx_14 = dx.rolling(window=14).mean()

    df["ADX_14"] = adx_14

    # Final cleanup
    df = df.dropna().reset_index(drop=True)
    return df



def get_feature_cols() -> list[str]:
    return [
        "Close",
        "Volume",
        "SMA_10",
        "SMA_20",
        "RSI_14",
        "Return_1d",
        "Return_5d",
        "Volatility_10d",
        "Price_vs_SMA10",
        "SMA_Diff",
        "SMA_Ratio",
        "Return_3d_mean",
        "Return_7d_mean",
        "Volatility_3d",
        "Volatility_7d",
        "Range_HL",
        "Range_OC",

        # New MACD features
        "MACD_Line",
        "MACD_Signal",
        "MACD_Hist",

        # New Bollinger features
        "BB_Width",
        "BB_Position",

        # New ADX feature
        "ADX_14",
    ]


# endregion


# region Dataset Builders (1-day, 5-day, regression)

def build_1d_datasets(df: pd.DataFrame):
    """
    Build 1-day classification and regression datasets from indicator df.
    Returns:
      df_ml, X_cls, y_cls, X_reg, y_reg, feature_cols, train_size
    """
    df = df.copy().reset_index(drop=True)
    df["Tomorrow_Close"] = df["Close"].shift(-1)

    mask = df["Tomorrow_Close"].notna()
    df_ml = df.loc[mask].copy()

    today_close = df_ml["Close"].to_numpy().reshape(-1)
    tomorrow_close = df_ml["Tomorrow_Close"].to_numpy().reshape(-1)
    df_ml["Target_Up"] = (tomorrow_close > today_close).astype(int)

    df_ml["Return_1d_future"] = (df_ml["Tomorrow_Close"] - df_ml["Close"]) / df_ml["Close"]

    feature_cols = get_feature_cols()
    X_cls = df_ml[feature_cols]
    y_cls = df_ml["Target_Up"]

    X_reg = df_ml[feature_cols]
    y_reg = df_ml["Return_1d_future"]

    n_rows = len(df_ml)
    train_size = int(n_rows * TRAIN_FRACTION)

    return df_ml, X_cls, y_cls, X_reg, y_reg, feature_cols, train_size


def build_5d_dataset(df: pd.DataFrame, feature_cols: list[str]):
    """
    Build 5-day classification dataset from indicator df.
    Returns:
      df_5d, X_5d, y_5d, train_size_5d
    """
    df_5d = df.copy().reset_index(drop=True)
    df_5d["Close_5d"] = df_5d["Close"].shift(-5)

    mask_5d = df_5d["Close_5d"].notna()
    df_5d = df_5d.loc[mask_5d].copy()

    df_5d["Target_Up_5d"] = (df_5d["Close_5d"] > df_5d["Close"]).astype(int)
    df_5d["Return_5d_future"] = (df_5d["Close_5d"] - df_5d["Close"]) / df_5d["Close"]

    X_5d = df_5d[feature_cols]
    y_5d = df_5d["Target_Up_5d"]

    n_rows_5d = len(df_5d)
    train_size_5d = int(n_rows_5d * TRAIN_FRACTION)

    return df_5d, X_5d, y_5d, train_size_5d

# endregion


# region Model Training & Evaluation

def train_1d_classifier(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_1d_classifier(model, X_test, y_test, feature_cols):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    msg = f"\033[1;96mTest Accuracy: {acc:.2f}\033[0m"
    print("+" + "-" * (len(msg) + 2) + "+")
    print("| " + msg + " |")
    print("+" + "-" * (len(msg) + 2) + "+")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=2))

    importances = model.feature_importances_
    importance_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importance (higher = more important):")
    print(importance_series.round(3))

    return y_pred


def train_regression_model(X_reg_train, y_reg_train) -> RandomForestRegressor:
    reg_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )
    reg_model.fit(X_reg_train, y_reg_train)
    return reg_model


def evaluate_regression_model(reg_model, X_reg_test, y_reg_test):
    y_reg_pred = reg_model.predict(X_reg_test)

    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)

    print("\n+---------------- REGRESSION METRICS ----------------+")
    print(f"MAE (mean abs error): {mae:.4f}")
    print(f"RMSE (root mean sq error): {rmse:.4f}")
    print(f"R^2 (explained variance): {r2:.4f}")
    print("+---------------------------------------------------+")

    true_direction = (y_reg_test > 0).astype(int)
    pred_direction = (y_reg_pred > 0).astype(int)
    dir_acc = accuracy_score(true_direction, pred_direction)
    print(f"\nDirection accuracy from regression model: {dir_acc:.2f}")

    return y_reg_pred


def train_5d_classifier(X_5d_train, y_5d_train) -> RandomForestClassifier:
    model_5d = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )
    model_5d.fit(X_5d_train, y_5d_train)
    return model_5d


def evaluate_5d_classifier(model_5d, X_5d_test, y_5d_test):
    y_5d_pred = model_5d.predict(X_5d_test)
    acc_5d = accuracy_score(y_5d_test, y_5d_pred)

    print("\n+---------------- 5-DAY MODEL ACCURACY ----------------+")
    print(f"5-Day Test Accuracy: {acc_5d:.2f}")
    print("+-----------------------------------------------------+")

    print("\n5-Day Classification report:")
    print(classification_report(y_5d_test, y_5d_pred, digits=2))

    return y_5d_pred

# endregion


# region Backtests (1-day, 5-day)

def backtest_1d(df_ml: pd.DataFrame, train_size: int, y_pred):
    df_test_bt = df_ml.iloc[train_size:].copy()

    df_test_bt["Pred_Class"] = y_pred
    df_test_bt["True_Class"] = df_test_bt["Target_Up"].to_numpy()
    df_test_bt["Future_Return"] = df_test_bt["Return_1d_future"]

    print("\nBacktest sample rows:")
    print(df_test_bt[["Close", "Tomorrow_Close", "Future_Return", "Pred_Class", "True_Class"]].head(10))

    initial_capital = INITIAL_CAPITAL

    # Buy & hold
    df_test_bt["BH_Equity"] = (1 + df_test_bt["Future_Return"]).cumprod() * initial_capital

    # Strategy: only apply return when Pred_Class == 1
    df_test_bt["Strategy_Return"] = df_test_bt["Future_Return"] * df_test_bt["Pred_Class"]
    df_test_bt["Strategy_Equity"] = (1 + df_test_bt["Strategy_Return"]).cumprod() * initial_capital

    final_bh = df_test_bt["BH_Equity"].iloc[-1]
    final_strategy = df_test_bt["Strategy_Equity"].iloc[-1]

    total_return_bh = (final_bh / initial_capital) - 1
    total_return_strategy = (final_strategy / initial_capital) - 1

    days_in_market = df_test_bt["Pred_Class"].sum()
    total_days = len(df_test_bt)

    print("\n+----------------- BACKTEST SUMMARY -----------------+")
    print(f"Buy & Hold final equity:   ₹{final_bh:,.2f}  (Return: {total_return_bh*100:.2f}%)")
    print(f"Strategy final equity:      ₹{final_strategy:,.2f}  (Return: {total_return_strategy*100:.2f}%)")
    print(f"Days in test period:        {total_days}")
    print(f"Days in market (strategy):  {days_in_market}")
    print("+---------------------------------------------------+")


def backtest_5d(df_5d: pd.DataFrame, train_size_5d: int, y_5d_pred, y_5d_test):
    df_5d_test = df_5d.iloc[train_size_5d:].copy()

    df_5d_test["Pred_Class_5d"] = y_5d_pred
    df_5d_test["True_Class_5d"] = y_5d_test.to_numpy()
    df_5d_test["Future_Return_5d"] = df_5d_test["Return_5d_future"]

    print("\n5-Day Backtest sample rows:")
    print(df_5d_test[["Close", "Close_5d", "Future_Return_5d", "Pred_Class_5d", "True_Class_5d"]].head(10))

    # Strategy rules
    signal_model_up = df_5d_test["Pred_Class_5d"] == 1
    signal_trend_up = df_5d_test["SMA_10"] > df_5d_test["SMA_20"]
    signal_rsi_ok = df_5d_test["RSI_14"].between(40, 70)

    entry_signal = signal_model_up & signal_trend_up & signal_rsi_ok
    df_5d_test["Entry_Signal"] = entry_signal.astype(int)

    print("\n5-Day strategy signal counts:")
    print("Total test rows:", len(df_5d_test))
    print("Rows where model says UP:", signal_model_up.sum())
    print("Rows passing all filters (Entry_Signal=1):", df_5d_test["Entry_Signal"].sum())

    initial_capital_5d = INITIAL_CAPITAL

    # Buy & hold (5-day overlaps, simplified)
    df_5d_test["BH_Equity_5d"] = (1 + df_5d_test["Future_Return_5d"]).cumprod() * initial_capital_5d

    # Strategy returns
    df_5d_test["Strategy_Return_5d"] = df_5d_test["Future_Return_5d"] * df_5d_test["Entry_Signal"]
    df_5d_test["Strategy_Equity_5d"] = (1 + df_5d_test["Strategy_Return_5d"]).cumprod() * initial_capital_5d

    final_bh_5d = df_5d_test["BH_Equity_5d"].iloc[-1]
    final_strategy_5d = df_5d_test["Strategy_Equity_5d"].iloc[-1]

    total_return_bh_5d = (final_bh_5d / initial_capital_5d) - 1
    total_return_strategy_5d = (final_strategy_5d / initial_capital_5d) - 1

    num_trades_5d = df_5d_test["Entry_Signal"].sum()
    total_rows_5d = len(df_5d_test)

    print("\n+----------- 5-DAY STRATEGY BACKTEST SUMMARY -----------+")
    print(f"5-Day Buy & Hold final equity:   ₹{final_bh_5d:,.2f}  (Return: {total_return_bh_5d*100:.2f}%)")
    print(f"5-Day Strategy final equity:      ₹{final_strategy_5d:,.2f}  (Return: {total_return_strategy_5d*100:.2f}%)")
    print(f"5-Day test rows:                  {total_rows_5d}")
    print(f"5-Day strategy entries (trades):  {num_trades_5d}")
    print("+-------------------------------------------------------+")

# endregion


# region Plotting

def plot_price_and_rsi(df: pd.DataFrame, ticker: str):
    plt.figure(figsize=(12, 8))

    # Price + MAs
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["Close"], label="Close")
    plt.plot(df.index, df["SMA_10"], label="SMA 10")
    plt.plot(df.index, df["SMA_20"], label="SMA 20")
    plt.title(f"{ticker} - Price with Moving Averages")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()

    # RSI
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["RSI_14"], label="RSI 14")
    plt.axhline(70, linestyle="--")
    plt.axhline(30, linestyle="--")
    plt.title("RSI (14)")
    plt.xlabel("Index")
    plt.ylabel("RSI")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# endregion


# region Main

def main():
    # 1) Download & feature engineering
    raw = download_data(TICKER, PERIOD, INTERVAL)
    df_ind = add_indicators(raw)

    # 2) 1-day datasets
    df_ml, X_cls, y_cls, X_reg, y_reg, feature_cols, train_size = build_1d_datasets(df_ind)

    X_train = X_cls.iloc[:train_size]
    y_train = y_cls.iloc[:train_size]
    X_test = X_cls.iloc[train_size:]
    y_test = y_cls.iloc[train_size:]

    X_reg_train = X_reg.iloc[:train_size]
    y_reg_train = y_reg.iloc[:train_size]
    X_reg_test = X_reg.iloc[train_size:]
    y_reg_test = y_reg.iloc[train_size:]

    # 3) 5-day dataset
    df_5d, X_5d, y_5d, train_size_5d = build_5d_dataset(df_ind, feature_cols)
    X_5d_train = X_5d.iloc[:train_size_5d]
    y_5d_train = y_5d.iloc[:train_size_5d]
    X_5d_test = X_5d.iloc[train_size_5d:]
    y_5d_test = y_5d.iloc[train_size_5d:]

    # 4) Train & evaluate 5-day classifier + backtest
    model_5d = train_5d_classifier(X_5d_train, y_5d_train)
    y_5d_pred = evaluate_5d_classifier(model_5d, X_5d_test, y_5d_test)
    backtest_5d(df_5d, train_size_5d, y_5d_pred, y_5d_test)

    # 5) Train & evaluate regression model
    reg_model = train_regression_model(X_reg_train, y_reg_train)
    evaluate_regression_model(reg_model, X_reg_test, y_reg_test)

    # 6) Train & evaluate 1-day classifier + backtest
    model_1d = train_1d_classifier(X_train, y_train)
    y_pred = evaluate_1d_classifier(model_1d, X_test, y_test, feature_cols)
    backtest_1d(df_ml, train_size, y_pred)

    # 7) Plot
    plot_price_and_rsi(df_ind, TICKER)


if __name__ == "__main__":
    main()

# endregion
