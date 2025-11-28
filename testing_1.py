# ----- STEP 1: IMPORT LIBRARIES -----

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# ----- STEP 2: DOWNLOAD STOCK DATA -----

TICKER = "RELIANCE.NS"

data = yf.download(TICKER, period="5y", interval="1d")

# Flatten the MultiIndex columns
data.columns = data.columns.get_level_values(0)

# ----- STEP 3: CLEAN BASIC DATA -----

df = data[["Close", "Volume"]].copy()
df.dropna(inplace=True)

# ----- STEP 4: ADD MOVING AVERAGES -----

df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()

# ----- STEP 5: ADD RSI (14) -----

delta = df["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

window = 14
avg_gain = gain.rolling(window=window, min_periods=window).mean()
avg_loss = loss.rolling(window=window, min_periods=window).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
df["RSI_14"] = rsi

df = df.reset_index(drop=True)

# ----- STEP 8: ADD MORE FEATURES -----

df["Return_1d"] = df["Close"].pct_change()
df["Return_5d"] = df["Close"].pct_change(5)
df["Volatility_10d"] = df["Return_1d"].rolling(window=10).std()
df["Price_vs_SMA10"] = df["Close"] / df["SMA_10"]

# ----- STEP 12: TREND STRENGTH -----

df["SMA_Diff"] = df["SMA_10"] - df["SMA_20"]
df["SMA_Ratio"] = df["SMA_10"] / df["SMA_20"]

# ----- STEP 13: ROLLING RETURNS -----

df["Return_3d_mean"] = df["Return_1d"].rolling(window=3).mean()
df["Return_7d_mean"] = df["Return_1d"].rolling(window=7).mean()

# ----- STEP 14: VOLATILITY CLUSTERING -----

df["Volatility_3d"] = df["Return_1d"].rolling(window=3).std()
df["Volatility_7d"] = df["Return_1d"].rolling(window=7).std()

# Add High-Low-Open
df["High"] = data["High"]
df["Low"] = data["Low"]
df["Open"] = data["Open"]

# ----- STEP 15: RANGE FEATURES -----

df["Range_HL"] = df["High"] - df["Low"]
df["Range_OC"] = df["Close"] - df["Open"]

# ----- STEP 7: BUILD ML DATASET -----

df = df.reset_index(drop=True)
df["Tomorrow_Close"] = df["Close"].shift(-1)

mask_valid = df["Tomorrow_Close"].notna()
df_ml = df.loc[mask_valid].copy()

today_close = df_ml["Close"].to_numpy().reshape(-1)
tomorrow_close = df_ml["Tomorrow_Close"].to_numpy().reshape(-1)

target_up_array = (tomorrow_close > today_close).astype(int)
df_ml["Target_Up"] = target_up_array

# ----- SELECT FEATURES AND TARGET -----

feature_cols = [
    "Close", "Volume", "SMA_10", "SMA_20", "RSI_14", "Return_1d",
    "Return_5d", "Volatility_10d", "Price_vs_SMA10",
    "SMA_Diff", "SMA_Ratio", "Return_3d_mean", "Return_7d_mean",
    "Volatility_3d", "Volatility_7d", "Range_HL", "Range_OC",
]

X = df_ml[feature_cols]
y = df_ml["Target_Up"]

# Train-test split

for i in range(1, 10):       # 1 to 9
    split = i / 10           # 0.1 to 0.9

    n_rows = len(df_ml)
    train_size = int(n_rows * split)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    # Train model

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Only this print remains
    msg = f"\033[1;96mTest Accuracy: {acc:.2f}\033[0m"
    print("+" + "-" * (len(msg) + 2) + "+")
    print("| " + msg + " |")
    print("+" + "-" * (len(msg) + 2) + "+")

# Plotting (unchanged)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(df.index, df["Close"], label="Close")
plt.plot(df.index, df["SMA_10"], label="SMA 10")
plt.plot(df.index, df["SMA_20"], label="SMA 20")
plt.title(f"{TICKER} - Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df.index, df["RSI_14"], label="RSI 14", color="orange")
plt.axhline(70, linestyle="--")
plt.axhline(30, linestyle="--")
plt.title("RSI (14)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
