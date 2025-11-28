
# ----- STEP 1: IMPORT LIBRARIES -----

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# ----- STEP 2: DOWNLOAD STOCK DATA -----

TICKER = "RELIANCE.NS"   # change this anytime

data = yf.download(TICKER, period="5y", interval="1d")

# print(data.head())
# print("Shape:", data.shape)

# ----- STEP 3: CLEAN BASIC DATA -----

df = data[["Close", "Volume"]].copy()   # keep only what we need initially
df.dropna(inplace=True)                 # remove missing rows

print(df.head())
# ----- STEP 4: ADD MOVING AVERAGES -----

df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()

# ----- STEP 5: ADD RSI (14) -----

# 1. Price difference between today and yesterday
delta = df["Close"].diff()

# 2. Separate gains (positive changes)
gain = delta.where(delta > 0, 0.0)

# 3. Separate losses (negative changes, made positive)
loss = -delta.where(delta < 0, 0.0)

# 4. Rolling averages of gains and losses (14-day)
window = 14
avg_gain = gain.rolling(window=window, min_periods=window).mean()
avg_loss = loss.rolling(window=window, min_periods=window).mean()

# 5. Calculate the RS value (Relative Strength)
rs = avg_gain / avg_loss

# 6. Final RSI formula
rsi = 100 - (100 / (1 + rs))

df["RSI_14"] = rsi

df = df.reset_index(drop=True)


# ----- STEP 7: BUILD ML DATASET -----

# Make sure index is clean
df = df.reset_index(drop=True)

# 1) Create Tomorrow_Close as a Series (NOT DataFrame)
df["Tomorrow_Close"] = df["Close"].shift(-1)

# Sanity check: both should be Series with same length
print("Close type and shape:", type(df["Close"]), df["Close"].shape)
print("Tomorrow_Close type and shape:", type(df["Tomorrow_Close"]), df["Tomorrow_Close"].shape)

# 2) Remove rows where Tomorrow_Close is NaN (typically the last row)
mask_valid = df["Tomorrow_Close"].notna()
df_ml = df.loc[mask_valid].copy()

# 3) Use SERIES -> NUMPY 1D arrays for comparison
today_close = df_ml["Close"].to_numpy().reshape(-1)
tomorrow_close = df_ml["Tomorrow_Close"].to_numpy().reshape(-1)

print("today_close shape:", today_close.shape)
print("tomorrow_close shape:", tomorrow_close.shape)

target_up_array = (tomorrow_close > today_close).astype(int)  # 1D array

print("target_up_array shape:", target_up_array.shape)

# 4) Assign Target_Up back to df_ml
df_ml["Target_Up"] = target_up_array

print(df_ml[["Close", "Tomorrow_Close", "Target_Up"]].head(10))

# ----- SELECT FEATURES AND TARGET -----

feature_cols = ["Close", "Volume", "SMA_10", "SMA_20", "RSI_14"]

X = df_ml[feature_cols]          # inputs to the model
y = df_ml["Target_Up"]           # label we want to predict

print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# 5) Train-test split (time-based: old data for training, recent data for testing)

n_rows = len(df_ml)
train_size = int(n_rows * 0.8)   # 80% train, 20% test

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# 6) Define and train the model

model = RandomForestClassifier(
    n_estimators=200,      # number of trees
    max_depth=5,           # limit depth to avoid overfitting for now
    random_state=42
)

model.fit(X_train, y_train)

# 7) Evaluate on the test set

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=2))

# ----- STEP 6: PLOTTING -----

plt.figure(figsize=(12, 8))

# Price + MA
plt.subplot(2, 1, 1)
plt.plot(df.index, df["Close"], label="Close")
plt.plot(df.index, df["SMA_10"], label="SMA 10")
plt.plot(df.index, df["SMA_20"], label="SMA 20")
plt.title(f"{TICKER} - Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()

# RSI
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

