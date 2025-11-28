
# ----- STEP 1: IMPORT LIBRARIES -----

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# ----- STEP 2: DOWNLOAD STOCK DATA -----

TICKER = "RELIANCE.NS"   # change this anytime

data = yf.download(TICKER, period="5y", interval="1d")

# ----- FIX MULTI-INDEX COLUMNS (IMPORTANT) -----
# Flatten the MultiIndex columns into simple names
data.columns = data.columns.get_level_values(0)


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

# ----- STEP 8: ADD MORE FEATURES -----

# 8.1: Daily return (percentage change from previous close)
df["Return_1d"] = df["Close"].pct_change()

# 8.2: 5-day return (percentage change compared to 5 days ago)
df["Return_5d"] = df["Close"].pct_change(5)

# 8.3: 10-day volatility (standard deviation of daily returns over last 10 days)
df["Volatility_10d"] = df["Return_1d"].rolling(window=10).std()

# 8.4: Price relative to SMA_10 (how far above/below its 10-day average)
df["Price_vs_SMA10"] = df["Close"] / df["SMA_10"]

# ----- STEP 12: TREND STRENGTH -----

# 12.1: Distance between SMA 10 and SMA 20 (trend direction indicator)
df["SMA_Diff"] = df["SMA_10"] - df["SMA_20"]

# 12.2: Ratio of SMA 10 to SMA 20 (scale-independent)
df["SMA_Ratio"] = df["SMA_10"] / df["SMA_20"]

# ----- STEP 13: ROLLING RETURNS -----

df["Return_3d_mean"] = df["Return_1d"].rolling(window=3).mean()
df["Return_7d_mean"] = df["Return_1d"].rolling(window=7).mean()

# ----- STEP 14: VOLATILITY CLUSTERING -----

df["Volatility_3d"] = df["Return_1d"].rolling(window=3).std()
df["Volatility_7d"] = df["Return_1d"].rolling(window=7).std()

# Add High-Low-Open columns
df["High"] = data["High"]
df["Low"] = data["Low"]
df["Open"] = data["Open"]

# ----- STEP 15: RANGE FEATURES -----

df["Range_HL"] = df["High"] - df["Low"]
df["Range_OC"] = df["Close"] - df["Open"]



# ----- STEP 7: BUILD ML DATASET -----

# Make sure index is clean
df = df.reset_index(drop=True)

# 1) Create Tomorrow_Close as a Series (NOT DataFrame)
df["Tomorrow_Close"] = df["Close"].shift(-1)

# ----- PHASE B STEP 1: 5-DAY FUTURE CLOSE -----

df["Close_5d"] = df["Close"].shift(-5)


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

# ----- PHASE A STEP 2: REGRESSION TARGET (TOMORROW'S RETURN) -----

df_ml["Return_1d_future"] = (df_ml["Tomorrow_Close"] - df_ml["Close"]) / df_ml["Close"]

print("\nSample of regression target (Return_1d_future):")
print(df_ml[["Close", "Tomorrow_Close", "Return_1d_future"]].head(10))


print(df_ml[["Close", "Tomorrow_Close", "Target_Up"]].head(10))

# ----- PHASE B STEP 2: 5-DAY TARGET (UP OR DOWN) -----

mask_5d = df["Close_5d"].notna()

df_5d = df.loc[mask_5d].copy()

df_5d["Target_Up_5d"] = (df_5d["Close_5d"] > df_5d["Close"]).astype(int)

print("\nSample of 5-day target:")
print(df_5d[["Close", "Close_5d", "Target_Up_5d"]].head(10))


# ----- SELECT FEATURES AND TARGET -----

feature_cols = [
    "Close",
    "Volume",
    "SMA_10",
    "SMA_20",
    "RSI_14",
    "Return_1d",
    "Return_5d",
    "Volatility_10d",
    "Price_vs_SMA10",
    
    # New features:
    "SMA_Diff",
    "SMA_Ratio",
    "Return_3d_mean",
    "Return_7d_mean",
    "Volatility_3d",
    "Volatility_7d",
    "Range_HL",
    "Range_OC",
]

X = df_ml[feature_cols]          # inputs to the model
y = df_ml["Target_Up"]           # label we want to predict

# ----- PHASE A STEP 3: REGRESSION FEATURES AND TARGET -----

X_reg = df_ml[feature_cols]                 # same features as classification
y_reg = df_ml["Return_1d_future"]           # numeric target

print("\nRegression X shape:", X_reg.shape)
print("Regression y shape:", y_reg.shape)


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

# ----- PHASE A STEP 4: TRAIN/TEST SPLIT FOR REGRESSION -----

n_rows_reg = len(df_ml)
train_size_reg = int(n_rows_reg * 0.8)

X_reg_train = X_reg.iloc[:train_size_reg]
y_reg_train = y_reg.iloc[:train_size_reg]

X_reg_test = X_reg.iloc[train_size_reg:]
y_reg_test = y_reg.iloc[train_size_reg:]

print("Regression train size:", X_reg_train.shape[0])
print("Regression test size:", X_reg_test.shape[0])

# ----- PHASE B STEP 3: FEATURES & TARGET FOR 5-DAY MODEL -----

X_5d = df_5d[feature_cols]
y_5d = df_5d["Target_Up_5d"]

train_size_5d = int(len(df_5d) * 0.8)

X_5d_train = X_5d.iloc[:train_size_5d]
y_5d_train = y_5d.iloc[:train_size_5d]

X_5d_test = X_5d.iloc[train_size_5d:]
y_5d_test = y_5d.iloc[train_size_5d:]

# ----- PHASE B STEP 4: TRAIN 5-DAY CLASSIFIER -----

model_5d = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

model_5d.fit(X_5d_train, y_5d_train)

# ----- PHASE B STEP 5: EVALUATE 5-DAY MODEL -----

y_5d_pred = model_5d.predict(X_5d_test)

acc_5d = accuracy_score(y_5d_test, y_5d_pred)

print("\n+---------------- 5-DAY MODEL ACCURACY ----------------+")
print(f"5-Day Test Accuracy: {acc_5d:.2f}")
print("+-----------------------------------------------------+")

print("\n5-Day Classification report:")
print(classification_report(y_5d_test, y_5d_pred, digits=2))

initial_capital_5d = 100_000


# ----- PHASE C (5-DAY BACKTEST) STEP 1: 5-DAY FUTURE RETURN -----

# 5-day future return based on Close and Close_5d
df_5d["Return_5d_future"] = (df_5d["Close_5d"] - df_5d["Close"]) / df_5d["Close"]

# ----- PHASE C STEP 2: PREPARE 5-DAY TEST DATA WITH PREDICTIONS -----

# Align df_5d test rows with X_5d_test / y_5d_test
df_5d_test = df_5d.iloc[train_size_5d:].copy()

# Attach model predictions and true label
df_5d_test["Pred_Class_5d"] = y_5d_pred          # model's 5-day prediction: 0 or 1
df_5d_test["True_Class_5d"] = y_5d_test.to_numpy()
df_5d_test["Future_Return_5d"] = df_5d_test["Return_5d_future"]

print("\n5-Day Backtest sample rows:")
print(df_5d_test[[
    "Close", "Close_5d", "Future_Return_5d", "Pred_Class_5d", "True_Class_5d"
]].head(10))

# ----- PHASE C STEP 3: STRATEGY RULES FOR 5-DAY TRADES -----

# Basic signal: model predicts UP over next 5 days
signal_model_up = df_5d_test["Pred_Class_5d"] == 1

# Trend filter: short-term trend is up
signal_trend_up = df_5d_test["SMA_10"] > df_5d_test["SMA_20"]

# RSI filter: avoid extreme zones
signal_rsi_ok = df_5d_test["RSI_14"].between(40, 70)

# Final entry condition: all rules must be true
entry_signal = signal_model_up & signal_trend_up & signal_rsi_ok

df_5d_test["Entry_Signal"] = entry_signal.astype(int)

print("\n5-Day strategy signal counts:")
print("Total test rows:", len(df_5d_test))
print("Rows where model says UP:", signal_model_up.sum())
print("Rows passing all filters (Entry_Signal=1):", df_5d_test["Entry_Signal"].sum())

# ----- PHASE C STEP 4: EQUITY CURVES FOR 5-DAY STRATEGY -----

initial_capital_5d = 100_000  # ₹1 lakh

# Buy & Hold using 5-day future returns (simplified overlapping view)
df_5d_test["BH_Equity_5d"] = (1 + df_5d_test["Future_Return_5d"]).cumprod() * initial_capital_5d

# Strategy: only apply 5-day returns when entry_signal is 1
df_5d_test["Strategy_Return_5d"] = df_5d_test["Future_Return_5d"] * df_5d_test["Entry_Signal"]

df_5d_test["Strategy_Equity_5d"] = (1 + df_5d_test["Strategy_Return_5d"]).cumprod() * initial_capital_5d


# ----- PHASE C STEP 5: SUMMARY STATS FOR 5-DAY STRATEGY -----

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



# ----- PHASE A STEP 5: TRAIN REGRESSION MODEL -----

reg_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

reg_model.fit(X_reg_train, y_reg_train)

# ----- PHASE A STEP 6: EVALUATE REGRESSION MODEL -----

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

# ----- PHASE A STEP 7: DIRECTION ACCURACY FROM REGRESSION -----

# True direction from regression target
true_direction = (y_reg_test > 0).astype(int)           # 1 if future return > 0

# Predicted direction from reg_model output
pred_direction = (y_reg_pred > 0).astype(int)

dir_acc = accuracy_score(true_direction, pred_direction)

print(f"\nDirection accuracy from regression model: {dir_acc:.2f}")



# 6) Define and train the model

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,        # was 5
    min_samples_leaf=5, # prevent extreme overfitting
    random_state=42
)

model.fit(X_train, y_train)

# 7) Evaluate on the test set

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

# print(f"Test Accuracy: {acc:.2f}")

# print(f"\033[92mTest Accuracy: {acc:.2f}\033[0m") #change colours

# print(f"\033[1;96mTest Accuracy: {acc:.2f}\033[0m") # change colours with bold

msg = f"\033[1;96mTest Accuracy: {acc:.2f}\033[0m"
print("+" + "-" * (len(msg) + 2) + "+")
print("| " + msg + " |")
print("+" + "-" * (len(msg) + 2) + "+")




print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=2))

# ----- STEP 10: FEATURE IMPORTANCE -----

importances = model.feature_importances_
importance_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

print("\nFeature importance (higher = more important):")
print(importance_series.round(3))

# ----- PHASE C STEP 1: PREPARE TEST DATA FOR BACKTEST -----

# df_ml has all rows; train_size is the 80% split index we already used
df_test_bt = df_ml.iloc[train_size:].copy()   # same rows as X_test / y_test

# Attach classifier predictions and true label
df_test_bt["Pred_Class"] = y_pred             # model's prediction: 0 or 1
df_test_bt["True_Class"] = y_test.to_numpy()  # actual 0 or 1

# We already created this earlier in PHASE A (1-day future return):
# df_ml["Return_1d_future"]
# So here we just rename it for clarity

df_test_bt["Future_Return"] = df_test_bt["Return_1d_future"]

print("\nBacktest sample rows:")
print(df_test_bt[["Close", "Tomorrow_Close", "Future_Return", "Pred_Class", "True_Class"]].head(10))

# ----- PHASE C STEP 2: BUY & HOLD EQUITY CURVE -----

initial_capital = 100_000  # ₹1 lakh

# Buy & hold: always in the market, apply all future returns
df_test_bt["BH_Equity"] = (1 + df_test_bt["Future_Return"]).cumprod() * initial_capital

# ----- PHASE C STEP 3: STRATEGY EQUITY CURVE (USING MODEL) -----

# When Pred_Class == 1 → we are “in the market”
# When Pred_Class == 0 → we are “out” (no return that day)

df_test_bt["Strategy_Return"] = df_test_bt["Future_Return"] * df_test_bt["Pred_Class"]

# Cumulative product of (1 + Strategy_Return) gives equity curve
df_test_bt["Strategy_Equity"] = (1 + df_test_bt["Strategy_Return"]).cumprod() * initial_capital

# ----- PHASE C STEP 4: SUMMARY STATS -----

final_bh = df_test_bt["BH_Equity"].iloc[-1]
final_strategy = df_test_bt["Strategy_Equity"].iloc[-1]

total_return_bh = (final_bh / initial_capital) - 1
total_return_strategy = (final_strategy / initial_capital) - 1

days_in_market = df_test_bt["Pred_Class"].sum()  # number of days model was "in"
total_days = len(df_test_bt)

print("\n+----------------- BACKTEST SUMMARY -----------------+")
print(f"Buy & Hold final equity:   ₹{final_bh:,.2f}  (Return: {total_return_bh*100:.2f}%)")
print(f"Strategy final equity:      ₹{final_strategy:,.2f}  (Return: {total_return_strategy*100:.2f}%)")
print(f"Days in test period:        {total_days}")
print(f"Days in market (strategy):  {days_in_market}")
print("+---------------------------------------------------+")

# ----- PHASE C STEP 5: PLOT EQUITY CURVES (OPTIONAL) -----

plt.figure(figsize=(12, 6))
plt.plot(df_test_bt.index, df_test_bt["BH_Equity"], label="Buy & Hold")
plt.plot(df_test_bt.index, df_test_bt["Strategy_Equity"], label="Strategy (Model)", linestyle="--")
plt.title(f"{TICKER} - Backtest: Buy & Hold vs Model Strategy")
plt.xlabel("Index (test period rows)")
plt.ylabel("Equity (₹)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



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

