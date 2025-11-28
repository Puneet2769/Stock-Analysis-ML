# 📊 Personal Stock ML Research — Multi-script Pipeline  
### Multiple prototypes, experiments and backtests for single-ticker equity signals (personal project — not Kaggle)

This repository is a personal research workspace for building technical indicators, training ML models (classification & regression), and running simple backtests. It contains several iterations of the pipeline (phase1 → phase4), backups, test scripts, and a final pipeline script. These are research artifacts — experiments, not production trading code.

---

## ⚙️ Quick overview

- Purpose: explore technical indicators (SMA, RSI, MACD, Bollinger, ADX, returns, volatility), train ML models (RandomForest / regression / LightGBM), and compare simple strategy backtests (1-day, 5-day).
- Project is **personal**: you run it locally, analyze outputs, and iterate rapidly.
- Several versions of the pipeline exist — keep the ones you want and archive the rest. Filenames include backups and exploratory tests.

---

## 🚩 Important note (read first)

This is research code with simplified backtests (no costs/slippage, full allocation). Do **not** use as-is for live trading. Backtests are illustrative and meant for experimental comparisons only.

---

## ▶️ Which script to run (recommended order)

1. `MAIN-phase1_stock_indicators_with_regression_5day_backtest.py`  
   - This looks like your main orchestrator combining indicators → models → 5-day backtest. Run this first to reproduce the main pipeline behavior.

2. `stock_ml_pipeline_after_MAIN.py` or `FINAL_stock_ml_pipeline_after_MAIN_diff_stocks_too.py`  
   - Use these to run the more polished/final pipeline or to test multiple tickers.

3. `phase1_stock_indicators_with_regression.py` / `phase1_stock_indicators_with_regression_5day.py`  
   - Individual phase scripts (indicator builder + model training). Good for stepping through parts.

4. Backup or staged files (do not run unless you know why):  
   - `phase1_stock_indicators_Backup_till_phase2.py`  
   - `phase1_stock_indicators_backup_2_till_phase4.py`  
   - `phase1_stock_indicators_with_regression_5day_backtest copy.py`

5. Quick tests / scratch files:  
   - `testing_1.py`  
   - `testing_2.py`

If you want a single canonical entrypoint, rename your chosen MAIN file to `main.py` and run:

```bash
python main.py
```
Or run directly:

bash
Copy code
python MAIN-phase1_stock_indicators_with_regression_5day_backtest.py
```bash
## 📁 Current repository structure (mirror of your working folder)
css
Copy code
├── FINAL_stock_ml_pipeline_after_MAIN_diff_stocks_too.py
├── MAIN-phase1_stock_indicators_with_regression_5day_backtest.py
├── phase1_stock_indicators_backup_2_till_phase4.py
├── phase1_stock_indicators_Backup_till_phase2.py
├── phase1_stock_indicators_with_regression.py
├── phase1_stock_indicators_with_regression_5day.py
├── phase1_stock_indicators_with_regression_5day_backtest copy.py
├── stock_ml_pipeline_after_MAIN.py
├── testing_1.py
├── testing_2.py
├── requirements.txt
└── README.md
(If any filename is slightly different locally, keep the version you actually use. The above list is taken from your screenshot.)
```
🛠️ How the pieces fit (short)
Indicator builders: generate df_ind with Close/Volume/MA/RSI/MACD/BB/ADX/volatility returns.

Dataset builders: create 1-day and 5-day classification/regression datasets and split into train/test by TRAIN_FRACTION.

Models: RandomForest classifiers and regressors (and LightGBM in some scripts) for signals and return prediction.

Backtests: simple strategy vs buy & hold; 5-day strategy adds trend + RSI filters.

Plots: matplotlib visual checks (price, SMA, RSI).

Outputs: printed metrics, feature importances, and optional CSVs/plots you save in an outputs/ folder.

✅ Recommendations / housekeeping (short, actionable)
Pick one MAIN entrypoint (rename to main.py) and keep that as canonical. Delete or move backups to archive/ to avoid confusion.

Create an outputs/ folder and save plots / CSVs there instead of cluttering the repo root.

Version your experiments with short tags in filenames (e.g., v1, v2) instead of copy, backup. It’s cleaner.

Add a small run.sh or Makefile to create venv and run the main script — I can add that for you.

Store important params at top of MAIN (TICKER, PERIOD, INTERVAL, TRAIN_FRACTION) and avoid hardcoding them deep in code. You already have constants — good.

Save model artifacts (joblib.dump) and encoders if you plan to replicate results later.

🔧 Minimal requirements
Add a requirements.txt with at least:

text
Copy code
pandas
numpy
yfinance
matplotlib
scikit-learn
lightgbm   # optional if you use it in some scripts
(If you want, I’ll pin exact versions from your environment.)

👤 Author
Puneet Poddar
Kaggle: https://www.kaggle.com/puneet2769
