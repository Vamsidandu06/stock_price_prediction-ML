# AlphaPredict.AI - Stock Price Prediction & Technical Analysis

An interactive full-stack Machine Learning web application designed for quantitative stock price forecasting, technical indicator analysis, and time-series model benchmarking.

![AlphaPredict Terminal](static/css/style.css)

---

## 🌟 Key Features

- **Live Market Ingestion**: Fetches real-time OHLCV data for any stock, ETF, or crypto ticker (e.g. `AAPL`, `NVDA`, `TSLA`, `MSFT`, `AMZN`, `BTC-USD`) via `yfinance`, with an offline simulation fallback for 100% uptime.
- **Quantitative Technical Indicators**:
  - **Trend**: Simple Moving Averages (`SMA 10`, `SMA 20`, `SMA 50`), Exponential Moving Averages (`EMA 12`, `EMA 26`).
  - **Momentum**: Relative Strength Index (`RSI 14`), Moving Average Convergence Divergence (`MACD 12, 26, 9` & Signal Line).
  - **Volatility**: Bollinger Bands (`Upper`, `Lower`, `%B`, `Width`), 20-Day Rolling Historical Volatility.
  - **Price Action & Lags**: Autoregressive price lags ($t-1, t-2, t-3, t-5$), return lags, High-Low spread, Close-Open spread, and Volume-to-SMA ratio.
- **Machine Learning Rigor (Zero Data Leakage)**:
  - **Chronological Split**: Strictly splits time-series data chronologically (80% Train, 20% Test) without random shuffling.
  - **Independent Scaling**: `RobustScaler` is fit solely on training data to prevent look-ahead bias.
  - **Model Comparison**: Benchmarks 4 models side-by-side:
    1. **Gradient Boosting** (`HistGradientBoostingRegressor` - fast, handles complex non-linear tabular features)
    2. **Random Forest Regressor** (120 decision trees with depth control)
    3. **Ridge Regression** (L2-regularized linear baseline)
    4. **Ensemble Blend** (Inverse-RMSE weighted combination of all models)
  - **Out-of-Sample Metrics**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), $R^2$ Score, and Directional Accuracy (% correct market direction predictions).
- **Multi-Step Recursive Forecasting**:
  - Projects future stock price trajectories $5$ to $45$ business days into the future.
  - Computes expanding $95\%$ confidence interval bounds based on empirical model error.
- **Algorithmic Trading Signal**:
  - Synthesizes projected returns, RSI oversold/overbought extremes, and MACD momentum crossover into an actionable signal (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`) with confidence scoring.
- **Modern Interactive Dashboard**:
  - Dark glassmorphic financial terminal theme.
  - Interactive Plotly.js charts (candlesticks, overlays, volume, test set predictions, and future forecast cone).
  - Dedicated technical oscillator subplots and feature importance rankings.

---

## 📁 Project Structure

```
stock-prediction-app/
├── app.py                  # Flask web server and JSON REST API endpoints
├── ml_engine.py            # Quantitative feature engineering, model training & forecasting
├── templates/
│   └── index.html          # Responsive HTML5 dashboard layout
├── static/
│   ├── css/
│   │   └── style.css       # Dark financial terminal glassmorphism styling
│   └── js/
│       └── app.js          # Plotly.js charts, async API queries, and state management
├── requirements.txt        # Python dependency specification
├── run.bat                 # 1-click Windows launcher
└── README.md               # Documentation and usage guide
```

---

## 🚀 Quick Start Instructions

### 1. Prerequisites
Ensure Python (3.9+) is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
Run via the terminal:
```bash
python app.py
```
*Or on Windows, simply double-click `run.bat`.*

### 4. Open in Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## 📊 How to Use the Dashboard

1. **Search Any Ticker**: Enter any ticker symbol (e.g. `NVDA`, `TSLA`, `MSFT`) or click the trending chips.
2. **Select Historical Range**: Choose `3M`, `6M`, `1Y`, `2Y`, or `5Y` to define the training dataset size.
3. **Select ML Model**: Choose between Gradient Boosting, Random Forest, Ridge Regression, or the Ensemble Blend.
4. **Adjust Horizon**: Slide the horizon control to project 5 to 45 business days ahead.
5. **Toggle Overlays**: Turn on/off SMAs, EMAs, or Bollinger Bands directly on the live chart.
6. **Analyze Tabs**:
   - **Price & ML Forecast**: Interactive Candlestick + Future Forecast Cone.
   - **Model Benchmarks**: Test set error table and actual vs predicted overlay.
   - **Technical Oscillators**: RSI and MACD subcharts.
   - **Feature Importance**: Most impactful predictive factors.

---

## ⚠️ Disclaimer
*This software is intended for educational, research, and technical analysis demonstration purposes only. Machine learning forecasts are probabilistic estimates and should not be considered financial or investment advice.*
