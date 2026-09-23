"""
Machine Learning Engine for Stock Price Prediction
Implements strict time-series best practices, comprehensive technical indicator engineering,
chronological data validation, multi-model benchmarking, and autoregressive forecasting.
"""

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def generate_synthetic_data(ticker="SAMPLE", days=365, base_price=150.0):
    """
    Generates realistic synthetic stock market data using Geometric Brownian Motion
    with stochastic volatility and volume patterns. Used as a resilient fallback.
    """
    np.random.seed(abs(hash(ticker)) % (2**32))
    dates = pd.date_range(end=datetime.date.today(), periods=days, freq="B")
    n = len(dates)

    # Parameters for Geometric Brownian Motion
    dt = 1 / 252.0
    mu = 0.08  # 8% annual expected drift
    sigma = 0.25  # 25% annual volatility

    # Generate daily log returns
    random_shocks = np.random.normal(0, 1, n)
    returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks)

    prices = np.zeros(n)
    prices[0] = base_price
    for i in range(1, n):
        prices[i] = prices[i - 1] * returns[i]

    # Generate realistic high, low, open, close and volume
    open_prices = prices * (1 + np.random.normal(0, 0.004, n))
    close_prices = prices
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.008, n)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.008, n)))
    base_vol = 15_000_000
    volume = np.random.lognormal(mean=np.log(base_vol), sigma=0.4, size=n).astype(int)

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def fetch_stock_data(ticker="AAPL", period="1y"):
    """
    Fetches historical OHLCV data using yfinance.
    Flattens MultiIndex columns from newer yfinance versions and falls back to synthetic
    data if the network is unavailable or the ticker is invalid.
    """
    ticker_clean = ticker.strip().upper()
    try:
        data = yf.download(
            ticker_clean,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data is not None and not data.empty and len(data) >= 40:
            # Flatten MultiIndex columns if present (e.g. ('Close', 'AAPL') -> 'Close')
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]

            # Ensure required columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if all(col in data.columns for col in required_cols):
                cleaned_df = data[required_cols].copy()
                cleaned_df.dropna(inplace=True)
                if len(cleaned_df) >= 40:
                    return cleaned_df, False  # Not synthetic
    except Exception as e:
        print(f"Warning: yfinance fetch failed for {ticker_clean} ({e}). Falling back to simulation engine.")

    # Fallback to realistic synthetic series
    base_price = 180.0 if "AAPL" in ticker_clean else (250.0 if "TSLA" in ticker_clean else 120.0)
    days_map = {"3mo": 75, "6mo": 135, "1y": 252, "2y": 504, "5y": 1260}
    days = days_map.get(period, 252)
    return generate_synthetic_data(ticker=ticker_clean, days=days, base_price=base_price), True


def compute_technical_indicators(df):
    """
    Computes professional quantitative indicators and lag features:
    - Trend: SMA (10, 20, 50), EMA (12, 26)
    - Momentum: RSI (14), MACD & Signal Line
    - Volatility: Bollinger Bands (Upper, Lower, Width, %B), 20-day Rolling Volatility
    - Price Action & Lags: Returns, Lags (1, 2, 3, 5), High-Low Spread, Volume Ratio
    """
    data = df.copy()

    # Trend Indicators
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # Momentum: RSI (14 periods)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD & Signal Line
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # Volatility: Bollinger Bands (20 days, 2 std)
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = data["SMA_20"] + (2 * bb_std)
    data["BB_Lower"] = data["SMA_20"] - (2 * bb_std)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (data["SMA_20"] + 1e-9)
    data["BB_Pct"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"] + 1e-9)

    # Volatility & Returns
    data["Daily_Return"] = data["Close"].pct_change()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()

    # Price Spreads
    data["HL_Spread"] = (data["High"] - data["Low"]) / (data["Close"] + 1e-9)
    data["CO_Spread"] = (data["Close"] - data["Open"]) / (data["Open"] + 1e-9)

    # Volume Dynamics
    vol_sma_20 = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"] = data["Volume"] / (vol_sma_20 + 1e-9)

    # Lagged features (strictly past values, no future leakage)
    data["Lag_1"] = data["Close"].shift(1)
    data["Lag_2"] = data["Close"].shift(2)
    data["Lag_3"] = data["Close"].shift(3)
    data["Lag_5"] = data["Close"].shift(5)
    data["Return_Lag_1"] = data["Daily_Return"].shift(1)

    return data


def get_feature_columns():
    """List of predictive features used by machine learning models."""
    return [
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "BB_Pct",
        "Volatility_20",
        "HL_Spread",
        "CO_Spread",
        "Volume_Ratio",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_5",
        "Return_Lag_1",
    ]


def train_and_evaluate_models(df, test_size=0.2):
    """
    Executes ML training workflow following time-series best practices:
    - Computes next-day target: Close(t+1)
    - Chronological split (80% Train, 20% Test)
    - Independent RobustScaler fitted ONLY on training set
    - Trains Ridge, Random Forest, HistGradientBoosting, and Ensemble Blend
    - Evaluates RMSE, MAE, MAPE, R2, and Directional Accuracy
    """
    feature_df = compute_technical_indicators(df)

    # Target is next trading day's close price
    feature_df["Target_Close"] = feature_df["Close"].shift(-1)

    feature_cols = get_feature_columns()

    # The last row has features for today, but its Target_Close is unknown (future)
    latest_row_features = feature_df.iloc[[-1]][feature_cols].copy()
    latest_date = feature_df.index[-1].strftime("%Y-%m-%d")
    latest_close = float(feature_df["Close"].iloc[-1])

    # Drop rows with NaN (from rolling windows and the final target row)
    clean_data = feature_df.dropna(subset=feature_cols + ["Target_Close"]).copy()

    if len(clean_data) < 30:
        raise ValueError("Not enough historical data points after computing indicators. Please select a longer timeframe.")

    X = clean_data[feature_cols]
    y = clean_data["Target_Close"]
    dates = clean_data.index.strftime("%Y-%m-%d").tolist()
    actual_closes = clean_data["Close"].values

    # Strict Chronological Split (No Shuffling)
    split_idx = int(len(clean_data) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = dates[split_idx:]
    test_actual_closes = actual_closes[split_idx:]

    # Scale features strictly fitted on training data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Candidate Models
    models = {
        "Ridge": Ridge(alpha=2.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=8, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150, max_depth=5, learning_rate=0.06, random_state=42
        ),
    }

    model_metrics = {}
    test_predictions = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        fitted_models[name] = model
        test_predictions[name] = preds

        # Evaluation metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        mape = float(np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100)
        r2 = float(r2_score(y_test, preds))

        # Directional Accuracy: did the model predict the correct direction of next day move?
        actual_direction = np.sign(y_test.values - test_actual_closes)
        pred_direction = np.sign(preds - test_actual_closes)
        dir_acc = float(np.mean(actual_direction == pred_direction) * 100)

        model_metrics[name] = {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
            "directional_accuracy": round(dir_acc, 1),
        }

    # Ensemble Blend (Weighted combination based on inverse RMSE)
    weights = [1.0 / model_metrics[name]["rmse"] for name in models]
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    ensemble_preds = np.zeros_like(y_test.values, dtype=float)
    for i, name in enumerate(models):
        ensemble_preds += norm_weights[i] * test_predictions[name]

    ensemble_rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))
    ensemble_mae = float(mean_absolute_error(y_test, ensemble_preds))
    ensemble_mape = float(np.mean(np.abs((y_test.values - ensemble_preds) / y_test.values)) * 100)
    ensemble_r2 = float(r2_score(y_test, ensemble_preds))
    ens_actual_dir = np.sign(y_test.values - test_actual_closes)
    ens_pred_dir = np.sign(ensemble_preds - test_actual_closes)
    ensemble_dir_acc = float(np.mean(ens_actual_dir == ens_pred_dir) * 100)

    model_metrics["Ensemble Blend"] = {
        "rmse": round(ensemble_rmse, 2),
        "mae": round(ensemble_mae, 2),
        "mape": round(ensemble_mape, 2),
        "r2": round(ensemble_r2, 4),
        "directional_accuracy": round(ensemble_dir_acc, 1),
    }
    test_predictions["Ensemble Blend"] = ensemble_preds

    # Extract Feature Importances from Random Forest
    rf_model = fitted_models["Random Forest"]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:8]  # Top 8
    top_features = [
        {"feature": feature_cols[idx], "importance": round(float(importances[idx] * 100), 1)}
        for idx in sorted_idx
    ]

    return {
        "fitted_models": fitted_models,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_metrics": model_metrics,
        "test_dates": test_dates,
        "test_actual": [round(float(v), 2) for v in y_test.values],
        "test_predictions": {k: [round(float(v), 2) for v in p] for k, p in test_predictions.items()},
        "top_features": top_features,
        "latest_features": latest_row_features,
        "latest_date": latest_date,
        "latest_close": latest_close,
        "historical_df": feature_df,
    }


def multi_step_forecast(
    model_name,
    results,
    forecast_days=14,
):
    """
    Executes multi-step recursive autoregressive forecasting for future trading days.
    Generates predicted trajectory and expanding 95% confidence intervals.
    """
    fitted_models = results["fitted_models"]
    scaler = results["scaler"]
    feature_cols = results["feature_cols"]
    latest_features = results["latest_features"].copy()
    latest_close = results["latest_close"]
    latest_date_str = results["latest_date"]
    latest_date = datetime.datetime.strptime(latest_date_str, "%Y-%m-%d").date()

    # Selected model or ensemble weights
    is_ensemble = model_name == "Ensemble Blend"
    if not is_ensemble:
        model = fitted_models.get(model_name, fitted_models["Gradient Boosting"])

    # Determine residual standard error for confidence cone
    rmse_err = results["model_metrics"][model_name]["rmse"]

    forecast_dates = []
    forecast_prices = []
    upper_bounds = []
    lower_bounds = []

    curr_features = latest_features.copy()
    curr_price = latest_close
    curr_date = latest_date

    # Approximate price history queue for dynamic lag updating
    price_history = [curr_price] * 10

    for step in range(1, forecast_days + 1):
        # Advance to next business day
        curr_date += datetime.timedelta(days=1)
        while curr_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
            curr_date += datetime.timedelta(days=1)

        # Scale features
        scaled_x = scaler.transform(curr_features[feature_cols])

        # Predict next price
        if is_ensemble:
            p_ridge = fitted_models["Ridge"].predict(scaled_x)[0]
            p_rf = fitted_models["Random Forest"].predict(scaled_x)[0]
            p_gb = fitted_models["Gradient Boosting"].predict(scaled_x)[0]
            w_ridge = 1.0 / results["model_metrics"]["Ridge"]["rmse"]
            w_rf = 1.0 / results["model_metrics"]["Random Forest"]["rmse"]
            w_gb = 1.0 / results["model_metrics"]["Gradient Boosting"]["rmse"]
            pred_price = float((p_ridge * w_ridge + p_rf * w_rf + p_gb * w_gb) / (w_ridge + w_rf + w_gb))
        else:
            pred_price = float(model.predict(scaled_x)[0])

        # Expanding confidence interval (t^0.5 scaled error)
        margin = 1.96 * rmse_err * np.sqrt(step / 3.0)
        upper = pred_price + margin
        lower = max(0.01, pred_price - margin)

        forecast_dates.append(curr_date.strftime("%Y-%m-%d"))
        forecast_prices.append(round(pred_price, 2))
        upper_bounds.append(round(upper, 2))
        lower_bounds.append(round(lower, 2))

        # Update price history queue
        price_history.append(pred_price)
        if len(price_history) > 20:
            price_history.pop(0)

        # Autoregressively update lag features for step + 1
        curr_features["Lag_1"] = price_history[-1]
        curr_features["Lag_2"] = price_history[-2]
        curr_features["Lag_3"] = price_history[-3]
        curr_features["Lag_5"] = price_history[-5]
        curr_features["SMA_10"] = np.mean(price_history[-10:])
        curr_features["EMA_12"] = curr_features["EMA_12"].values[0] * 0.85 + pred_price * 0.15
        curr_price = pred_price

    return {
        "dates": forecast_dates,
        "prices": forecast_prices,
        "upper_bounds": upper_bounds,
        "lower_bounds": lower_bounds,
    }


def generate_trading_signal(current_price, forecasted_price, rsi, macd, macd_signal):
    """
    Generates quantitative algorithmic trading recommendation by combining
    predicted trend trajectory with momentum indicators (RSI & MACD).
    """
    expected_return_pct = ((forecasted_price - current_price) / current_price) * 100
    macd_diff = macd - macd_signal

    score = 0
    rationales = []

    # ML Price Target Evaluation
    if expected_return_pct > 5.0:
        score += 2
        rationales.append(f"ML forecast projects a robust bullish gain of +{expected_return_pct:.1f}%.")
    elif expected_return_pct > 1.5:
        score += 1
        rationales.append(f"ML forecast indicates moderate upward momentum (+{expected_return_pct:.1f}%).")
    elif expected_return_pct < -5.0:
        score -= 2
        rationales.append(f"ML forecast projects a significant downward move of {expected_return_pct:.1f}%.")
    elif expected_return_pct < -1.5:
        score -= 1
        rationales.append(f"ML forecast indicates moderate downward risk ({expected_return_pct:.1f}%).")
    else:
        rationales.append("ML forecast indicates range-bound, sideways price consolidation.")

    # RSI Evaluation
    if rsi < 32:
        score += 2
        rationales.append(f"RSI is oversold at {rsi:.1f}, indicating high potential for mean-reversion rally.")
    elif rsi > 68:
        score -= 2
        rationales.append(f"RSI is overbought at {rsi:.1f}, signaling elevated short-term pullback risk.")
    elif 45 <= rsi <= 55:
        rationales.append(f"RSI is neutral at {rsi:.1f}.")

    # MACD Trend Confluence
    if macd_diff > 0:
        score += 1
        rationales.append("MACD line sits above the Signal line, confirming positive momentum.")
    else:
        score -= 1
        rationales.append("MACD line sits below the Signal line, confirming bearish pressure.")

    # Final Signal Determination
    if score >= 3:
        signal = "STRONG BUY"
        signal_color = "#10b981"  # Emerald
        confidence = min(94, 75 + abs(score) * 4)
    elif score in (1, 2):
        signal = "BUY"
        signal_color = "#34d399"
        confidence = min(85, 68 + abs(score) * 4)
    elif score in (-1, -2):
        signal = "SELL"
        signal_color = "#f87171"
        confidence = min(85, 68 + abs(score) * 4)
    elif score <= -3:
        signal = "STRONG SELL"
        signal_color = "#ef4444"  # Red
        confidence = min(94, 75 + abs(score) * 4)
    else:
        signal = "HOLD / NEUTRAL"
        signal_color = "#f59e0b"  # Amber
        confidence = 65

    return {
        "signal": signal,
        "signal_color": signal_color,
        "confidence": confidence,
        "expected_return_pct": round(expected_return_pct, 2),
        "rationales": rationales,
    }
