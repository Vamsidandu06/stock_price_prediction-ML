"""
Flask Web Application for Interactive Stock Price Prediction
Serves RESTful APIs for market quotes, technical indicators, ML model comparisons,
and autoregressive price projections.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import ml_engine

app = Flask(__name__)

# Popular market tickers for metadata lookups
POPULAR_TICKERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla, Inc.",
    "META": "Meta Platforms, Inc.",
    "BTC-USD": "Bitcoin USD",
}


@app.route("/")
def index():
    """Renders the primary dashboard interface."""
    return render_template("index.html")


@app.route("/api/quote", methods=["GET"])
def get_quote():
    """Fetches real-time price snapshot and summary profile for a given ticker."""
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    name = POPULAR_TICKERS.get(ticker, ticker)

    try:
        t = yf.Ticker(ticker)
        info = t.fast_info if hasattr(t, "fast_info") else {}
        current_price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)

        if current_price is None or prev_close is None:
            # Fallback to fetching recent history
            df, _ = ml_engine.fetch_stock_data(ticker, period="5d")
            current_price = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price

        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0.0

        return jsonify({
            "ticker": ticker,
            "name": name,
            "price": round(float(current_price), 2),
            "prev_close": round(float(prev_close), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "currency": getattr(info, "currency", "USD"),
        })
    except Exception as e:
        # Resilient fallback
        df, _ = ml_engine.fetch_stock_data(ticker, period="5d")
        curr = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else curr
        chg = curr - prev
        return jsonify({
            "ticker": ticker,
            "name": name,
            "price": round(curr, 2),
            "prev_close": round(prev, 2),
            "change": round(chg, 2),
            "change_pct": round((chg / prev) * 100, 2),
            "currency": "USD",
        })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction pipeline:
    1. Fetches historical stock data
    2. Calculates quantitative technical features
    3. Executes chronological train-test split and trains 4 models
    4. Evaluates RMSE, MAE, MAPE, R2, and Directional Accuracy
    5. Computes multi-step autoregressive future forecast with confidence intervals
    6. Formulates quantitative trading signals
    """
    data = request.get_json() or {}
    ticker = data.get("ticker", "AAPL").strip().upper()
    period = data.get("period", "1y")
    model_name = data.get("model_name", "Gradient Boosting")
    forecast_days = int(data.get("forecast_days", 14))

    # Constrain forecast days between 5 and 60
    forecast_days = max(5, min(forecast_days, 60))

    try:
        # 1. Fetch data
        df, is_synthetic = ml_engine.fetch_stock_data(ticker, period=period)

        # 2. Train and evaluate all models
        results = ml_engine.train_and_evaluate_models(df, test_size=0.2)

        # 3. Autoregressive future forecast
        forecast = ml_engine.multi_step_forecast(
            model_name=model_name,
            results=results,
            forecast_days=forecast_days,
        )

        # 4. Generate quantitative trading signal
        latest_row = results["historical_df"].iloc[-1]
        current_price = results["latest_close"]
        projected_price = forecast["prices"][-1]
        rsi_val = float(latest_row.get("RSI_14", 50.0))
        macd_val = float(latest_row.get("MACD", 0.0))
        macd_sig = float(latest_row.get("MACD_Signal", 0.0))

        signal_info = ml_engine.generate_trading_signal(
            current_price=current_price,
            forecasted_price=projected_price,
            rsi=rsi_val,
            macd=macd_val,
            macd_signal=macd_sig,
        )

        # 5. Extract historical candlestick & indicator series for plotting
        hist_df = results["historical_df"].dropna(subset=["Close", "Open", "High", "Low"]).copy()
        hist_dates = hist_df.index.strftime("%Y-%m-%d").tolist()

        return jsonify({
            "status": "success",
            "ticker": ticker,
            "period": period,
            "model_name": model_name,
            "forecast_days": forecast_days,
            "is_synthetic": is_synthetic,
            "latest_close": round(current_price, 2),
            "latest_date": results["latest_date"],
            "model_metrics": results["model_metrics"],
            "top_features": results["top_features"],
            "test_dates": results["test_dates"],
            "test_actual": results["test_actual"],
            "test_predictions": results["test_predictions"],
            "forecast": forecast,
            "trading_signal": signal_info,
            "history": {
                "dates": hist_dates,
                "open": [round(float(v), 2) for v in hist_df["Open"].values],
                "high": [round(float(v), 2) for v in hist_df["High"].values],
                "low": [round(float(v), 2) for v in hist_df["Low"].values],
                "close": [round(float(v), 2) for v in hist_df["Close"].values],
                "volume": [int(v) for v in hist_df["Volume"].values],
                "sma_10": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("SMA_10", [])],
                "sma_50": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("SMA_50", [])],
                "ema_12": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("EMA_12", [])],
                "ema_26": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("EMA_26", [])],
                "bb_upper": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("BB_Upper", [])],
                "bb_lower": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("BB_Lower", [])],
                "rsi_14": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("RSI_14", [])],
                "macd": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("MACD", [])],
                "macd_signal": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("MACD_Signal", [])],
                "macd_hist": [round(float(v), 2) if not pd.isna(v) else None for v in hist_df.get("MACD_Hist", [])],
            },
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    print("Starting Stock Price Prediction Web Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
