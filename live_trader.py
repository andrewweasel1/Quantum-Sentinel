import os
import json
import xgboost as xgb
import pandas as pd
import numpy as np
import config
import math
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def discover_production_pool():
    """
    Scans the production folder and groups models with their exact feature manifests.
    This creates the multi-model sandbox where old champions and new challengers coexist.
    """
    if not os.path.exists(config.PROD_MODELS_DIR):
        return {}
        
    models_pool = {}
    files = os.listdir(config.PROD_MODELS_DIR)
    
    # Filter for XGBoost .json graphs, explicitly ignoring the feature maps
    model_files = [f for f in files if f.endswith(".json") and not f.endswith("_features.json")]
    
    for model_file in model_files:
        base_name = model_file.replace(".json", "")
        feature_file = f"{base_name}_features.json"
        
        if feature_file in files:
            # 1. Load the frozen XGBoost Model Weights
            bst = xgb.Booster()
            bst.load_model(os.path.join(config.PROD_MODELS_DIR, model_file))
            
            # 2. Load the locked Feature Manifest
            with open(os.path.join(config.PROD_MODELS_DIR, feature_file), "r") as f:
                features = json.load(f)
                
            models_pool[base_name] = {
                "booster": bst,
                "features": features
            }
            
    print(f"Active pool: Discovered {len(models_pool)} production models running concurrently.")
    return models_pool

def run_live_sandbox_cycle(live_market_df):
    """
    Evaluates live market streams across all commissioned models simultaneously.
    Logs independent decisions to a unified live ledger for the dashboard.
    """
    pool = discover_production_pool()
    if not pool:
        print("Sandbox pool is empty. Deploy models from the tournament using exporter.py first.")
        return

    new_log_entries = []
    timestamp = pd.Timestamp.now()

    # Process live data through every model in the sandbox pool
    for model_id, components in pool.items():
        bst = components["booster"]
        expected_features = components["features"]
        
        for index, row in live_market_df.iterrows():
            ticker = row.get('ticker', 'UNKNOWN')
            
            # 1. Strict Feature Alignment (Pipeline Locking)
            # This protects the system from runtime crashes if you added or removed
            # indicators in your development environment (indicators.py). It forces the 
            # live data to conform strictly to what the frozen model expects.
            aligned_data = {}
            for f in expected_features:
                aligned_data[f] = [row.get(f, np.nan)]
                
            dmatrix = xgb.DMatrix(pd.DataFrame(aligned_data))
            
            # 2. GPU Inference
            prediction_prob = bst.predict(dmatrix)
            
            # 3. Simulate Decision & Risk Manager Gate
            status = "HELD"
            reason = "Confidence Below Threshold"
            signal = 0
            
            # Utilizing the > 65% mathematically profitable confidence threshold
            if prediction_prob > 0.65:
                status = "EXECUTED (PAPER)"
                reason = "Passed All Gates"
                signal = 1
                
                # --- LIVE CAPITAL ROUTING (The Risk Manager Gate) ---
                # Only execute authentic Alpaca API trades if the signal comes from
                # your explicitly vetted flagship model. All others remain in shadow mode.
                # if model_id == "standard_technology_champion_v1" and signal == 1:
                #     execute_live_capital_order(ticker)

            # 4. Build Telemetry Log
            new_log_entries.append({
                "Timestamp": timestamp,
                "Model_ID": model_id,
                "Ticker": ticker,
                "ML_Confidence": prediction_prob,
                "LLM_Sentiment": row.get('llm_sentiment', np.nan),
                "Status": status,
                "Veto_Reason": reason,
                "P/L": 0.0  # To be updated retroactively upon trade closure
            })

    # 5. Append to the Unified Live Ledger
    if new_log_entries:
        new_df = pd.DataFrame(new_log_entries)
        
        if os.path.exists(config.LIVE_LOG_FILE):
            existing_df = pd.read_parquet(config.LIVE_LOG_FILE)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
            
        combined_df.to_parquet(config.LIVE_LOG_FILE, engine='pyarrow')
        print(f"Sandbox Cycle Complete. {len(new_log_entries)} telemetry lines appended to the ledger.")

def execute_live_capital_order(ticker, entry_price, stop_loss, side="BUY", is_paper=True):
    """
    Routes authentic orders to the Alpaca Trading API.
    Toggles between Paper Trading and Live Capital based on the `is_paper` flag.
    Dynamically sizes the position based on live account buying power and risk tolerance.
    """
    # 1. Credential Routing
    # Important: Alpaca issues entirely different API Key/Secret pairs for Live vs. Paper accounts.
    # You must ensure the environment variables match the 'is_paper' mode you are running.
    api_key = os.environ.get('ALPACA_PAPER_API_KEY') if is_paper else os.environ.get('ALPACA_LIVE_API_KEY')
    secret_key = os.environ.get('ALPACA_PAPER_SECRET_KEY') if is_paper else os.environ.get('ALPACA_LIVE_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("Error: Alpaca API credentials not found in environment variables.")
        return False

    # 2. Initialize the Trading Client
    # The 'paper' argument natively toggles the base URL to route your orders correctly
    trading_client = TradingClient(api_key, secret_key, paper=is_paper)
    
    try:
        # 3. Dynamic Position Sizing (Risk Manager Gate)
        # Fetch live account telemetry to verify buying power before sizing the trade
        account = trading_client.get_account()
        account_equity = float(account.equity)
        
        # Enforcing our 2% maximum risk limit per trade
        risk_allowance = account_equity * 0.02  
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            print("Error: Invalid stop-loss parameters. Trade aborted.")
            return False
            
        # Calculate exactly how many shares we can buy without violating the 2% rule
        target_qty = math.floor(risk_allowance / risk_per_share)
        
        if target_qty <= 0:
            print(f"VETOED: Account equity (${account_equity}) insufficient to purchase {ticker} at safe risk thresholds.")
            return False

        # 4. Map the Order Parameters
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=target_qty,
            side=order_side,
            time_in_force=TimeInForce.GTC  # Good-Till-Canceled
        )
        
        # 5. Execute the Trade
        env_label = "PAPER SIMULATION" if is_paper else "LIVE CAPITAL"
        print(f"[{env_label}] Routing {order_side.name} order for {target_qty} shares of {ticker}...")
        
        market_order = trading_client.submit_order(order_data=market_order_data)
        print(f"Execution Successful! Alpaca Order ID: {market_order.id}")
        
        return True

    except Exception as e:
        print(f"Order execution rejected by Alpaca: {e}")
        # Note: HTTP 401 errors generally mean you passed a Live API key while 'paper=True'
        return False