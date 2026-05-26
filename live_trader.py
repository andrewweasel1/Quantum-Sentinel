import os
import json
import math
import logging
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
# FIXED: Corrected dunder method from **name** to __name__
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PRODUCTION POOL DISCOVERY
# ==============================================================================
def discover_production_pool() -> Dict[str, Dict[str, Any]]:
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
            
    logger.info(f"Active pool: Discovered {len(models_pool)} production models running concurrently.")
    return models_pool

# ==============================================================================
# 2. LIVE SANDBOX EXECUTION CYCLE
# ==============================================================================
def run_live_sandbox_cycle(live_market_df: pd.DataFrame) -> None:
    """
    Evaluates live market streams across all commissioned models simultaneously.
    Logs independent decisions to a partitioned dataset ledger for the dashboard.
    """
    pool = discover_production_pool()
    if not pool:
        # FIXED: Removed the garbage syntax error text attached to the return statement
        logger.warning("Sandbox pool is empty. Deploy the tournament using exporter.py first.")
        return

    new_log_entries = []
    timestamp = pd.Timestamp.now()
    
    # Extract strings/metadata safely as numpy arrays for fast iteration later
    tickers = live_market_df.get('ticker', pd.Series(['UNKNOWN'] * len(live_market_df))).values
    sentiments = live_market_df.get('llm_sentiment', pd.Series([np.nan] * len(live_market_df))).values

    # Process live data through every model in the sandbox pool
    for model_id, components in pool.items():
        bst = components["booster"]
        expected_features = components["features"]
        
        # 1. BATCH FEATURE ALIGNMENT (Eliminates iterrows bottleneck)
        aligned_data = {}
        for f in expected_features:
            if f in live_market_df.columns:
                aligned_data[f] = live_market_df[f].values
            else:
                aligned_data[f] = np.full(len(live_market_df), np.nan)
            
        aligned_df = pd.DataFrame(aligned_data)
        
        # 2. BATCH GPU INFERENCE (Eliminates per-row DMatrix creation)
        dmatrix = xgb.DMatrix(aligned_df)
        predictions = bst.predict(dmatrix)
        
        # 3. FAST ARRAY ITERATION 
        for idx, prediction_prob in enumerate(predictions):
            ticker = tickers[idx]
            status = "HELD"
            reason = "Confidence Below Threshold"
            signal = 0
            
            # Using the centralized threshold defined in config.py
            if prediction_prob > config.CONFIDENCE_THRESHOLD:
                status = "EXECUTED (PAPER)"
                reason = "Passed All Gates"
                signal = 1
                
                # --- LIVE CAPITAL ROUTING GATE ---
                # if model_id == "standard_technology_champion_v1" and signal == 1:
                #     execute_live_capital_order(ticker, entry_price, stop_loss)
                
            new_log_entries.append({
                "Timestamp": timestamp,
                "Model_ID": model_id,
                "Ticker": ticker,
                "ML_Confidence": float(prediction_prob),
                "LLM_Sentiment": sentiments[idx],
                "Status": status,
                "Veto_Reason": reason,
                "P/L": 0.0 
            })

    # 4. Append to the Unified Live Ledger (Optimized PyArrow I/O Partitioning)
    if new_log_entries:
        new_df = pd.DataFrame(new_log_entries)
        table = pa.Table.from_pandas(new_df)
        
        pq.write_to_dataset(
            table,
            root_path=config.LIVE_LOG_DIR,
            partition_cols=['Model_ID']
        )
        logger.info(f"Sandbox Cycle Complete. {len(new_log_entries)} telemetry lines safely appended to the dataset.")

# ==============================================================================
# 3. AUTHENTIC ALPACA ORDER EXECUTION
# ==============================================================================
def execute_live_capital_order(ticker: str, entry_price: float, stop_loss: float, side: str = "BUY", is_paper: bool = True) -> bool:
    """
    Routes authentic orders to the Alpaca Trading API.
    Toggles between Paper Trading and Live Capital based on the `is_paper` flag.
    Dynamically sizes the position based on live account buying power and risk tolerance.
    """
    # 1. Credential Routing
    api_key = os.environ.get('ALPACA_PAPER_API_KEY') if is_paper else os.environ.get('ALPACA_LIVE_API_KEY')
    secret_key = os.environ.get('ALPACA_PAPER_SECRET_KEY') if is_paper else os.environ.get('ALPACA_LIVE_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.error("Alpaca API credentials not found in environment variables.")
        return False

    # 2. Initialize the Trading Client
    trading_client = TradingClient(api_key, secret_key, paper=is_paper)
    
    try:
        # 3. Dynamic Position Sizing (Risk Manager Gate)
        account = trading_client.get_account()
        account_equity = float(account.equity)
        
        risk_allowance = account_equity * 0.02  
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            logger.error(f"Invalid stop-loss parameters for {ticker}. Trade aborted.")
            return False
            
        target_qty = math.floor(risk_allowance / risk_per_share)
        
        if target_qty <= 0:
            logger.warning(f"VETOED: Account equity (${account_equity}) insufficient to purchase {ticker} at safe risk thresholds.")
            return False

        # 4. Map the Order Parameters
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=target_qty,
            side=order_side,
            time_in_force=TimeInForce.GTC
        )
        
        # 5. Execute the Trade
        env_label = "PAPER SIMULATION" if is_paper else "LIVE CAPITAL"
        logger.info(f"[{env_label}] Routing {order_side.name} order for {target_qty} shares of {ticker}...")
        
        market_order = trading_client.submit_order(order_data=market_order_data)
        logger.info(f"Execution Successful! Alpaca Order ID: {market_order.id}")
        
        return True

    except Exception as e:
        logger.error(f"Order execution rejected by Alpaca for {ticker}.", exc_info=True)
        return False