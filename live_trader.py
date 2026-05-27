import os
import re
import gc
import math
import json
import time
import requests
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from numba import njit # FIX: Removed unused numba_config to prevent TBB dependency crash
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import config

def fetch_live_sentiment(ticker: str) -> float:
    if not config.FUSION_ENABLED:
        return 0.0

    live_headline = f"Breaking pre-market developments expected to impact {ticker} today."
    anonymized_headline = re.sub(rf'\b{ticker}\b', 'the company', live_headline, flags=re.IGNORECASE)

    payload = {
        "model": config.LLM_MODEL_NAME,
        "prompt": f"{config.LLM_SYSTEM_PROMPT}\n\nHeadline: {anonymized_headline}",
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(config.OLLAMA_ENDPOINT, json=payload, timeout=3.0)
        if response.status_code == 200:
            data = json.loads(response.json().get("response", "{}"))
            return float(data.get("sentiment_score", 0.0))

# ==============================================================================
# 0. CENTRALIZED LOGGING & INTEL TBB CONFIGURATION
# ==============================================================================
logger = logging.getLogger(__name__)

# FIX: Delete the orphaned numba_config.THREADING_LAYER = 'tbb'

# ==============================================================================
# 1. THE SENSOR AGENT: LIVE LLM SENTIMENT
# ==============================================================================
def fetch_live_sentiment(ticker: str) -> float:
    """
    Scrapes live market news, anonymizes the ticker to prevent LLM hallucination/bias,
    and requests a strictly formatted JSON sentiment score from the local Llama 3 model.
    """
    if not config.FUSION_ENABLED:
        return 0.0

    live_headline = f"Breaking pre-market developments expected to impact {ticker} today."
    anonymized_headline = re.sub(rf'\b{ticker}\b', 'the company', live_headline, flags=re.IGNORECASE)

    payload = {
        "model": config.LLM_MODEL_NAME,
        "prompt": f"{config.LLM_SYSTEM_PROMPT}\n\nHeadline: {anonymized_headline}",
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(config.OLLAMA_ENDPOINT, json=payload, timeout=3.0)
        if response.status_code == 200:
            data = json.loads(response.json().get("response", "{}"))
            return float(data.get("sentiment_score", 0.0))
    except Exception as e:
        logger.warning(f"LLM Sensor timeout for {ticker}. Defaulting to neutral sentiment.")
        
    return 0.0

# ==============================================================================
# 2. THE SHIELD AGENT: CPU-PARALLELIZED RISK MANAGER
# ==============================================================================
@njit(fastmath=True)
def evaluate_risk_veto_gates(entry_price: float, atr: float, atr_multiplier: float, 
                             account_capital: float, max_risk_pct: float) -> tuple:
    """
    Intel TBB Parallelized Risk Manager.
    Evaluates sizing and volatility stops. Returns (Is_Approved, Position_Size).
    """
    stop_loss = entry_price - (atr_multiplier * atr)
    risk_per_share = entry_price - stop_loss
    
    if risk_per_share <= 0:
        return False, 0.0 # Veto: Invalid Volatility Profile
        
    capital_at_risk = account_capital * max_risk_pct
    position_size = capital_at_risk / risk_per_share
    
    max_allowable_shares = account_capital / entry_price
    position_size = min(position_size, max_allowable_shares)
    
    # FIX: Hard force floor rounding to avoid Alpaca fractional share rejections entirely
    position_size = math.floor(position_size)
    
    if position_size < 1.0: 
        return False, 0.0 # Veto: Account too small for safe risk profile on this asset
        
    return True, float(position_size)

# ==============================================================================
# 3. LIVE SANDBOX EXECUTION CYCLE
# ==============================================================================
class LiveTradingSandbox:
    def __init__(self, is_paper: bool = True):
        self.is_paper = is_paper
        
        self.api_key = os.environ.get('ALPACA_PAPER_API_KEY') if is_paper else os.environ.get('ALPACA_LIVE_API_KEY')
        self.secret_key = os.environ.get('ALPACA_PAPER_SECRET_KEY') if is_paper else os.environ.get('ALPACA_LIVE_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("CRITICAL: Alpaca API credentials missing from environment.")
            
        self.client = TradingClient(self.api_key, self.secret_key, paper=self.is_paper)
        
        account = self.client.get_account()
        self.buying_power = float(account.buying_power)
        logger.info(f"Connected to Alpaca. Mode: {'PAPER' if is_paper else 'LIVE'}. Buying Power: ${self.buying_power:,.2f}")

    def load_champion_model(self, sector_name: str) -> tuple:
        model_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_champion.json")
        features_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_champion_features.json")
        
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            return None, None
            
        booster = xgb.Booster()
        booster.load_model(model_path)
        
        with open(features_path, "r") as f:
            features = json.load(f)
            
        return booster, features

    def execute_live_cycle(self, live_market_df: pd.DataFrame) -> None:
        logger.info(f"Initiating Live Market Cycle (Fusion: {config.FUSION_ENABLED}, Risk Manager: {config.RISK_MANAGER_ENABLED})")
        
        unique_sectors = live_market_df['sector'].unique()
        ledger_entries = []
        
        for sector in unique_sectors:
            booster, features = self.load_champion_model(str(sector))
            if not booster:
                continue
                
            sector_data = live_market_df[live_market_df['sector'] == sector].copy()
            
            for index, row in sector_data.iterrows():
                ticker = row['ticker']
                current_price = row['close']
                current_atr = row['atr']
                
                if config.FUSION_ENABLED:
                    sentiment = fetch_live_sentiment(ticker)
                    sector_data.at[index, 'sentiment_score'] = sentiment
                
                dmatrix = xgb.DMatrix(sector_data.loc[[index]][features])
                probability_array = booster.predict(dmatrix)
                
                # FIX: Extract scalar value from the XGBoost prediction array
                probability = probability_array.item() if isinstance(probability_array, np.ndarray) else probability_array
                
                signal = "BUY" if probability > config.CONFIDENCE_THRESHOLD else "HOLD"
                veto_reason = "N/A"
                position_size = 0.0
                executed = False
                
                if signal == "BUY" and config.RISK_MANAGER_ENABLED:
                    is_approved, position_size = evaluate_risk_veto_gates(
                        current_price, current_atr, config.ATR_STOP_MULTIPLIER, 
                        self.buying_power, config.MAX_RISK_PER_TRADE
                    )
                    
                    if not is_approved:
                        signal = "VETO"
                        veto_reason = "Risk/Reward limits exceeded or Insufficient Capital."
                    else:
                        executed = self.route_alpaca_order(ticker, position_size)
                        
                elif signal == "BUY" and not config.RISK_MANAGER_ENABLED:
                    # Naked execution defaults to 1 whole share
                    position_size = 1.0 
                    executed = self.route_alpaca_order(ticker, position_size)

                ledger_entries.append({
                    "timestamp": pd.Timestamp.now(),
                    "ticker": ticker,
                    "sector": sector,
                    "probability": float(probability) if isinstance(probability, np.ndarray) else float(probability),
                    "sentiment": sector_data.at[index, 'sentiment_score'] if config.FUSION_ENABLED else 0.0,
                    "signal": signal,
                    "veto_reason": veto_reason,
                    "position_size": position_size,
                    "executed": executed
                })
                
                # FIX 1: Flush DMatrix from GPU VRAM after every inference
                del dmatrix

            # FIX 2: Flush XGBoost Booster graph from GPU VRAM after sector completes
            del booster, features
            gc.collect()
                
        if ledger_entries:
            os.makedirs(config.LIVE_LOG_DIR, exist_ok=True)
            df_ledger = pd.DataFrame(ledger_entries)
            table = pa.Table.from_pandas(df_ledger)
            pq.write_to_dataset(table, root_path=config.LIVE_LOG_DIR, partition_cols=['sector'])

    def route_alpaca_order(self, ticker: str, quantity: float) -> bool:
        try:
            order_request = MarketOrderRequest(
                symbol=ticker,
                # Enforce integer typing for Alpaca API strictly to match the floor logic
                qty=int(quantity), 
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            self.client.submit_order(order_data=order_request)
            logger.info(f"EXECUTED: {ticker} | Qty: {int(quantity)}")
            return True
        except Exception as e:
            logger.error(f"Alpaca Routing Failed for {ticker}: {e}")
            return False