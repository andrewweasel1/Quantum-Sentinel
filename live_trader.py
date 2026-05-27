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
from alpaca.trading.requests import LimitOrderRequest
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
        # Initialize Alpaca client for secure execution [1]
        self.client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=is_paper)
        logger.info(f"LiveTradingSandbox Initialized (Paper: {is_paper}).")

    def sync_portfolio_state(self) -> dict:
        """
        Polls the broker to map current inventory, preventing recursive over-allocation.
        """
        try:
            positions = self.client.get_all_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            logger.error(f"Failed to synchronize portfolio state: {e}")
            return {}

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

    def execute_live_cycle(self, current_data: pd.DataFrame, booster: xgb.Booster) -> None:
        """
        The terminal execution loop. Processes the feature manifold, queries the LLM,
        validates risk, and dispatches dynamic limit orders.
        """
        logger.info("Initiating Live Execution Cycle...")
        current_inventory = self.sync_portfolio_state()
        account = self.client.get_account()
        available_capital = float(account.buying_power)

        for index, row in current_data.iterrows():
            ticker = row['ticker']
            
            # 1. XGBoost & LLM Inference Handoff
            features = [c for c in current_data.columns if c not in config.METADATA_COLS]
            dmatrix = xgb.DMatrix(current_data.loc[[index]][features])
            probability = booster.predict(dmatrix)
            
            if probability > config.CONFIDENCE_THRESHOLD:
                # 2. Risk Management Gate
                is_approved, target_size = evaluate_risk_veto_gates(
                    entry_price=row['close'], 
                    atr=row['atr'], 
                    atr_multiplier=config.ATR_STOP_MULTIPLIER, 
                    account_capital=available_capital, 
                    max_risk_pct=config.MAX_RISK_PER_TRADE
                )
                
                if is_approved:
                    # 3. Portfolio Delta Calculation
                    current_qty = current_inventory.get(ticker, 0.0)
                    delta_qty = target_size - current_qty
                    
                    if delta_qty > 0:
                        # 4. Dynamic Limit Order Routing
                        # Protects against adverse execution using local microstructure volatility
                        limit_price = row['close'] + (0.1 * row['atr'])
                        
                        order_data = LimitOrderRequest(
                            symbol=ticker,
                            qty=delta_qty,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                            limit_price=round(limit_price, 2)
                        )
                        
                        try:
                            order = self.client.submit_order(order_data)
                            logger.info(f"[{ticker}] ORDER DISPATCHED: {delta_qty} shares @ {limit_price:.2f} Limit.")
                            
                            # Log to PyArrow Ledger
                            self._log_to_ledger(ticker, "BUY", delta_qty, limit_price)
                        except Exception as e:
                            logger.error(f"[{ticker}] Execution rejected by broker: {e}")
                    else:
                        logger.info(f"[{ticker}] Target size met. Current inventory sufficient.")
                else:
                    logger.warning(f"[{ticker}] VETO: Risk parameters exceeded.")