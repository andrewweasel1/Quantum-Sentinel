import os
import shutil
import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
import dask.dataframe as dd
import pandas_ta as ta
from numba import njit

import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
logger = logging.getLogger(__name__)

def processed_vault_is_populated() -> bool:
    """Checks if the processed vault directory exists and contains data."""
    if not os.path.exists(config.PROCESSED_VAULT_DIR):
        return False
    subdirs = [d for d in os.listdir(config.PROCESSED_VAULT_DIR) if os.path.isdir(os.path.join(config.PROCESSED_VAULT_DIR, d))]
    return len(subdirs) > 0

def reset_processed_vault() -> None:
    """Clears existing processed data to prevent duplicate or corrupted legacy files."""
    if os.path.exists(config.PROCESSED_VAULT_DIR):
        shutil.rmtree(config.PROCESSED_VAULT_DIR)
    os.makedirs(config.PROCESSED_VAULT_DIR, exist_ok=True)

# ==============================================================================
# 1. JIT-COMPILED MICROSTRUCTURE & TAIL-RISK SENSORS
# ==============================================================================

@njit
def compute_roll_spread(closes: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Estimates the effective bid-ask spread using Roll's Serial Covariance model.
    A negative serial covariance of price changes implies bid-ask bounce friction.
    """
    n = len(closes)
    spreads = np.zeros(n, dtype=np.float64)
    diffs = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        diffs[i] = closes[i] - closes[i-1]

    for i in range(window, n):
        window_diffs = diffs[i-window+1:i+1]
        x = window_diffs[1:]
        y = window_diffs[:-1]
        
        # Calculate Numba-safe covariance
        if len(x) > 1:
            cov = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(x) - 1)
            if cov < 0:
                spreads[i] = 2.0 * np.sqrt(-cov)
            else:
                spreads[i] = 0.0001 # Minimum noise floor
    return spreads

@njit
def compute_crash_risk(returns: np.ndarray, window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Negative Conditional Skewness (NCSKEW) and Down-to-Up Volatility (DUVOL).
    These metrics allow the model to detect impending asymmetrical crashes.
    """
    n = len(returns)
    ncskew = np.zeros(n, dtype=np.float64)
    duvol = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        window_ret = returns[i-window:i]
        mean_ret = np.mean(window_ret)
        std_ret = np.std(window_ret)
        
        # 1. NCSKEW
        if std_ret > 0:
            skew = np.sum(((window_ret - mean_ret) / std_ret)**3) / window
            ncskew[i] = -skew  
            
        # 2. DUVOL
        down_returns = window_ret[window_ret < mean_ret]
        up_returns = window_ret[window_ret > mean_ret]
        
        if len(down_returns) > 2 and len(up_returns) > 2:
            var_down = np.var(down_returns)
            var_up = np.var(up_returns)
            if var_up > 0 and var_down > 0:
                duvol[i] = np.log(var_down / var_up)
                
    return ncskew, duvol

@njit
def compute_amihud_illiquidity(returns: np.ndarray, closes: np.ndarray, volumes: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Amihud Illiquidity Measure: Captures the daily price response associated with one dollar of trading volume.
    """
    n = len(returns)
    amihud = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        window_ret = np.abs(returns[i-window:i])
        window_dollar_vol = closes[i-window:i] * volumes[i-window:i]
        
        # Prevent division by zero
        valid_idx = window_dollar_vol > 0
        if np.sum(valid_idx) > 0:
            amihud[i] = np.mean(window_ret[valid_idx] / window_dollar_vol[valid_idx])
            
    return amihud

# ==============================================================================
# 2. FRICTION-ADJUSTED TRIPLE BARRIER LABEL GENERATION 
# ==============================================================================

@njit
def _compute_friction_adjusted_labels(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, 
                                      atrs: np.ndarray, spreads: np.ndarray, current_volumes: np.ndarray, 
                                      advs: np.ndarray, rr_ratio: float, max_hold_days: int) -> np.ndarray:
    """
    Triple-Barrier Method with Dynamic Friction.
    A trade is only labeled as successful (1) if it hits the profit target AFTER 
    paying the Corwin-Schultz/Roll bid-ask spread and the square-root market impact slippage.
    """
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    
    for i in range(n - max_hold_days):
        if np.isnan(atrs[i]) or np.isnan(spreads[i]) or advs[i] == 0:
            continue
            
        # 1. Estimate Entry Friction
        vol_ratio = current_volumes[i] / advs[i]
        entry_slippage = closes[i] * 0.001 * np.sqrt(vol_ratio) 
        entry_friction = (spreads[i] / 2.0) + entry_slippage
        
        true_entry_price = closes[i] + entry_friction
        stop_loss = true_entry_price - atrs[i]
        take_profit = true_entry_price + (atrs[i] * rr_ratio)
        
        # 2. Path Simulation (Triple Barrier)
        for j in range(i + 1, i + max_hold_days):
            # Lower barrier hit -> Stop Loss Executed
            if lows[j] <= stop_loss:
                break
                
            # Upper barrier hit -> Must clear the exit friction to be truly profitable
            exit_vol_ratio = current_volumes[j] / advs[j]
            exit_friction = (spreads[j] / 2.0) + (closes[j] * 0.001 * np.sqrt(exit_vol_ratio))
            
            if (highs[j] - exit_friction) >= take_profit:
                labels[i] = 1
                break
                
    return labels

def apply_institutional_labels(df: pd.DataFrame, rr_ratio: float = config.RR_RATIO, max_hold_days: int = config.MAX_HOLD_DAYS) -> pd.DataFrame:
    df['target_label'] = _compute_friction_adjusted_labels(
        df['close'].values, df['high'].values, df['low'].values, df['atr'].values, 
        df['roll_spread'].values, df['volume'].values, df['adv_20'].values, 
        rr_ratio, max_hold_days
    )
    return df

# ==============================================================================
# 3. DASK WORKER EXECUTION MAPPING
# ==============================================================================

def compute_partition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the massive institutional sensor suite to a localized data chunk.
    Executed lazily inside Dask worker allocation threads.
    """
    if df.empty or len(df) < 252:
        return pd.DataFrame(columns=df.columns)
        
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Base Analytics
    df['returns'] = df['close'].pct_change().fillna(0)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['adv_20'] = df['volume'].rolling(window=20).mean()
    
    # 2. Market Microstructure & Liquidity
    df['roll_spread'] = compute_roll_spread(df['close'].values, window=20)
    df['amihud_illiq'] = compute_amihud_illiquidity(df['returns'].values, df['close'].values, df['volume'].values, window=20)
    
    # 3. Crash Risk & Asymmetry
    ncskew, duvol = compute_crash_risk(df['returns'].values, window=60)
    df['ncskew'] = ncskew
    df['duvol'] = duvol
    
    # 4. Friction-Adjusted Labeling
    if config.RUN_MODE == "STANDARD":
        df = apply_institutional_labels(df)
        
    # Drop NaNs to protect XGBoost DMatrix creation
    df = df.dropna()
    return df

def compile_features_from_raw() -> None:
    """
    Orchestrates the offline Dask-powered transformation pipeline.
    """
    if not os.path.exists(config.RAW_VAULT_DIR):
        logger.error("Raw storage vault missing. Run ingestion sequence first.")
        return
        
    logger.info("Initiating offline Dask-powered feature compilation using institutional metrics...")
    reset_processed_vault()

    ddf = dd.read_parquet(config.RAW_VAULT_DIR, **config.DASK_READ_KWARGS)

    # Apply the advanced processing map to each partition
    ddf_processed = ddf.map_partitions(compute_partition_features)

    ddf_processed.to_parquet(
        config.PROCESSED_VAULT_DIR,
        engine="pyarrow",
        partition_on=['sector'],
        write_metadata_file=False
    )
    logger.info(f"Friction-adjusted matrices safely exported to {config.PROCESSED_VAULT_DIR}.")