import os
import shutil
import dask.dataframe as dd
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from numba import njit
from typing import Any
import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
# FIXED: Corrected dunder method from **name** to __name__
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

def populate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populates the mathematical sensor suite using vectorized operations.
    Enforces lower-case column names to protect against key errors downstream.
    """
    df.columns = [c.lower() for c in df.columns]
    
    # Trend & Momentum
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    
    # MACD Histogram extraction
    macd = ta.macd(df['close'])
    if macd is not None and not macd.empty:
        df['macd_hist'] = macd[macd.columns[1]] 
    else:
        df['macd_hist'] = np.nan
        
    # Volatility & Volume
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['obv'] = ta.obv(df['close'], df['volume'])
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if stoch is not None and not stoch.empty:
        df['stoch_k'] = stoch[stoch.columns]
    else:
        df['stoch_k'] = np.nan
        
    return df


# ==============================================================================
# JIT COMPILED LABEL GENERATION (Eliminates strict Python loop bottlenecks)
# ==============================================================================
@njit
def _compute_rmultiple_labels(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, atrs: np.ndarray, rr_ratio: float, max_hold_days: int) -> np.ndarray:
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - max_hold_days):
        if np.isnan(atrs[i]):
            continue
            
        stop_loss = closes[i] - atrs[i]
        tp = closes[i] + (atrs[i] * rr_ratio)
        
        for j in range(i + 1, i + max_hold_days):
            if lows[j] <= stop_loss:
                break
            if highs[j] >= tp:
                labels[i] = 1
                break
    return labels

def add_rmultiple_labels(df: pd.DataFrame, rr_ratio: float = config.RR_RATIO, max_hold_days: int = config.MAX_HOLD_DAYS) -> pd.DataFrame:
    """
    Simulates a standard directional trade for every row.
    Assigns a '1' if the 2:1 risk/reward target is hit before the 1R ATR stop-loss.
    """
    df['target_label'] = _compute_rmultiple_labels(
        df['close'].values, df['high'].values, df['low'].values, df['atr'].values, 
        rr_ratio, max_hold_days
    )
    return df

@njit
def _compute_options_labels(closes: np.ndarray, atrs: np.ndarray, dte: int, target_premium_gain: float) -> np.ndarray:
    n = len(closes)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - dte):
        if np.isnan(atrs[i]):
            continue
            
        entry_price = closes[i]
        target_price = entry_price + (atrs[i] * 2) 
        stop_loss = entry_price - atrs[i]

        for j in range(i + 1, i + dte):
            if closes[j] <= stop_loss:
                break
            if closes[j] >= target_price:
                labels[i] = 1
                break
    return labels

def add_options_labels(df: pd.DataFrame, dte: int = config.OPTIONS_DTE, target_premium_gain: float = config.TARGET_PREMIUM_GAIN) -> pd.DataFrame:
    """
    Simulates a Delta-adjusted option contract return.
    Accounts for time horizons across the specific Days to Expiration (DTE).
    """
    # FIXED: Removed the stranded docstring and dead code variables that were 
    # previously stuck underneath the return statement.
    df['option_target_label'] = _compute_options_labels(
        df['close'].values, df['atr'].values, dte, target_premium_gain
    )
    return df


# ==============================================================================
# DASK WORKER EXECUTION MAPPING
# ==============================================================================
def compute_partition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies massive mechanical sensors to a localized data chunk. 
    Executed lazily inside Dask worker allocation threads.
    """
    if df.empty or len(df) < 252:
        return pd.DataFrame(columns=df.columns)
        
    df = populate_indicators(df)
    
    # Branch label processing based on Global Config RUN_MODE
    if config.RUN_MODE == "STANDARD":
        df = add_rmultiple_labels(df)
    elif config.RUN_MODE == "OPTIONS":
        df = add_options_labels(df)
        
    # Drop rows containing NaNs introduced by indicator lookback windows
    df = df.dropna()
    return df

def compile_features_from_raw() -> None:
    """
    Orchestrates the offline Dask-powered transformation pipeline.
    Reads partitioned raw data out-of-core, computes math, and flushes to the processed vault.
    """
    if not os.path.exists(config.RAW_VAULT_DIR):
        logger.error("Raw storage vault missing. Run ingestion sequence first.")
        return
        
    logger.info("Initiating offline Dask-powered feature compilation...")
    reset_processed_vault()

    # Load raw data via PyArrow backed Dask utilizing config settings 
    ddf = dd.read_parquet(config.RAW_VAULT_DIR, **config.DASK_READ_KWARGS)

    # Apply the processing map to each partition independently
    ddf_processed = ddf.map_partitions(compute_partition_features)

    # Dump the compiled matrices directly into a Hive-partitioned directory scheme
    ddf_processed.to_parquet(
        config.PROCESSED_VAULT_DIR,
        engine="pyarrow",
        partition_on=['sector'],
        write_metadata_file=False
    )
    logger.info(f"Feature matrices successfully compiled and safely exported to {config.PROCESSED_VAULT_DIR}.")