import os
import re
import math
import json
import shutil
import requests
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
import pandas_ta as ta
from numba import cuda

import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
logger = logging.getLogger(__name__)

def processed_vault_is_populated() -> bool:
    if not os.path.exists(config.PROCESSED_VAULT_DIR):
        return False
    subdirs = [d for d in os.listdir(config.PROCESSED_VAULT_DIR) if os.path.isdir(os.path.join(config.PROCESSED_VAULT_DIR, d))]
    return len(subdirs) > 0

def reset_processed_vault() -> None:
    if os.path.exists(config.PROCESSED_VAULT_DIR):
        shutil.rmtree(config.PROCESSED_VAULT_DIR)
    os.makedirs(config.PROCESSED_VAULT_DIR, exist_ok=True)

# ==============================================================================
# 1. ENTITY ANONYMIZATION & LLM INFERENCE (THE SENSOR)
# ==============================================================================
def extract_llm_sentiment(headline: str, ticker: str) -> float:
    """
    Passes news text to the local LLM. 
    Crucially utilizes Entity Anonymization to scrub the ticker from the text, 
    preventing the LLM from relying on memorized historical look-ahead bias.
    """
    if pd.isna(headline) or not str(headline).strip():
        return 0.0

    # Entity Anonymization: Mask the specific company identifier
    anonymized_headline = re.sub(rf'\b{ticker}\b', 'the company', str(headline), flags=re.IGNORECASE)

    payload = {
        "model": config.LLM_MODEL_NAME,
        "prompt": f"{config.LLM_SYSTEM_PROMPT}\n\nHeadline: {anonymized_headline}",
        "format": "json",
        "stream": False
    }

    try:
        # Route to local Ollama server (running on CPU/System RAM)
        response = requests.post(config.OLLAMA_ENDPOINT, json=payload, timeout=2.0)
        if response.status_code == 200:
            result = response.json()
            data = json.loads(result.get("response", "{}"))
            return float(data.get("sentiment_score", 0.0))
    except Exception:
        # Failsafe: Default to neutral sentiment if the LLM server times out or hallucinate formatting
        pass

    return 0.0

# ==============================================================================
# 2. CUDA JIT KERNELS (VRAM FAST-MATH EXECUTION)
# ==============================================================================

@cuda.jit(fastmath=True)
def compute_roll_spread_cuda(closes, spreads, window):
    i = cuda.grid(1)
    n = len(closes)
    
    if i >= window and i < n:
        mean_x = 0.0
        mean_y = 0.0
        
        for j in range(i - window + 1, i):
            mean_x += (closes[j+1] - closes[j])
            mean_y += (closes[j] - closes[j-1])
        
        mean_x /= (window - 1)
        mean_y /= (window - 1)

        cov = 0.0
        for j in range(i - window + 1, i):
            x = (closes[j+1] - closes[j])
            y = (closes[j] - closes[j-1])
            cov += (x - mean_x) * (y - mean_y)
            
        cov /= (window - 2)

        if cov < 0:
            spreads[i] = 2.0 * math.sqrt(-cov)
        else:
            spreads[i] = 0.0001

@cuda.jit(fastmath=True)
def compute_crash_risk_cuda(returns, ncskew, duvol, window):
    i = cuda.grid(1)
    n = len(returns)
    
    if i >= window and i < n:
        mean_ret = 0.0
        for j in range(i - window, i):
            mean_ret += returns[j]
        mean_ret /= window
        
        var_ret = 0.0
        for j in range(i - window, i):
            var_ret += (returns[j] - mean_ret)**2
        std_ret = math.sqrt(var_ret / window)
        
        if std_ret > 0:
            skew = 0.0
            for j in range(i - window, i):
                skew += ((returns[j] - mean_ret) / std_ret)**3
            ncskew[i] = -(skew / window)
            
        down_sum, up_sum = 0.0, 0.0
        down_count, up_count = 0, 0
        
        for j in range(i - window, i):
            if returns[j] < mean_ret:
                down_sum += (returns[j] - mean_ret)**2
                down_count += 1
            elif returns[j] > mean_ret:
                up_sum += (returns[j] - mean_ret)**2
                up_count += 1
                
        if down_count > 2 and up_count > 2:
            var_down = down_sum / (down_count - 1)
            var_up = up_sum / (up_count - 1)
            if var_up > 0 and var_down > 0:
                duvol[i] = math.log(var_down / var_up)

@cuda.jit(fastmath=True)
def compute_amihud_illiquidity_cuda(returns, closes, volumes, amihud, window):
    i = cuda.grid(1)
    n = len(returns)
    
    if i >= window and i < n:
        amihud_sum = 0.0
        valid_days = 0
        
        for j in range(i - window, i):
            dollar_vol = closes[j] * volumes[j]
            if dollar_vol > 0:
                amihud_sum += math.fabs(returns[j]) / dollar_vol
                valid_days += 1
                
        if valid_days > 0:
            amihud[i] = amihud_sum / valid_days

@cuda.jit(fastmath=True)
def compute_friction_labels_cuda(closes, highs, lows, atrs, spreads, current_volumes, advs, labels, rr_ratio, max_hold_days):
    i = cuda.grid(1)
    n = len(closes)
    
    if i < n - max_hold_days:
        if not (math.isnan(atrs[i]) or math.isnan(spreads[i]) or advs[i] == 0):
            vol_ratio = current_volumes[i] / advs[i]
            entry_slippage = closes[i] * 0.001 * math.sqrt(vol_ratio)
            entry_friction = (spreads[i] / 2.0) + entry_slippage
            
            true_entry_price = closes[i] + entry_friction
            stop_loss = true_entry_price - atrs[i]
            take_profit = true_entry_price + (atrs[i] * rr_ratio)
            
            for j in range(i + 1, i + max_hold_days):
                if lows[j] <= stop_loss:
                    break
                    
                exit_vol_ratio = current_volumes[j] / advs[j]
                exit_friction = (spreads[j] / 2.0) + (closes[j] * 0.001 * math.sqrt(exit_vol_ratio))
                
                if (highs[j] - exit_friction) >= take_profit:
                    labels[i] = 1
                    break

@cuda.jit(fastmath=True)
def compute_options_labels_cuda(closes, atrs, labels, dte, target_premium_gain):
    i = cuda.grid(1)
    n = len(closes)
    
    if i < n - dte:
        if not math.isnan(atrs[i]) and atrs[i] > 0:
            entry = closes[i]
            target = entry + (atrs[i] * target_premium_gain)
            stop = entry - atrs[i]
            
            for j in range(i + 1, i + dte):
                if closes[j] <= stop:
                    break
                if closes[j] >= target:
                    labels[i] = 1
                    break

# ==============================================================================
# 3. DASK WORKER EXECUTION MAPPING
# ==============================================================================

def compute_partition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies massive mechanical sensors to a localized data chunk. 
    Routes text to the CPU-bound LLM before mapping structural math to the GPU.
    """
    if df.empty or len(df) < 252:
        return pd.DataFrame(columns=df.columns)
        
    df.columns = [c.lower() for c in df.columns]
    
    # -------------------------------------------------------------------------
    # MULTI-AGENT HANDOFF: CPU-Bound LLM Sentiment Scoring
    # -------------------------------------------------------------------------
    if config.FUSION_ENABLED and 'raw_news_headline' in df.columns:
        df['sentiment_score'] = df.apply(
            lambda row: extract_llm_sentiment(str(row['raw_news_headline']), str(row['ticker'])),
            axis=1
        ).astype(np.float32)

    # Base CPU Analytics
    df['returns'] = df['close'].pct_change().fillna(0)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['adv_20'] = df['volume'].rolling(window=20).mean()
    
    # Stage continuous NumPy arrays for GPU transfer
    closes = np.ascontiguousarray(df['close'].values.astype(np.float64))
    highs = np.ascontiguousarray(df['high'].values.astype(np.float64))
    lows = np.ascontiguousarray(df['low'].values.astype(np.float64))
    volumes = np.ascontiguousarray(df['volume'].values.astype(np.float64))
    returns = np.ascontiguousarray(df['returns'].values.astype(np.float64))
    atrs = np.ascontiguousarray(df['atr'].fillna(0).values.astype(np.float64))
    advs = np.ascontiguousarray(df['adv_20'].fillna(0).values.astype(np.float64))
    
    n = len(closes)
    
    spreads = np.zeros(n, dtype=np.float64)
    amihud = np.zeros(n, dtype=np.float64)
    ncskew = np.zeros(n, dtype=np.float64)
    duvol = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int8)

    # PUSH MEMORY TO VRAM
    d_closes = cuda.to_device(closes)
    d_highs = cuda.to_device(highs)
    d_lows = cuda.to_device(lows)
    d_volumes = cuda.to_device(volumes)
    d_returns = cuda.to_device(returns)
    d_atrs = cuda.to_device(atrs)
    d_advs = cuda.to_device(advs)
    
    d_spreads = cuda.to_device(spreads)
    d_amihud = cuda.to_device(amihud)
    d_ncskew = cuda.to_device(ncskew)
    d_duvol = cuda.to_device(duvol)
    d_labels = cuda.to_device(labels)

    # KERNEL THREAD CONFIGURATION
    threads_per_block = 256
    blocks_per_grid = math.ceil(n / threads_per_block)

    # DISPATCH ASYNCHRONOUS KERNELS
    compute_roll_spread_cuda[blocks_per_grid, threads_per_block](d_closes, d_spreads, 20)
    compute_amihud_illiquidity_cuda[blocks_per_grid, threads_per_block](d_returns, d_closes, d_volumes, d_amihud, 20)
    compute_crash_risk_cuda[blocks_per_grid, threads_per_block](d_returns, d_ncskew, d_duvol, 60)
    
    cuda.synchronize()

    if config.RUN_MODE == "STANDARD":
        compute_friction_labels_cuda[blocks_per_grid, threads_per_block](
            d_closes, d_highs, d_lows, d_atrs, d_spreads, d_volumes, d_advs, d_labels, 2.0, 20
        )
        df['target_label'] = d_labels.copy_to_host()
    else:
        compute_options_labels_cuda[blocks_per_grid, threads_per_block](
            d_closes, d_atrs, d_labels, 21, 0.50
        )
        df['option_target_label'] = d_labels.copy_to_host()

    # PULL VRAM DATA BACK TO HOST RAM
    df['roll_spread'] = d_spreads.copy_to_host()
    df['amihud_illiq'] = d_amihud.copy_to_host()
    df['ncskew'] = d_ncskew.copy_to_host()
    df['duvol'] = d_duvol.copy_to_host()

    # If the LLM wasn't triggered, safely drop the raw text column so PyArrow Parquet doesn't bloat
    if 'raw_news_headline' in df.columns:
        df = df.drop(columns=['raw_news_headline'])

    df = df.dropna()
    return df

def compile_features_from_raw() -> None:
    """
    Orchestrates the offline Dask-powered transformation pipeline.
    """
    if not os.path.exists(config.RAW_VAULT_DIR):
        logger.error("Raw storage vault missing. Run ingestion sequence first.")
        return
        
    logger.info(f"Engaging CUDA compilation... (Fusion Mode: {'ON' if config.FUSION_ENABLED else 'OFF'})")
    reset_processed_vault()

    ddf = dd.read_parquet(config.RAW_VAULT_DIR, **config.DASK_READ_KWARGS)
    
    ddf_processed = ddf.map_partitions(compute_partition_features)

    ddf_processed.to_parquet(
        config.PROCESSED_VAULT_DIR,
        engine="pyarrow",
        partition_on=['sector'],
        write_metadata_file=False
    )
    logger.info(f"Data arrays safely exported to {config.PROCESSED_VAULT_DIR}.")