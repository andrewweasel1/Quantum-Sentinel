import os
import re
import math
import json
import shutil
import asyncio
import aiohttp
import nest_asyncio
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
import pandas_ta as ta
from numba import cuda

import config

# FIX: Allow asyncio.run() to operate securely inside Dask's existing worker event loops
nest_asyncio.apply()

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. ASYNCHRONOUS ENTITY ANONYMIZATION & LLM INFERENCE (THE SENSOR)
# ==============================================================================
async def fetch_sentiment_async(semaphore: asyncio.Semaphore, session: aiohttp.ClientSession, headline: str, ticker: str) -> float:
    """
    Asynchronously queries the local LLM.
    Crucially utilizes Entity Anonymization to scrub the ticker from the text, 
    preventing the LLM from relying on memorized historical look-ahead bias.
    """
    if pd.isna(headline) or not str(headline).strip():
        return 0.0

    anonymized_headline = re.sub(rf'\b{ticker}\b', 'the company', str(headline), flags=re.IGNORECASE)

    payload = {
        "model": config.LLM_MODEL_NAME,
        "prompt": f"{config.LLM_SYSTEM_PROMPT}\n\nHeadline: {anonymized_headline}",
        "format": "json",
        "stream": False
    }

    # The semaphore acts as a traffic controller to prevent local GPU queue overflow
    async with semaphore:
        try:
            # Timeout increased slightly to accommodate local queuing
            async with session.post(config.OLLAMA_ENDPOINT, json=payload, timeout=10.0) as response:
                if response.status == 200:
                    result = await response.json()
                    data = json.loads(result.get("response", "{}"))
                    return float(data.get("sentiment_score", 0.0))
        except Exception:
            pass
            
    return 0.0

async def process_llm_batch_async(df: pd.DataFrame) -> list:
    """
    Batches HTTP requests to the local LLM concurrently across the pandas partition.
    """
    # Limit concurrent requests to 20 to protect the local Ollama server from crashing
    semaphore = asyncio.Semaphore(20)
    
    # TCPConnector limits the total active connections pooling
    connector = aiohttp.TCPConnector(limit=20, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_sentiment_async(semaphore, session, str(row['raw_news_headline']), str(row['ticker']))
            for _, row in df.iterrows()
        ]
        return await asyncio.gather(*tasks)

# ==============================================================================
# 2. CUDA JIT KERNELS (VRAM FAST-MATH EXECUTION)
# ==============================================================================
# ... [Keep all existing @cuda.jit math kernels unchanged] ...

# ==============================================================================
# 3. DASK WORKER EXECUTION MAPPING
# ==============================================================================

def compute_partition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies massive mechanical sensors to a localized data chunk. 
    Routes text to the CPU-bound LLM asynchronously before mapping structural math to the GPU.
    """
    if df.empty or len(df) < 252:
        return pd.DataFrame(columns=df.columns)
        
    df.columns = [c.lower() for c in df.columns]
    
    # -------------------------------------------------------------------------
    # MULTI-AGENT HANDOFF: CPU-Bound Asynchronous LLM Sentiment Batching
    # -------------------------------------------------------------------------
    if config.FUSION_ENABLED and 'raw_news_headline' in df.columns:
        # FIX: Replaced synchronous pandas `.apply()` with an asynchronous event loop
        # that fires multiple local LLM queries concurrently without blocking
        sentiment_results = asyncio.run(process_llm_batch_async(df))
        df['sentiment_score'] = np.array(sentiment_results, dtype=np.float32)

    # Base CPU Analytics
    df['returns'] = df['close'].pct_change().fillna(0)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['adv_20'] = df['volume'].rolling(window=20).mean()
    
   # FIX: Purge NaNs prior to VRAM transfer to prevent IEEE-754 CUDA array poisoning
    df = df.dropna()
    
    # Stage continuous NumPy arrays for GPU transfer
    # FIX: Safely serialize PyArrow Extension Arrays into strict contiguous C-arrays for Numba VRAM 
    closes = np.ascontiguousarray(df['close'].to_numpy(dtype=np.float64, na_value=np.nan))
    highs = np.ascontiguousarray(df['high'].to_numpy(dtype=np.float64, na_value=np.nan))
    lows = np.ascontiguousarray(df['low'].to_numpy(dtype=np.float64, na_value=np.nan))
    volumes = np.ascontiguousarray(df['volume'].to_numpy(dtype=np.float64, na_value=np.nan))
    returns = np.ascontiguousarray(df['returns'].to_numpy(dtype=np.float64, na_value=np.nan))
    atrs = np.ascontiguousarray(df['atr'].fillna(0).to_numpy(dtype=np.float64, na_value=np.nan))
    advs = np.ascontiguousarray(df['adv_20'].fillna(0).to_numpy(dtype=np.float64, na_value=np.nan))
    
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