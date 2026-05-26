import os
from datetime import datetime, timedelta
import pyarrow as pa
import pandas as pd
import argparse
import logging
import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.SYSTEM_LOG_FILE),  # Saves to disk
        logging.StreamHandler()                       # Prints to console
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules AFTER logging is configured
import data_ingestion
import feature_compiler
import tournament

def main():
    logger.info(f"=== QUANTUM SENTINEL ORCHESTRATOR [{config.RUN_MODE} MODE] ===")
    
    # ... [rest of the arg parsing logic remains the same]

    # Replace print statements with logger.info
    if args.refresh_raw:
        logger.info("[COMMAND] --refresh-raw detected. Synchronizing raw market data...")
        # ...

# ==============================================================================
# 1. GLOBAL SYSTEM RUN MODE CONFIGURATION
# ==============================================================================
# Toggles operational mode: Use "STANDARD" (Equity) or "OPTIONS" (Delta Simulation)
RUN_MODE = "STANDARD"

# 3-Year Historical Lookback Window
START_DATE = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# ==============================================================================
# 2. DATA VAULT DIRECTORY PATHS
# ==============================================================================
RAW_VAULT_DIR = "./market_vault_raw"
PROCESSED_VAULT_DIR = f"./market_vault_processed_{RUN_MODE.lower()}"
RESULTS_FILE = "tournament_final_results.parquet"
LIVE_LOG_DIR = "./live_execution_ledger"
PROD_MODELS_DIR = "./production_models"

# ==============================================================================
# 3. MACHINE LEARNING METADATA BOUNDARIES
# ==============================================================================
# Columns exclusively used for tracking and execution that must be dropped
# before feeding the feature matrix into the XGBoost algorithm.
METADATA_COLS = [
    'open', 'high', 'low', 'close', 'adj close', 'volume', 
    'target_label', 'option_target_label', 'ticker', 'sector', 'date'
]

# ==============================================================================
# 4. OUT-OF-CORE DASK & PARQUET CONFIGURATION
# ==============================================================================
# Targeting an in-memory size of 100-300 MiB per file partition balances 
# worker memory usage against Dask scheduling overhead [1].
PARQUET_BLOCKSIZE = "256MiB"

# Dask configuration dictionary for memory-mapped Parquet loading
DASK_READ_KWARGS = {
    "engine": "pyarrow",
    "blocksize": PARQUET_BLOCKSIZE,
    # Setting adaptive row-group splitting ensures Dask strictly respects the 
    # blocksize limit to prevent memory exhaustion [2].
    "split_row_groups": "adaptive",
    # Setting this to 'numpy_nullable' explicitly forces Pandas to use its nullable 
    # extension dtypes natively via the backend [3].
    "dtype_backend": "numpy_nullable"
}

# ==============================================================================
# 5. EXPLICIT PYARROW SCHEMA MAPPING
# ==============================================================================
# Arrow natively supports missing data, but standard Pandas will destructively upcast 
# integers to floats when NaN values are present [4].
# This explicit schema uses Dictionary encoding for memory efficiency on categorical
# columns (like sectors) and locks numerical data types.
MARKET_SCHEMA = pa.schema([
    pa.field('date', pa.timestamp('ns')),
    pa.field('ticker', pa.string()),
    pa.field('sector', pa.dictionary(pa.int32(), pa.string())),
    pa.field('open', pa.float32()),
    pa.field('high', pa.float32()),
    pa.field('low', pa.float32()),
    pa.field('close', pa.float32()),
    pa.field('adj close', pa.float32()),
    pa.field('volume', pa.int64()),
    pa.field('target_label', pa.int8())
])

# If Dask is not used, this fallback types_mapper can be passed directly to 
# `pyarrow.Table.to_pandas()` to prevent upcasting by instructing Arrow to create 
# a Pandas DataFrame mapped directly to nullable extension dtypes [5].
NULLABLE_TYPES_MAPPER = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.float32(): pd.Float32Dtype(),
    pa.float64(): pd.Float64Dtype(),
    pa.string(): pd.StringDtype()
}

# Add this to the bottom of config.py
LOG_DIR = "./system_logs"
os.makedirs(LOG_DIR, exist_ok=True)
SYSTEM_LOG_FILE = os.path.join(LOG_DIR, "quantum_sentinel.log")