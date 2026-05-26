import os
from datetime import datetime, timedelta
import pyarrow as pa
import pandas as pd

# ==============================================================================
# 1. GLOBAL SYSTEM RUN MODE CONFIGURATION
# ==============================================================================
# Toggles operational mode: Use "STANDARD" (Equity) or "OPTIONS" (Delta Simulation)
RUN_MODE = "STANDARD"

# 3-Year Historical Lookback Window
START_DATE = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# ==============================================================================
# 2. DATA VAULT & LOGGING DIRECTORY PATHS
# ==============================================================================
RAW_VAULT_DIR = "./market_vault_raw"
PROCESSED_VAULT_DIR = f"./market_vault_processed_{RUN_MODE.lower()}"
RESULTS_FILE = "tournament_final_results.parquet"
LIVE_LOG_DIR = "./live_execution_ledger"
PROD_MODELS_DIR = "./production_models"

# System Logging
LOG_DIR = "./system_logs"
os.makedirs(LOG_DIR, exist_ok=True)
SYSTEM_LOG_FILE = os.path.join(LOG_DIR, "quantum_sentinel.log")

# ==============================================================================
# 3. HYPERPARAMETERS & MAGIC NUMBERS
# ==============================================================================
# Centralized thresholds to prevent hardcoded values inside execution modules
CONFIDENCE_THRESHOLD = 0.65
MAX_HOLD_DAYS = 20
RR_RATIO = 2.0
OPTIONS_DTE = 21
TARGET_PREMIUM_GAIN = 0.50

# ==============================================================================
# 4. MACHINE LEARNING METADATA BOUNDARIES
# ==============================================================================
# Columns exclusively used for tracking and execution that must be dropped
# before feeding the feature matrix into the XGBoost algorithm.
METADATA_COLS = [
    'open', 'high', 'low', 'close', 'adj close', 'volume', 
    'target_label', 'option_target_label', 'ticker', 'sector', 'date'
]

# ==============================================================================
# 5. OUT-OF-CORE DASK & PARQUET CONFIGURATION
# ==============================================================================
# Targeting an in-memory size of 100-300 MiB per file partition balances 
# worker memory usage against Dask scheduling overhead.
PARQUET_BLOCKSIZE = "256MiB"

# Dask configuration dictionary for memory-mapped Parquet loading
DASK_READ_KWARGS = {
    "engine": "pyarrow",
    "blocksize": PARQUET_BLOCKSIZE,
    "split_row_groups": "adaptive",
    "dtype_backend": "numpy_nullable"
}

# ==============================================================================
# 6. EXPLICIT PYARROW SCHEMA MAPPING
# ==============================================================================
# Explicit schema uses Dictionary encoding for memory efficiency on categorical
# columns (like sectors) and locks numerical data types to prevent upcasting.
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

NULLABLE_TYPES_MAPPER = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.float32(): pd.Float32Dtype(),
    pa.float64(): pd.Float64Dtype(),
    pa.string(): pd.StringDtype()
}