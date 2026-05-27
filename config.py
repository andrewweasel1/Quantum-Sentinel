import os
import pyarrow as pa
from datetime import datetime, timedelta

# ==============================================================================
# 1. GLOBAL SYSTEM RUN MODE & FEATURE TOGGLES
# ==============================================================================
# Toggles operational mode: Use "STANDARD" (Equity) or "OPTIONS" (Delta Simulation)
RUN_MODE = "STANDARD"

# These will be dynamically overridden by main.py argparse at runtime
FUSION_ENABLED = False
RISK_MANAGER_ENABLED = True

# 3-Year Historical Lookback Window
START_DATE = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# ==============================================================================
# 2. LLM & SENTIMENT FUSION PARAMETERS (The Sensor)
# ==============================================================================
# Defaults designed for a local quantized model via Ollama to protect privacy
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
LLM_MODEL_NAME = "llama3:8b"
LLM_TOKEN_LIMIT = 512
LLM_SYSTEM_PROMPT = """You are an objective Quantitative Risk Analyst. 
Read the following anonymized financial news headline. 
Output ONLY a valid JSON dictionary with a single key "sentiment_score" 
and a float value between -1.0 (extreme bearish) and 1.0 (extreme bullish). 
Do not provide any other text or reasoning."""

# ==============================================================================
# 3. RISK MANAGER PARAMETERS (The Shield)
# ==============================================================================
MAX_DAILY_DRAWDOWN = -0.03   # 3% daily account drawdown limit (Veto parameter)
ATR_STOP_MULTIPLIER = 1.5    # Trailing stop distance multiplier
MAX_RISK_PER_TRADE = 0.02    # Risk exactly 2% of account capital per position
MAX_CORRELATION_LIMIT = 0.7  # Veto trades if highly correlated assets exceed this

# ==============================================================================
# 4. DATA VAULT & LOGGING DIRECTORY PATHS
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
# 5. HYPERPARAMETERS & MAGIC NUMBERS
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.65
MAX_HOLD_DAYS = 20
RR_RATIO = 2.0
OPTIONS_DTE = 21
TARGET_PREMIUM_GAIN = 0.50

# Excluded from the XGBoost feature matrix
METADATA_COLS = [
    'open', 'high', 'low', 'close', 'adj close', 'volume', 
    'target_label', 'option_target_label', 'ticker', 'sector', 'date'
]

# ==============================================================================
# 6. OUT-OF-CORE DASK & PARQUET CONFIGURATION
# ==============================================================================
PARQUET_BLOCKSIZE = "256MiB"

# Upgraded dtype_backend to 'pyarrow' to prevent Pandas memory duplication
DASK_READ_KWARGS = {
    "engine": "pyarrow",
    "blocksize": PARQUET_BLOCKSIZE,
    "split_row_groups": "infer",
    "dtype_backend": "pyarrow"  
}

# ==============================================================================
# 7. DYNAMIC PYARROW SCHEMA MAPPING
# ==============================================================================
def get_market_schema() -> pa.Schema:
    """
    Dynamically generates the memory-mapped PyArrow schema.
    If FUSION_ENABLED is True, it appends the 'sentiment_score' feature so Dask expects it.
    """
    fields = [
        ("date", pa.timestamp('ns')),
        ("ticker", pa.dictionary(pa.int32(), pa.string())),
        ("sector", pa.dictionary(pa.int32(), pa.string())),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
    ]
    
    if FUSION_ENABLED:
        fields.append(("sentiment_score", pa.float32()))
        
    return pa.schema(fields)
2. main.py (The Orchestrator)
Now we must ensure the argparse toggles execute before any of the other machine learning modules are imported into memory. If we import tournament or feature_compiler before injecting the boolean toggles into config, Python will cache the old states, breaking the fusion execution.
import os
import argparse
import logging

# ==============================================================================
# 1. ARGPARSE & GLOBAL STATE INJECTION
# ==============================================================================
# Parse arguments FIRST before touching downstream quantitative modules
parser = argparse.ArgumentParser(description="Quantum Sentinel V6 - Multi-Agent Engine")
parser.add_argument("--refresh-raw", action="store_true", help="Refresh raw market data")
parser.add_argument("--fusion", action="store_true", help="Enable LLM Sentiment Fusion Agent")
parser.add_argument("--disable-risk-manager", action="store_true", help="Disable the Risk Manager Agent")
args = parser.parse_args()

# Inject the toggled states into config BEFORE other modules load
import config
config.FUSION_ENABLED = args.fusion
config.RISK_MANAGER_ENABLED = not args.disable_risk_manager

# ==============================================================================
# 2. CENTRALIZED LOGGING CONFIGURATION
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

# ==============================================================================
# 3. DOWNSTREAM MODULE IMPORTS
# ==============================================================================
# Import custom modules AFTER config state is locked and logging is configured
import data_ingestion
import feature_compiler
import tournament

def main():
    logger.info(f"=== QUANTUM SENTINEL ORCHESTRATOR [{config.RUN_MODE} MODE] ===")
    logger.info(f"LLM Fusion Agent: {'ONLINE' if config.FUSION_ENABLED else 'OFFLINE'}")
    logger.info(f"Risk Manager Agent: {'ONLINE' if config.RISK_MANAGER_ENABLED else 'OFFLINE'}")
    
    if args.refresh_raw:
        universe = data_ingestion.get_survivorship_adjusted_universe()
        data_ingestion.build_raw_vault(universe)

    feature_compiler.compile_features_from_raw()
    
    director = tournament.ModularTournamentDirector()
    director.execute_gauntlet()

if __name__ == "__main__":
    main()