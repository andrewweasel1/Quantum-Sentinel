import os
import argparse
import logging
import pandas as pd

# ==============================================================================
# 1. ARGPARSE & GLOBAL STATE INJECTION
# ==============================================================================
parser = argparse.ArgumentParser(description="Quantum Sentinel V6 - Multi-Agent Engine")
parser.add_argument("--refresh-raw", action="store_true", help="Refresh raw market data")
parser.add_argument("--fusion", action="store_true", help="Enable LLM Sentiment Fusion Agent")
parser.add_argument("--disable-risk-manager", action="store_true", help="Disable the Risk Manager Agent")
parser.add_argument("--evaluate", action="store_true", help="Run the statistical Evaluator to promote models")
parser.add_argument("--live", action="store_true", help="Launch the Live Trading Sandbox")
args = parser.parse_args()

import config
config.FUSION_ENABLED = args.fusion
config.RISK_MANAGER_ENABLED = not args.disable_risk_manager

# ==============================================================================
# 2. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
# FIX: Logging MUST be configured before importing dependent modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.SYSTEM_LOG_FILE),  
        logging.StreamHandler()                       
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 3. DOWNSTREAM MODULE IMPORTS
# ==============================================================================
import data_ingestion
import feature_compiler
import tournament
import evaluator
import live_trader

def main():
    logger.info(f"=== QUANTUM SENTINEL ORCHESTRATOR [{config.RUN_MODE} MODE] ===")
    logger.info(f"LLM Fusion Agent: {'ONLINE' if config.FUSION_ENABLED else 'OFFLINE'}")
    logger.info(f"Risk Manager Agent: {'ONLINE' if config.RISK_MANAGER_ENABLED else 'OFFLINE'}")
    
    # PHASE 1: DATA PIPELINE & TRAINING
    if args.refresh_raw:
        universe = data_ingestion.get_survivorship_adjusted_universe()
        data_ingestion.build_raw_vault(universe)
        feature_compiler.compile_features_from_raw()
        
        director = tournament.ModularTournamentDirector()
        director.execute_gauntlet()

    # PHASE 2: STATISTICAL EVALUATION
    if args.evaluate:
        stat_evaluator = evaluator.QuantitativeEvaluator()
        stat_evaluator.run_evaluation_gauntlet()

    # PHASE 3: LIVE MARKET EXECUTION
    if args.live:
        logger.info("Initializing Live Trading Sandbox via Alpaca...")
        sandbox = live_trader.LiveTradingSandbox(is_paper=True)
        
        logger.info("Sourcing live market data for active champions...")
        live_market_df = pd.read_parquet(config.PROCESSED_VAULT_DIR, engine="pyarrow")
        
        latest_date = live_market_df['date'].max()
        current_data = live_market_df[live_market_df['date'] == latest_date].copy()
        
        sandbox.execute_live_cycle(current_data)

if __name__ == "__main__":
    main()