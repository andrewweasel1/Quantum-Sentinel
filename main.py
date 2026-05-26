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
# FIXED: Corrected dunder method from **name** to __name__
logger = logging.getLogger(__name__)

# Import custom modules AFTER logging is configured so they inherit the root logger
import data_ingestion
import feature_compiler
import tournament

def main():
    logger.info(f"=== QUANTUM SENTINEL ORCHESTRATOR [{config.RUN_MODE} MODE] ===")
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="AI Quantitative Strategy Tournament Runner")
    parser.add_argument(
        "--refresh-raw", 
        action="store_true", 
        help="Force a point-in-time constituent refresh and clean re-download of all raw daily data"
    )
    parser.add_argument(
        "--refresh-features", 
        action="store_true", 
        help="Force offline Dask re-compilation of technical indicator feature matrices"
    )
    args = parser.parse_args()

    # ==============================================================================
    # PHASE 1: RAW DATA INGESTION
    # ==============================================================================
    if args.refresh_raw:
        logger.info("[COMMAND] --refresh-raw detected. Synchronizing raw market data...")
        # Retrieves the survivorship-bias free universe map
        universe = data_ingestion.get_survivorship_adjusted_universe()
        # Multi-threaded download into the raw Parquet vault
        data_ingestion.build_raw_vault(universe)
        
        # If raw data is refreshed, we must automatically recompile the downstream features
        args.refresh_features = True

    # ==============================================================================
    # PHASE 2: OFFLINE FEATURE COMPILATION
    # ==============================================================================
    if args.refresh_features:
        logger.info("[COMMAND] --refresh-features detected. Compiling offline indicator matrices...")
        # Maps partitions via Dask and dumps to the processed Hive-partitioned vault
        feature_compiler.compile_features_from_raw()

    # ==============================================================================
    # PHASE 3: OUT-OF-CORE MACHINE LEARNING TOURNAMENT
    # ==============================================================================
    logger.info("[COMMAND] Initializing XGBoost Out-of-Core Tournament...")
    director = tournament.ModularTournamentDirector()
    director.execute_gauntlet()

# FIXED: Corrected dunder method from **name** == "**main**" to standard execution gate
if __name__ == "__main__":
    main()