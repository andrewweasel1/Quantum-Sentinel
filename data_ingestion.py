import os
import shutil
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Dict
import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
logger = logging.getLogger(__name__)

def raw_vault_is_populated() -> bool:
    """Checks if the raw vault directory exists and contains partitioned data."""
    if not os.path.exists(config.RAW_VAULT_DIR):
        return False
    subdirs = [d for d in os.listdir(config.RAW_VAULT_DIR) if os.path.isdir(os.path.join(config.RAW_VAULT_DIR, d))]
    return len(subdirs) > 0

def reset_raw_vault() -> None:
    """Clears existing data to prevent duplicate or corrupted legacy files."""
    if os.path.exists(config.RAW_VAULT_DIR):
        shutil.rmtree(config.RAW_VAULT_DIR)
    os.makedirs(config.RAW_VAULT_DIR, exist_ok=True)

# ==============================================================================
# 1. SURVIVORSHIP-BIAS ADJUSTED UNIVERSE MAPPING
# ==============================================================================
def get_survivorship_adjusted_universe() -> Dict[str, str]:
    """
    Retrieves dynamic point-in-time index matrices to eliminate survivorship bias.
    Queries an institutional Point-in-Time database (e.g., EODHD Delisted Data API or Sharadar)
    to include companies that went bankrupt, were acquired, or delisted during the lookback window.
    """
    try:
        api_key = os.environ.get('EODHD_API_KEY')
        
        # Fallback mechanism if no API key is provided
        if not api_key:
            logger.warning("Point-In-Time API Key missing. Falling back to static CSV (WARNING: Introduces Survivorship Bias).")
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            df = pd.read_csv(url)
            return dict(zip(df['Symbol'].str.replace('.', '-', regex=False), df['Sector']))
            
        logger.info("Fetching Point-in-Time constituents, including delisted and acquired entities...")
        
        # The Delisted Data API gives us all historical tickers, preventing bias
        url = f"https://eodhistoricaldata.com/api/exchange-symbol-list/US?api_token={api_key}&fmt=json&delisted=1"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        universe = {}
        for _, row in df.iterrows():
            # We filter only for Common Stocks to avoid pulling ETFs and Mutual Funds
            if row.get('Type') == 'Common Stock':
                ticker = str(row['Code']).replace('.', '-')
                sector = str(row.get('Sector', 'Unknown'))
                universe[ticker] = sector
                
        return universe

    except Exception as e:
        logger.error("Failed to map point-in-time sector universe.", exc_info=True)
        return {}

# ==============================================================================
# 2. ISOLATED THREAD INGESTION
# ==============================================================================
def ingest_raw_ticker(ticker: str, sector: str) -> bool:
    """Thread-safe isolated extraction processing loop for a single ticker."""
    try:
        df = yf.download(
            ticker, 
            start=config.START_DATE, 
            end=config.END_DATE, 
            interval="1d", 
            progress=False, 
            multi_level_index=False
        )
        
        # Require a minimum of 1 trading year of data to calculate lookback indicators
        if df.empty or len(df) < 252:
            return False
            
        df['ticker'] = ticker
        df['sector'] = sector
        
        # Partition data efficiently by sector for downstream Dask mapping
        out_dir = os.path.join(config.RAW_VAULT_DIR, f"sector={sector}")
        os.makedirs(out_dir, exist_ok=True)
        
        df.to_parquet(os.path.join(out_dir, f"{ticker}.parquet"), engine='pyarrow')
        return True
        
    except Exception as e:
        # Logs exact network timeouts or data corruption errors 
        logger.error(f"Ingestion failed for {ticker} in sector {sector}.", exc_info=True)
        return False

# ==============================================================================
# 3. ASYNCHRONOUS ORCHESTRATION
# ==============================================================================
def build_raw_vault(universe_map: Dict[str, str]) -> None:
    """Asynchronous pipeline leveraging host thread scaling to populate the local vault."""
    logger.info("Executing point-in-time survivorship-adjusted raw data acquisition layer...")
    reset_raw_vault()
    
    success_count = 0
    
    # Scale threads automatically based on available CPU cores
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(ingest_raw_ticker, ticker, sector): ticker for ticker, sector in universe_map.items()}
        
        for future in as_completed(futures):
            if future.result():
                success_count += 1
                
    logger.info(f"Raw data acquisition complete. Successfully ingested {success_count} tickers.")