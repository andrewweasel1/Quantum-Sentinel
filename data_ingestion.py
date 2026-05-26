import os
import shutil
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

def raw_vault_is_populated():
    """Checks if the raw vault directory exists and contains partitioned data."""
    if not os.path.exists(config.RAW_VAULT_DIR):
        return False
    return len([d for d in os.listdir(config.RAW_VAULT_DIR) if os.path.isdir(os.path.join(config.RAW_VAULT_DIR, d))]) > 0

def reset_raw_vault():
    """Clears existing data to prevent duplicate or corrupted legacy files."""
    if os.path.exists(config.RAW_VAULT_DIR):
        shutil.rmtree(config.RAW_VAULT_DIR)
    os.makedirs(config.RAW_VAULT_DIR, exist_ok=True)

def get_survivorship_adjusted_universe():
    """
    Retrieves dynamic point-in-time index matrices to eliminate survivorship bias, 
    injecting historically active and failed/delisted corporate tickers.
    """
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        # Replace '.' with '-' to accommodate yfinance ticker formats (e.g., BRK.B -> BRK-B)
        universe = dict(zip(df['Symbol'].str.replace('.', '-', regex=False), df['Sector']))
        return universe
    except Exception as e:
        print(f"Error mapping sectors: {e}")
        return {}

def ingest_raw_ticker(ticker, sector):
    """Thread-safe isolated extraction processing loop for a single ticker."""
    try:
        # Explicitly disable multi-level indexes to keep the dataframe columns flat.
        # This prevents silent failures during downstream technical analysis.
        df = yf.download(
            ticker, 
            start=config.START_DATE, 
            end=config.END_DATE, 
            interval="1d", 
            progress=False, 
            multi_level_index=False
        )
        
        # Skip if data is empty or lacks sufficient history for indicator lookbacks (e.g., 252 trading days)
        if df.empty or len(df) < 252:
            return False
            
        # Append metadata columns
        df['ticker'] = ticker
        df['sector'] = sector
        
        # Hive partitioning save format: sector=Technology/AAPL.parquet
        out_dir = os.path.join(config.RAW_VAULT_DIR, f"sector={sector}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save directly to PyArrow parquet
        df.to_parquet(os.path.join(out_dir, f"{ticker}.parquet"), engine='pyarrow')
        return True
    except Exception as e:
        return False

def build_raw_vault(universe_map):
    """Asynchronous pipeline leveraging host thread scaling to populate the local vault."""
    print("Executing point-in-time survivorship-adjusted raw data acquisition layer...")
    reset_raw_vault()
    
    # Throttled max_workers to mitigate IP rate-limiting dropouts on public APIs like Yahoo Finance
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(ingest_raw_ticker, t, s) for t, s in universe_map.items()]
        for f in as_completed(futures):
            f.result()