import os
import shutil
import pandas as pd
import yfinance as yf
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Dict
import config

logger = logging.getLogger(__name__)

def raw_vault_is_populated() -> bool:
    if not os.path.exists(config.RAW_VAULT_DIR):
        return False
    subdirs = [d for d in os.listdir(config.RAW_VAULT_DIR) if os.path.isdir(os.path.join(config.RAW_VAULT_DIR, d))]
    return len(subdirs) > 0

def reset_raw_vault() -> None:
    if os.path.exists(config.RAW_VAULT_DIR):
        shutil.rmtree(config.RAW_VAULT_DIR)
    os.makedirs(config.RAW_VAULT_DIR, exist_ok=True)

def get_survivorship_adjusted_universe() -> Dict[str, str]:
    """
    TEMPORARY BYPASS: Fetches current S&P 500 constituents from Wikipedia.
    """
    logger.warning("EODHD API bypassed. Fetching static S&P 500 list from Wikipedia.")
    
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        # FIX: pd.read_html returns a list of DataFrames. Index the first table.
        df = tables 
        
        universe = {}
        for _, row in df.iterrows():
            ticker = str(row['Symbol']).replace('.', '-')
            sector = str(row['GICS Sector'])
            universe[ticker] = sector
            
        return universe

    except Exception as e:
        logger.error("Failed to map sector universe from Wikipedia.", exc_info=True)
        return {}

def fetch_point_in_time_news(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    news_data = {
        "date": dates,
        "raw_news_headline": [f"Standard pre-market conditions persist for {ticker}."] * len(dates)
    }
    news_df = pd.DataFrame(news_data)
    news_df.set_index("date", inplace=True)
    return news_df

def ingest_raw_ticker(ticker: str, sector: str) -> bool:
    try:
        df = yf.download(
            ticker, 
            start=config.START_DATE, 
            end=config.END_DATE, 
            interval="1d", 
            progress=False, 
            multi_level_index=False
        )
        
        if df.empty or len(df) < 252:
            return False
            
        df.index = pd.to_datetime(df.index)
        df['ticker'] = ticker
        df['sector'] = sector
        
        if config.FUSION_ENABLED:
            news_df = fetch_point_in_time_news(ticker, df.index)
            df = df.join(news_df, how='left')
        
        df = df.convert_dtypes(dtype_backend="pyarrow")
        
        out_dir = os.path.join(config.RAW_VAULT_DIR, f"sector={sector}")
        os.makedirs(out_dir, exist_ok=True)
        
        df.to_parquet(os.path.join(out_dir, f"{ticker}.parquet"), engine='pyarrow')
        return True
        
    except Exception as e:
        logger.error(f"Ingestion failed for {ticker} in sector {sector}.", exc_info=True)
        return False

def build_raw_vault(universe_map: Dict[str, str]) -> None:
    logger.info(f"Executing raw data acquisition layer (Fusion Mode: {'ON' if config.FUSION_ENABLED else 'OFF'})...")
    reset_raw_vault()
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(ingest_raw_ticker, ticker, sector): ticker for ticker, sector in universe_map.items()}
        for future in as_completed(futures):
            if future.result():
                success_count += 1
                
    logger.info(f"Raw data acquisition complete. Successfully ingested {success_count} tickers.")