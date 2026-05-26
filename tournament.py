import os
import gc
import itertools
import logging
from typing import Tuple, List, Optional, Any, Dict

import numpy as np
import pandas as pd
import dask.dataframe as dd
import xgboost as xgb
from sklearn.metrics import precision_score

import config
import exporter

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
logger = logging.getLogger(__name__)

class ModularTournamentDirector:
    def __init__(self) -> None:
        """
        Initializes the Tournament Director by memory-mapping the compiled feature
        matrices directly from the processed vault using Dask and PyArrow.
        """
        logger.info(f"Mounting out-of-core data from {config.PROCESSED_VAULT_DIR}...")
        self.ddf: dd.DataFrame = dd.read_parquet(config.PROCESSED_VAULT_DIR, **config.DASK_READ_KWARGS)
        self.results: List[Dict[str, Any]] = []
        self.tuning_logs: List[Dict[str, Any]] = []

    def apply_purged_embargo_split(self, df: pd.DataFrame, train_fraction: float = 0.8, embargo_window: int = config.MAX_HOLD_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes Purging and Embargoing to prevent data leakage (look-ahead bias).
        Uses the globally configured MAX_HOLD_DAYS to dynamically pad the embargo blackout.
        """
        split_idx = int(len(df) * train_fraction)
        train_end_idx = split_idx - embargo_window
        
        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[split_idx:]
        
        return train_df, test_df

    def tune_sector_grid(self, sector_name: str) -> Optional[Tuple[xgb.Booster, List[str], float]]:
        """
        Filters the out-of-core data for a specific sector, caches it dynamically to the GPU 
        using QuantileDMatrix, and performs a rigorous hyperparameter grid search.
        """
        logger.info(f"--- Initiating Grid Search Tournament for Sector: {sector_name} ---")
        
        sector_df = self.ddf[self.ddf['sector'] == sector_name].compute().sort_values('date')
        
        if len(sector_df) < 500:
            logger.warning(f"Insufficient historical data for {sector_name}. Skipping...")
            return None

        target_col = 'target_label' if config.RUN_MODE == "STANDARD" else 'option_target_label'
        features = [c for c in sector_df.columns if c not in config.METADATA_COLS]

        train_df, test_df = self.apply_purged_embargo_split(sector_df)
        
        X_train, y_train = train_df[features], train_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]

        logger.info(f"Pre-caching {sector_name} features into GPU VRAM...")
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
        dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)

        param_grid = {
            'max_depth': [1-3],
            'min_child_weight': [1, 2, 4],
            'gamma': [0.1, 0.5],
            'learning_rate': [0.01, 0.05]
        }
        
        keys, values = zip(*param_grid.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_precision: float = -1.0
        best_booster: Optional[xgb.Booster] = None
        
        for params in grid_combinations:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            params['objective'] = 'binary:logistic'
            params['grow_policy'] = 'lossguide'
            
            evals_result: Dict[str, Any] = {}
            
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False,
                evals_result=evals_result
            )
            
            preds_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
            preds_binary = (preds_proba > config.CONFIDENCE_THRESHOLD).astype(int)
            
            if sum(preds_binary) > 0:
                precision = float(precision_score(y_test, preds_binary))
            else:
                precision = 0.0
                
            self.tuning_logs.append({
                'sector': sector_name,
                'max_depth': params['max_depth'],
                'gamma': params['gamma'],
                'precision': precision,
                'best_iteration': bst.best_iteration
            })

            if precision > best_precision and sum(preds_binary) >= 10:
                best_precision = precision
                best_booster = bst

        if best_booster is not None:
            logger.info(f"Sector {sector_name} Champion Precision: {best_precision*100:.2f}%")
        else:
            logger.warning(f"Sector {sector_name} failed to find a profitable model.")

        del dtrain
        del dtest
        gc.collect()

        if best_booster is not None:
            return best_booster, features, best_precision
        return None

    def execute_gauntlet(self) -> None:
        """
        Orchestrates the entire out-of-core pipeline, testing the hyperparameter
        grid across all unique sectors found in the Dask partitions.
        """
        if not os.path.exists(config.PROCESSED_VAULT_DIR):
            logger.error("Processed features missing. Run feature_compiler.py first.")
            return

        logger.info("=== COMMENCING OUT-OF-CORE XGBOOST TOURNAMENT ===")
        
        unique_sectors = self.ddf['sector'].unique().compute()
        
        for sector in unique_sectors:
            if pd.isna(sector):
                continue
                
            champion = self.tune_sector_grid(str(sector))
            
            if champion:
                bst, features, precision = champion
                self.results.append({
                    'sector': str(sector),
                    'composite_score': float(precision),
                    'feat_count': int(len(features))
                })
                
                # Automatically deploy the champion via the exporter module
                try:
                    exporter.deploy_champion(bst, features, str(sector), precision)
                except Exception as e:
                    logger.error(f"Error deploying champion for {sector}.", exc_info=True)

        if self.results:
            pd.DataFrame(self.results).to_parquet(config.RESULTS_FILE, engine='pyarrow')
        if self.tuning_logs:
            pd.DataFrame(self.tuning_logs).to_parquet("grid_search_raw_logs.parquet", engine='pyarrow')
            
        logger.info("=== TOURNAMENT CONCLUDED. TELEMETRY EXPORTED. ===")