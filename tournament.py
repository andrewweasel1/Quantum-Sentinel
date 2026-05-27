import os
import gc
import json
import itertools
import logging
from typing import Tuple, List, Optional, Any, Dict, Generator

import numpy as np
import pandas as pd
import dask.dataframe as dd
import pyarrow.parquet as pq
import xgboost as xgb
from itertools import combinations
from numba import njit

import config

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. OUT-OF-CORE PYARROW ITERATOR (TRUE ZERO-COPY) & RISK SIMULATOR
# ==============================================================================
class ParquetDataIter(xgb.DataIter):
    """
    A true out-of-core XGBoost Data Iterator.
    Reads chunked row groups directly from disk via PyArrow Tables, ensuring neither 
    Host RAM nor GPU VRAM is ever exhausted during massive CPCV evaluations.
    """
    def __init__(self, file_path: str, features: List[str], target_col: str):
        # Instantiate on_host=True so DMatrix loads pointers into GPU batches
        super().__init__(on_host=True)
        self.file_path = file_path
        self.features = features
        self.target_col = target_col
        
        # Initialize PyArrow Parquet reader
        self.pf = pq.ParquetFile(file_path)
        self.num_row_groups = self.pf.num_row_groups
        self.it = 0

    def reset(self) -> None:
        self.it = 0

    def next(self, input_data: Any) -> int:
        if self.it == self.num_row_groups:
            return 0
            
        # FIX: Read exactly one row group from disk as a native PyArrow Table.
        # We REMOVED .to_pandas() to prevent memory duplication and deserialization.
        chunk_table = self.pf.read_row_group(self.it, columns=self.features + [self.target_col])
        
        # Isolate features and target natively using Arrow's zero-copy .select()
        X_chunk = chunk_table.select(self.features)
        y_chunk = chunk_table.select([self.target_col])
        
        # Pass the raw Arrow Tables directly into the XGBoost CUDA allocator
        input_data(data=X_chunk, label=y_chunk)
        self.it += 1
        return 1

@njit(fastmath=True)
def simulate_risk_manager_njit(signals, closes, lows, atrs, atr_multiplier, max_risk_pct):
    n = len(signals)
    returns = np.zeros(n)
    
    for i in range(n - 1):
        if signals[i] == 1 and atrs[i] > 0:
            entry = closes[i]
            stop = entry - (atr_multiplier * atrs[i])
            
            risk_distance = (entry - stop) / entry
            size = max_risk_pct / risk_distance if risk_distance > 0 else 0.0
            size = min(size, 1.0) 
            
            if lows[i+1] <= stop:
                returns[i] = -risk_distance * size
            else:
                returns[i] = ((closes[i+1] - entry) / entry) * size
                
    return returns

# ==============================================================================
# 2. TOURNAMENT PIPELINE
# ==============================================================================
class ModularTournamentDirector:
    def __init__(self) -> None:
        logger.info(f"Mounting out-of-core data from {config.PROCESSED_VAULT_DIR}...")
        self.ddf: dd.DataFrame = dd.read_parquet(config.PROCESSED_VAULT_DIR, **config.DASK_READ_KWARGS)

    def generate_cpcv_splits(self, df: pd.DataFrame, n_groups: int = 6, test_groups: int = 2) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        indices = np.array_split(df.index, n_groups)
        group_ids = list(range(n_groups))
        
        for test_combo in combinations(group_ids, test_groups):
            test_indices = []
            for i in test_combo:
                test_indices.extend(indices[i])
                
            test_df = df.loc[test_indices]
            train_df = df.drop(index=test_indices)
            
            # FIX: Convert integer to Timedelta and isolate scalars
            embargo_gap = pd.Timedelta(days=config.MAX_HOLD_DAYS) 
            for test_idx in test_combo:
                boundary_start = indices[test_idx] - embargo_gap
                boundary_end = indices[test_idx][-1] + embargo_gap
                train_df = train_df.loc[~((train_df.index >= boundary_start) & (train_df.index <= boundary_end))]
                
            yield train_df, test_df

    def tune_sector_grid(self, sector_name: str) -> None:
        logger.info(f"--- Initiating CPCV Tournament for Sector: {sector_name} ---")
        
        sector_df = self.ddf[self.ddf['sector'] == sector_name].compute().sort_values('date')
        sector_df.reset_index(drop=True, inplace=True)
        
        if len(sector_df) < 1000:
            return

        target_col = 'target_label' if config.RUN_MODE == "STANDARD" else 'option_target_label'
        features = [c for c in sector_df.columns if c not in config.METADATA_COLS]

        param_grid = {
            'max_depth': [3-5],             
            'min_child_weight': [1.0, 3.0],  
            'gamma': [0.1],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8]  
        }
        
        keys, values = zip(*param_grid.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        returns_matrix = {}
        benchmark_returns = []
        fusion_signals = []
        
        best_sr: float = -1.0
        best_params: Optional[Dict] = None
        best_returns: np.ndarray = np.array([])
        benchmark_calculated = False
        
        temp_train_path = f"temp_train_{sector_name}.parquet"
        temp_test_path = f"temp_test_{sector_name}.parquet"
        temp_full_path = f"temp_full_{sector_name}.parquet"
        
        for trial_idx, params in enumerate(grid_combinations):
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            params['objective'] = 'binary:logistic'
            params['grow_policy'] = 'lossguide'
            params['sampling_method'] = 'gradient_based' 
            
            trial_oos_returns = []
            trial_sentiment = []
            
            for train_df, test_df in self.generate_cpcv_splits(sector_df):
                
                # Write CPCV splits dynamically to disk with a rigid row_group_size constraint 
                # This explicitly forces PyArrow to build the chunks the iterator will read
                train_df.to_parquet(temp_train_path, engine='pyarrow', row_group_size=100000)
                test_df.to_parquet(temp_test_path, engine='pyarrow', row_group_size=100000)

                train_iter = ParquetDataIter(temp_train_path, features, target_col)
                test_iter = ParquetDataIter(temp_test_path, features, target_col)
                
                dtrain = xgb.ExtMemQuantileDMatrix(train_iter)
                dtest = xgb.ExtMemQuantileDMatrix(test_iter, ref=dtrain)
                
                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    evals=[(dtrain, 'train'), (dtest, 'eval')],
                    early_stopping_rounds=25,
                    verbose_eval=False
                )
                
                preds_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
                signals = (preds_proba > config.CONFIDENCE_THRESHOLD).astype(int)
                
                if config.RISK_MANAGER_ENABLED:
                    closes = test_df['close'].values
                    lows = test_df['low'].values
                    atrs = test_df['atr'].values
                    strategy_returns = simulate_risk_manager_njit(
                        signals, closes, lows, atrs, config.ATR_STOP_MULTIPLIER, config.MAX_RISK_PER_TRADE
                    )
                else:
                    forward_returns = test_df['close'].pct_change().fillna(0).values
                    strategy_returns = signals * forward_returns
                    
                trial_oos_returns.extend(strategy_returns)

                if config.FUSION_ENABLED:
                    trial_sentiment.extend(test_df['sentiment_score'].values)

                if not benchmark_calculated:
                    benchmark_returns.extend(test_df['close'].pct_change().fillna(0).values)

                del dtrain, dtest, train_iter, test_iter
                
            benchmark_calculated = True
            trial_oos_returns = np.array(trial_oos_returns)
            
            returns_matrix[f"trial_{trial_idx}"] = trial_oos_returns
            
            trial_sr = np.mean(trial_oos_returns) / np.std(trial_oos_returns) if np.std(trial_oos_returns) > 0 else 0.0
                
            if trial_sr > best_sr:
                best_sr = trial_sr
                best_params = params
                best_returns = trial_oos_returns
                if config.FUSION_ENABLED:
                    fusion_signals = trial_sentiment

        if best_params is not None and len(best_returns) > 0:
            
            # Final Candidate Training mapped directly through PyArrow out-of-core
            sector_df.to_parquet(temp_full_path, engine='pyarrow', row_group_size=100000)
            full_iter = ParquetDataIter(temp_full_path, features, target_col)
            d_full = xgb.ExtMemQuantileDMatrix(full_iter)
            
            candidate_booster = xgb.train(best_params, d_full, num_boost_round=100)
            
            if config.FUSION_ENABLED:
                importances = candidate_booster.get_score(importance_type='gain')
                sorted_imp = sorted(importances.items(), key=lambda x: x[6], reverse=True)
                top_features = [x for x in sorted_imp[:5]]
                if 'sentiment_score' in top_features:
                    logger.info(f"[{sector_name}] FUSION SUCCESS: LLM Sentiment generated alpha (Top 5 Feature).")
                else:
                    logger.warning(f"[{sector_name}] FUSION DECAY: LLM Sentiment was displaced by technicals. Acting as noise.")

            os.makedirs(config.PROD_MODELS_DIR, exist_ok=True)
            candidate_booster.save_model(os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate.json"))
            
            with open(os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate_features.json"), "w") as f:
                json.dump(features, f)
            
            matrix_df = pd.DataFrame(returns_matrix)
            matrix_df.to_parquet(f"returns_matrix_{sector_name}.parquet", engine='pyarrow')
            
            bench_dict = {"benchmark": benchmark_returns, "champion": best_returns}
            if config.FUSION_ENABLED:
                bench_dict["sentiment_score"] = fusion_signals
                
            bench_df = pd.DataFrame(bench_dict)
            bench_df.to_parquet(f"benchmark_{sector_name}.parquet", engine='pyarrow')

            logger.info(f"[{sector_name}] Matrix & Benchmark exported. Awaiting Evaluator verification.")
        
        # Safely sweep temporary out-of-core PyArrow files from disk
        for file in [temp_train_path, temp_test_path, temp_full_path]:
            if os.path.exists(file):
                os.remove(file)
                
        gc.collect()

    def execute_gauntlet(self) -> None:
        if not os.path.exists(config.PROCESSED_VAULT_DIR):
            return

        logger.info(f"=== COMMENCING OUT-OF-CORE TOURNAMENT (Fusion: {config.FUSION_ENABLED}, Risk Manager: {config.RISK_MANAGER_ENABLED}) ===")
        unique_sectors = self.ddf['sector'].unique().compute()
        
        for sector in unique_sectors:
            if pd.isna(sector): continue
            self.tune_sector_grid(str(sector))
            
        logger.info("=== TOURNAMENT CONCLUDED. MATRICES STAGED FOR EVALUATOR. ===")