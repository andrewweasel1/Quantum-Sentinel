import os
import gc
import json
import itertools
import logging
from typing import Tuple, List, Optional, Any, Dict, Generator

import numpy as np
import pandas as pd
import dask.dataframe as dd
import xgboost as xgb
from itertools import combinations

import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. OUT-OF-CORE BATCH ITERATOR (VRAM OPTIMIZATION)
# ==============================================================================
class DataFrameIter(xgb.DataIter):
    """
    A custom XGBoost Data Iterator that feeds data in localized chunks [2].
    Designed to work with ExtMemQuantileDMatrix to prevent 12GB VRAM OOM 
    on hardware like the RTX 4070 by caching data across the host and device [3].
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, batch_size: int = 100000):
        # Initialize the base class. Setting on_host=True caches on CPU RAM [5]
        super().__init__(on_host=True)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(len(X) / batch_size))
        self.it = 0

    def reset(self) -> None:
        """Resets the iterator to the beginning [6]."""
        self.it = 0

    def next(self, input_data: Any) -> int:
        """Yields the next batch of data to the XGBoost internal API [2]."""
        if self.it == self.n_batches:
            return 0
        start = self.it * self.batch_size
        end = min((self.it + 1) * self.batch_size, len(self.X))
        
        # Pass the sliced batch to the internal XGBoost input function [2]
        input_data(data=self.X.iloc[start:end], label=self.y.iloc[start:end])
        
        self.it += 1
        return 1

# ==============================================================================
# 2. TOURNAMENT PIPELINE
# ==============================================================================
class ModularTournamentDirector:
    def __init__(self) -> None:
        """
        Initializes the Tournament Director by memory-mapping the compiled feature matrices.
        """
        logger.info(f"Mounting out-of-core data from {config.PROCESSED_VAULT_DIR}...")
        self.ddf: dd.DataFrame = dd.read_parquet(config.PROCESSED_VAULT_DIR, **config.DASK_READ_KWARGS)

    def generate_cpcv_splits(self, df: pd.DataFrame, n_groups: int = 6, test_groups: int = 2) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Executes Combinatorial Purged Cross-Validation (CPCV).
        Slices the timeseries into `n_groups`. Yields multiple train/test paths to generate
        a robust distribution of out-of-sample performance, rigorously purging and embargoing 
        overlapping observation horizons.
        """
        indices = np.array_split(df.index, n_groups)
        group_ids = list(range(n_groups))
        
        for test_combo in combinations(group_ids, test_groups):
            test_indices = []
            for i in test_combo:
                test_indices.extend(indices[i])
                
            test_df = df.loc[test_indices]
            train_df = df.drop(index=test_indices)
            
            embargo_gap = config.MAX_HOLD_DAYS 
            for test_idx in test_combo:
                boundary_start = indices[test_idx] - embargo_gap
                boundary_end = indices[test_idx][-1] + embargo_gap
                train_df = train_df.loc[~((train_df.index >= boundary_start) & (train_df.index <= boundary_end))]
                
            yield train_df, test_df

    def tune_sector_grid(self, sector_name: str) -> None:
        """
        Tests the hyperparameter grid across multiple CPCV backtest paths using VRAM-optimized structures.
        Records every combination's returns to export a full matrix for PBO evaluation.
        """
        logger.info(f"--- Initiating CPCV Tournament for Sector: {sector_name} ---")
        
        sector_df = self.ddf[self.ddf['sector'] == sector_name].compute().sort_values('date')
        sector_df.reset_index(drop=True, inplace=True)
        
        if len(sector_df) < 1000:
            logger.warning(f"Insufficient historical data for {sector_name}. Requires deeper panel for CPCV. Skipping...")
            return

        target_col = 'target_label' if config.RUN_MODE == "STANDARD" else 'option_target_label'
        features = [c for c in sector_df.columns if c not in config.METADATA_COLS]

        param_grid = {
            'max_depth': [7, 8],               
            'min_child_weight': [1.0, 3.0, 5.0],  
            'gamma': [0.1, 0.5],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8]  # Required < 1.0 fraction to enable stochastic sampling routines
        }
        
        keys, values = zip(*param_grid.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        returns_matrix = {}
        benchmark_returns = []
        best_sr: float = -1.0
        best_params: Optional[Dict] = None
        best_returns: np.ndarray = np.array([])
        benchmark_calculated = False
        
        # 1. Hyperparameter Search Space
        for trial_idx, params in enumerate(grid_combinations):
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            params['objective'] = 'binary:logistic'
            params['grow_policy'] = 'lossguide'
            
            # RTX 4070 Optimization: Instruct GPU to sample instances with higher gradient values [4]
            params['sampling_method'] = 'gradient_based' 
            
            trial_oos_returns = []
            
            # 2. Combinatorial Purged Cross-Validation
            for train_df, test_df in self.generate_cpcv_splits(sector_df):
                
                X_train, y_train = train_df[features], train_df[target_col]
                X_test, y_test = test_df[features], test_df[target_col]

                # RTX 4070 Optimization: Initialize custom Iterators [2, 5]
                train_iter = DataFrameIter(X_train, y_train, batch_size=100000)
                test_iter = DataFrameIter(X_test, y_test, batch_size=100000)
                
                # Stream the iterators dynamically into the ExtMemQuantileDMatrix [3]
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
                
                # 3. Translate Probabilities to Simulated Returns
                preds_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
                signals = (preds_proba > config.CONFIDENCE_THRESHOLD).astype(int)
                
                forward_returns = test_df['close'].pct_change().fillna(0).values
                strategy_returns = signals * forward_returns
                trial_oos_returns.extend(strategy_returns)

                if not benchmark_calculated:
                    benchmark_returns.extend(forward_returns)

                del dtrain, dtest, train_iter, test_iter
                
            benchmark_calculated = True
            trial_oos_returns = np.array(trial_oos_returns)
            
            returns_matrix[f"trial_{trial_idx}"] = trial_oos_returns
            
            if np.std(trial_oos_returns) > 0:
                trial_sr = np.mean(trial_oos_returns) / np.std(trial_oos_returns)
            else:
                trial_sr = 0.0
                
            if trial_sr > best_sr:
                best_sr = trial_sr
                best_params = params
                best_returns = trial_oos_returns

        # 4. Matrix Export & Candidate Staging
        if best_params is not None and len(best_returns) > 0:
            
            # Train the final candidate using the external memory matrices
            full_iter = DataFrameIter(sector_df[features], sector_df[target_col], batch_size=100000)
            d_full = xgb.ExtMemQuantileDMatrix(full_iter)
            candidate_booster = xgb.train(best_params, d_full, num_boost_round=100)
            
            os.makedirs(config.PROD_MODELS_DIR, exist_ok=True)
            candidate_booster.save_model(os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate.json"))
            
            with open(os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate_features.json"), "w") as f:
                json.dump(features, f)
            
            matrix_df = pd.DataFrame(returns_matrix)
            matrix_df.to_parquet(f"returns_matrix_{sector_name}.parquet", engine='pyarrow')
            
            bench_df = pd.DataFrame({"benchmark": benchmark_returns, "champion": best_returns})
            bench_df.to_parquet(f"benchmark_{sector_name}.parquet", engine='pyarrow')

            logger.info(f"[{sector_name}] Matrix & Benchmark exported. Raw candidate SR: {best_sr:.3f}. Awaiting Evaluator verification.")
        else:
            logger.warning(f"Sector {sector_name} failed to find any model with positive variance.")
        
        gc.collect()

    def execute_gauntlet(self) -> None:
        """
        Orchestrates the out-of-core pipeline across all unique sectors.
        """
        if not os.path.exists(config.PROCESSED_VAULT_DIR):
            logger.error("Processed features missing. Run feature_compiler.py first.")
            return

        logger.info("=== COMMENCING OUT-OF-CORE XGBOOST TOURNAMENT ===")
        
        unique_sectors = self.ddf['sector'].unique().compute()
        
        for sector in unique_sectors:
            if pd.isna(sector):
                continue
                
            self.tune_sector_grid(str(sector))
            
        logger.info("=== TOURNAMENT CONCLUDED. MATRICES STAGED FOR EVALUATOR. ===")