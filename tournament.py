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
from numba import njit

import config

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. OUT-OF-CORE BATCH ITERATOR & RISK SIMULATOR
# ==============================================================================
class DataFrameIter(xgb.DataIter):
    def __init__(self, X: pd.DataFrame, y: pd.Series, batch_size: int = 100000):
        super().__init__(on_host=True)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(len(X) / batch_size))
        self.it = 0

    def reset(self) -> None:
        self.it = 0

    def next(self, input_data: Any) -> int:
        if self.it == self.n_batches:
            return 0
        start = self.it * self.batch_size
        end = min((self.it + 1) * self.batch_size, len(self.X))
        input_data(data=self.X.iloc[start:end], label=self.y.iloc[start:end])
        self.it += 1
        return 1

@njit(fastmath=True)
def simulate_risk_manager_njit(signals, closes, lows, atrs, atr_multiplier, max_risk_pct):
    """
    Simulates the Risk Manager's fractional sizing and dynamic ATR stops during CPCV.
    Executes in microseconds via LLVM compiler to prevent CPU bottlenecking.
    """
    n = len(signals)
    returns = np.zeros(n)
    
    for i in range(n - 1):
        if signals[i] == 1 and atrs[i] > 0:
            entry = closes[i]
            stop = entry - (atr_multiplier * atrs[i])
            
            # Fractional Position Sizing
            risk_distance = (entry - stop) / entry
            size = max_risk_pct / risk_distance if risk_distance > 0 else 0.0
            size = min(size, 1.0) # Hard cap to prevent leverage blowouts
            
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
            
            embargo_gap = config.MAX_HOLD_DAYS 
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
            'max_depth': [4, 5],               
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
        
        for trial_idx, params in enumerate(grid_combinations):
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            params['objective'] = 'binary:logistic'
            params['grow_policy'] = 'lossguide'
            params['sampling_method'] = 'gradient_based' 
            
            trial_oos_returns = []
            trial_sentiment = []
            
            for train_df, test_df in self.generate_cpcv_splits(sector_df):
                X_train, y_train = train_df[features], train_df[target_col]
                X_test, y_test = test_df[features], test_df[target_col]

                train_iter = DataFrameIter(X_train, y_train, batch_size=100000)
                test_iter = DataFrameIter(X_test, y_test, batch_size=100000)
                
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
                
                # SHIELD AGENT: Apply Risk Manager Simulation
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
            full_iter = DataFrameIter(sector_df[features], sector_df[target_col], batch_size=100000)
            d_full = xgb.ExtMemQuantileDMatrix(full_iter)
            candidate_booster = xgb.train(best_params, d_full, num_boost_round=100)
            
            # FUSION EVALUATION: Feature Displacement Logging
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