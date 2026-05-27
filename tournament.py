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
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

import config

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. OUT-OF-CORE PYARROW ITERATOR & RISK SIMULATOR
# ==============================================================================
class ParquetDataIter(xgb.DataIter):
    """Zero-copy out-of-core XGBoost Data Iterator via PyArrow."""
    def __init__(self, file_path: str, features: List[str], target_col: str):
        super().__init__(on_host=True)
        self.file_path = file_path
        self.features = features
        self.target_col = target_col
        self.pf = pq.ParquetFile(file_path)
        self.num_row_groups = self.pf.num_row_groups
        self.it = 0

    def reset(self) -> None:
        self.it = 0

    def next(self, input_data: Any) -> int:
        if self.it == self.num_row_groups:
            return 0
        chunk_table = self.pf.read_row_group(self.it, columns=self.features + [self.target_col])
        input_data(data=chunk_table.select(self.features), label=chunk_table.select([self.target_col]))
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

def asymmetric_financial_loss(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom objective: Penalizes False Positives (Capital Loss) 5x more than False Negatives (Opportunity Loss).
    """
    labels = dtrain.get_label()
    preds_prob = 1.0 / (1.0 + np.exp(-preds))
    
    # Gradient and Hessian of logloss
    grad = preds_prob - labels
    hess = preds_prob * (1.0 - preds_prob)
    
    # Asymmetric penalty multiplier
    penalty_fp = 5.0
    penalty_fn = 1.0
    
    grad = np.where(labels == 0, grad * penalty_fp, grad * penalty_fn)
    hess = np.where(labels == 0, hess * penalty_fp, hess * penalty_fn)
    
    return grad, hess

# ==============================================================================
# 2. TOURNAMENT PIPELINE & CLUSTERED FEATURE SELECTION
# ==============================================================================
class ModularTournamentDirector:
    def __init__(self) -> None:
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
            
            purge_gap = pd.Timedelta(days=config.MAX_HOLD_DAYS)
            embargo_gap = pd.Timedelta(days=5) 
            
            for test_idx in test_combo:
                boundary_start = indices[test_idx] - purge_gap
                boundary_end = indices[test_idx][-1] + purge_gap + embargo_gap
                train_df = train_df.loc[~((train_df.index >= boundary_start) & (train_df.index <= boundary_end))]
                
            yield train_df, test_df

    def tune_sector_grid(self, sector_name: str) -> None:
        logger.info(f"--- CPCV Tournament for {sector_name} ---")
        sector_df = self.ddf[self.ddf['sector'] == sector_name].compute().sort_values('date')
        if len(sector_df) < 1000: return

        target_col = 'target_label'
        features = [c for c in sector_df.columns if c not in config.METADATA_COLS]

        param_grid = {'max_depth': [1, 2], 'learning_rate': [0.01, 0.05]}
        keys, values = zip(*param_grid.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        returns_matrix = {}
        benchmark_returns = []
        best_sr, best_params = -1.0, None
        
        temp_train_path = f"temp_train_{sector_name}.parquet"
        
        for trial_idx, params in enumerate(grid_combinations):
            params.update({'tree_method': 'hist', 'device': 'cuda', 'disable_default_eval_metric': 1})
            trial_oos_returns = []
            
            try:
                for train_df, test_df in self.generate_cpcv_splits(sector_df):
                    train_df.to_parquet(temp_train_path, engine='pyarrow', row_group_size=100000)
                    train_iter = ParquetDataIter(temp_train_path, features, target_col)
                    dtrain = xgb.ExtMemQuantileDMatrix(train_iter)
                    
                    # Test DMatrix materialized in RAM to prevent IO thrashing
                    dtest = xgb.DMatrix(test_df[features], label=test_df[target_col])
                    
                    bst = xgb.train(
                        params, dtrain, num_boost_round=300,
                        obj=asymmetric_financial_loss,
                        evals=[(dtrain, 'train'), (dtest, 'eval')],
                        early_stopping_rounds=25, verbose_eval=False,
                        custom_metric=lambda p, d: ('error', np.mean((1.0 / (1.0 + np.exp(-p)) > 0.5) != d.get_label()))
                    )
                    
                    preds = 1.0 / (1.0 + np.exp(-bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))))
                    signals = (preds > config.CONFIDENCE_THRESHOLD).astype(int)
                    
                    if config.RISK_MANAGER_ENABLED:
                        returns = simulate_risk_manager_njit(
                            signals, test_df['close'].values, test_df['low'].values, test_df['atr'].values,
                            config.ATR_STOP_MULTIPLIER, config.MAX_RISK_PER_TRADE
                        )
                    else:
                        returns = signals * test_df['close'].pct_change().fillna(0).values
                    trial_oos_returns.extend(returns)
                    
                    if trial_idx == 0: benchmark_returns.extend(test_df['close'].pct_change().fillna(0).values)
                    del dtrain, dtest, train_iter
                    
            finally:
                if os.path.exists(temp_train_path): os.remove(temp_train_path)
            
            trial_oos_returns = np.array(trial_oos_returns)
            returns_matrix[f"trial_{trial_idx}"] = trial_oos_returns
            trial_sr = np.mean(trial_oos_returns) / np.std(trial_oos_returns) if np.std(trial_oos_returns) > 0 else 0.0
                
            if trial_sr > best_sr:
                best_sr = trial_sr
                best_params = params

        # -------------------------------------------------------------------------
        # GLOBAL CLUSTERED FEATURE SELECTION (CFS)
        # -------------------------------------------------------------------------
        split_idx = int(len(sector_df) * 0.8)
        cfi_train, cfi_test = sector_df.iloc[:split_idx], sector_df.iloc[split_idx:]
        
        d_cfi_train = xgb.DMatrix(cfi_train[features], label=cfi_train[target_col])
        d_cfi_test = xgb.DMatrix(cfi_test[features], label=cfi_test[target_col])
        cfi_bst = xgb.train(best_params, d_cfi_train, obj=asymmetric_financial_loss, num_boost_round=100)
        
        base_preds = 1.0 / (1.0 + np.exp(-cfi_bst.predict(d_cfi_test)))
        base_returns = (base_preds > config.CONFIDENCE_THRESHOLD).astype(int) * cfi_test['close'].pct_change().fillna(0).values
        base_sharpe = np.mean(base_returns) / np.std(base_returns) if np.std(base_returns) > 0 else 0.0

        # Correlation distance matrix & Ward linkage
        corr_matrix = cfi_train[features].corr(method='spearman').fillna(0).values
        dist_matrix = np.sqrt(np.clip(0.5 * (1 - corr_matrix), 0, 1))
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = sch.linkage(condensed_dist, method='ward')
        clusters = sch.fcluster(linkage_matrix, t=0.5, criterion='distance')
        
        surviving_features = []
        for cluster_id in np.unique(clusters):
            cluster_features = [features[i] for i, c in enumerate(clusters) if c == cluster_id]
            X_test_perm = cfi_test[features].copy()
            shuffle_idx = np.random.permutation(len(X_test_perm))
            X_test_perm[cluster_features] = X_test_perm[cluster_features].values[shuffle_idx]
            
            perm_preds = 1.0 / (1.0 + np.exp(-cfi_bst.predict(xgb.DMatrix(X_test_perm))))
            perm_returns = (perm_preds > config.CONFIDENCE_THRESHOLD).astype(int) * cfi_test['close'].pct_change().fillna(0).values
            perm_sharpe = np.mean(perm_returns) / np.std(perm_returns) if np.std(perm_returns) > 0 else 0.0
            
            if (base_sharpe - perm_sharpe) > 0.02: # Feature group contains orthogonal alpha
                surviving_features.extend(cluster_features)

        if not surviving_features: surviving_features = features # Fallback if all decay
        
        logger.info(f"[{sector_name}] CFS pruned feature space from {len(features)} to {len(surviving_features)}.")

        # Train final candidate on orthogonalized feature subset
        d_full = xgb.DMatrix(sector_df[surviving_features], label=sector_df[target_col])
        candidate_booster = xgb.train(best_params, d_full, obj=asymmetric_financial_loss, num_boost_round=150)
        
        os.makedirs(config.PROD_MODELS_DIR, exist_ok=True)
        candidate_booster.save_model(os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate.json"))
        with open(os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate_features.json"), "w") as f:
            json.dump(surviving_features, f)
            
        pd.DataFrame(returns_matrix).to_parquet(f"returns_matrix_{sector_name}.parquet", engine='pyarrow')
        pd.DataFrame({"benchmark": benchmark_returns, "champion": returns_matrix[f"trial_{np.argmax([np.mean(returns_matrix[k])/np.std(returns_matrix[k]) if np.std(returns_matrix[k])>0 else 0 for k in returns_matrix])}"]}).to_parquet(f"benchmark_{sector_name}.parquet", engine='pyarrow')
        gc.collect()

    def execute_gauntlet(self) -> None:
        if not os.path.exists(config.PROCESSED_VAULT_DIR): return
        unique_sectors = self.ddf['sector'].unique().compute()
        for sector in unique_sectors:
            if not pd.isna(sector): self.tune_sector_grid(str(sector))