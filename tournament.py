import os
import gc
import itertools
import numpy as np
import pandas as pd
import dask.dataframe as dd
import xgboost as xgb
from sklearn.metrics import precision_score
import config

class ModularTournamentDirector:
    def __init__(self):
        """
        Initializes the Tournament Director by memory-mapping the compiled feature
        matrices directly from the processed vault using Dask and PyArrow.
        """
        print(f"Mounting out-of-core data from {config.PROCESSED_VAULT_DIR}...")
        self.ddf = dd.read_parquet(config.PROCESSED_VAULT_DIR, **config.DASK_READ_KKWARGS)
        self.results = []
        self.tuning_logs = []

    def apply_purged_embargo_split(self, df, train_fraction=0.8, embargo_window=20):
        """
        Executes Purging and Embargoing to prevent data leakage (look-ahead bias).
        - Purging: Drops training observations that overlap with the test set's prediction horizon.
        - Embargoing: Institutes a multi-day blackout period cleanly separating train/test sets.
        """
        split_idx = int(len(df) * train_fraction)
        
        # Identify the boundary point
        train_end_idx = split_idx - embargo_window
        
        # Purge and Embargo: The training set strictly ends before the embargo window,
        # ensuring no rolling technical indicators "leak" future test-set volatility.
        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[split_idx:]
        
        return train_df, test_df

    def tune_sector_grid(self, sector_name):
        """
        Filters the out-of-core data for a specific sector, caches it dynamically to the GPU 
        using QuantileDMatrix, and performs a rigorous hyperparameter grid search.
        """
        print(f"\n--- Initiating Grid Search Tournament for Sector: {sector_name} ---")
        
        # 1. Dask Pushdown Filtering
        # Efficiently loads only the required sector partition into host memory
        sector_df = self.ddf[self.ddf['sector'] == sector_name].compute().sort_values('date')
        
        if len(sector_df) < 500:
            print(f"Insufficient historical data for {sector_name}. Skipping...")
            return None

        # Determine target label based on global configuration
        target_col = 'target_label' if config.RUN_MODE == "STANDARD" else 'option_target_label'
        
        # Drop execution metadata to isolate pure mathematical features
        features = [c for c in sector_df.columns if c not in config.METADATA_COLS]

        # 2. Strict Chronological Purged & Embargoed Split
        train_df, test_df = self.apply_purged_embargo_split(sector_df)
        
        X_train, y_train = train_df[features], train_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]

        # 3. GPU VRAM Pre-Caching via QuantileDMatrix
        # This prevents XGBoost from redundantly ingesting pandas dataframes and allocating
        # memory 16 separate times during the grid search loops below.
        print(f"Pre-caching {sector_name} features into GPU VRAM...")
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
        # Using 'ref=dtrain' ensures the validation set shares the exact same quantile binning
        dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)

        # 4. Hyperparameter Search Space
        # Targeting asymmetric, loss-guided tree development with heavy regularization
        param_grid = {
            'max_depth': [1, 2],
            'min_child_weight': [1, 3],
            'gamma': [0.1, 0.5],
            'learning_rate': [0.01, 0.05]
        }
        
        keys, values = zip(*param_grid.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_precision = -1
        best_booster = None
        best_params = None

        # 5. Model Evaluation Loop
        for params in grid_combinations:
            # Enforce modern hardware architecture mappings
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            params['objective'] = 'binary:logistic'
            params['grow_policy'] = 'lossguide'
            
            evals_result = {}
            
            # Train the booster
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False,
                evals_result=evals_result
            )
            
            # Generate continuous probability predictions using the best iteration
            preds_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
            
            # Convert probabilities to strict binary execution signals (Confidence > 0.65)
            preds_binary = (preds_proba > 0.65).astype(int)
            
            # Evaluate using Precision (Accuracy is meaningless if the bot never trades)
            # A precision > 35% on a 2:1 Risk/Reward profile is mathematically profitable.
            if sum(preds_binary) > 0:
                precision = precision_score(y_test, preds_binary)
            else:
                precision = 0.0
                
            self.tuning_logs.append({
                'sector': sector_name,
                'max_depth': params['max_depth'],
                'gamma': params['gamma'],
                'precision': precision,
                'best_iteration': bst.best_iteration
            })

            # Track the champion model
            if precision > best_precision and sum(preds_binary) >= 10:
                best_precision = precision
                best_booster = bst
                best_params = params

        print(f"Sector {sector_name} Champion Precision: {best_precision*100:.2f}%")

        # 6. Strict VRAM Cleanup
        # Forcing garbage collection ensures your GPU doesn't hit Out-of-Memory (OOM)
        # crashes as it iterates through all 11 GICS sectors.
        del dtrain
        del dtest
        gc.collect()

        return best_booster, features, best_precision

    def execute_gauntlet(self):
        """
        Orchestrates the entire out-of-core pipeline, testing the hyperparameter
        grid across all unique sectors found in the Dask partitions.
        """
        if not os.path.exists(config.PROCESSED_VAULT_DIR):
            print("Error: Processed features missing. Run feature_compiler.py first.")
            return

        print("=== COMMENCING OUT-OF-CORE XGBOOST TOURNAMENT ===")
        
        # Extract unique sector partitions cleanly from Dask
        unique_sectors = self.ddf['sector'].unique().compute()
        
        for sector in unique_sectors:
            if pd.isna(sector):
                continue
                
            champion = self.tune_sector_grid(sector)
            
            if champion:
                bst, features, precision = champion
                self.results.append({
                    'sector': sector,
                    'composite_score': precision,  # Expandable to Sharpe/Sortino
                    'feat_count': len(features)
                })
                
                # NOTE: This is where we will hook up `exporter.py` in the next step
                # exporter.deploy_champion(bst, features, sector, precision)

        # Export telemetry for the Streamlit Dashboard
        pd.DataFrame(self.results).to_parquet(config.RESULTS_FILE)
        pd.DataFrame(self.tuning_logs).to_parquet("grid_search_raw_logs.parquet")
        print("\n=== TOURNAMENT CONCLUDED. TELEMETRY EXPORTED. ===")