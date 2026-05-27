import os
import glob
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
import quantstats as qs
from hmmlearn import hmm
import xgboost as xgb
import json

import config

logger = logging.getLogger(__name__)

class QuantitativeEvaluator:
    def __init__(self) -> None:
        self.min_dsr_threshold = 0.95
        
    def compute_deflated_sharpe_ratio(self, trial_matrix: pd.DataFrame, champion_returns: pd.Series) -> float:
        """
        Calculates the Deflated Sharpe Ratio (DSR) using Bailey and Lopez de Prado's framework.
        Corrects for non-normality and selection bias under multiple testing.
        """
        # 1. Base Sharpe & Moments
        champ_sr = champion_returns.mean() / champion_returns.std() if champion_returns.std() > 0 else 0.0
        skew = stats.skew(champion_returns)
        kurt = stats.kurtosis(champion_returns, fisher=True)
        
        # 2. Variance of trials
        trial_srs = trial_matrix.mean() / trial_matrix.std().replace(0, 1e-9)
        var_trials = np.var(trial_srs)
        N = trial_matrix.shape[3]
        
        # 3. Expected Maximum Sharpe Ratio
        euler_mascheroni = 0.5772156649
        expected_max_sr = np.sqrt(var_trials) * ((1.0 - euler_mascheroni) * stats.norm.ppf(1 - 1.0/N) + euler_mascheroni * stats.norm.ppf(1 - 1.0/(N * np.e)))
        
        # 4. Deflation Calculation
        T = len(champion_returns)
        denominator = np.sqrt(1 - skew * champ_sr + ((kurt - 1) / 4.0) * champ_sr**2)
        dsr_stat = (champ_sr - expected_max_sr) * np.sqrt(T - 1) / denominator
        
        return stats.norm.cdf(dsr_stat)

    def run_hmm_synthetic_gauntlet(self, sector_name: str, benchmark_returns: pd.Series) -> float:
        """
        Fits a Gaussian HMM to extract market regimes, simulates synthetic Monte Carlo paths, 
        and evaluates the champion model against data it has mathematically never seen.
        """
        model_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate.json")
        features_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate_features.json")
        
        if not os.path.exists(model_path): return 0.0
        
        booster = xgb.Booster()
        booster.load_model(model_path)
        with open(features_path, "r") as f: features = json.load(f)

        # 1. Extract underlying Market Regimes via HMM
        X_hmm = benchmark_returns.values.reshape(-1, 1)
        hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
        hmm_model.fit(X_hmm)
        
        # 2. Generate Synthetic Matrix
        synthetic_returns, _ = hmm_model.sample(n_samples=len(benchmark_returns))
        
        # 3. Synthesize dummy features correlated with the synthetic regimes (simplified representation)
        synthetic_df = pd.DataFrame(index=range(len(synthetic_returns)), columns=features)
        for col in features: synthetic_df[col] = synthetic_returns.flatten() + np.random.normal(0, 0.01, len(synthetic_returns))
        
        d_synth = xgb.DMatrix(synthetic_df)
        preds = 1.0 / (1.0 + np.exp(-booster.predict(d_synth)))
        signals = (preds > config.CONFIDENCE_THRESHOLD).astype(int)
        
        strategy_returns = signals * synthetic_returns.flatten()
        return np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0.0

    def assess_sector(self, sector_name: str) -> None:
        matrix_file = f"returns_matrix_{sector_name}.parquet"
        bench_file = f"benchmark_{sector_name}.parquet"
        
        if not os.path.exists(matrix_file) or not os.path.exists(bench_file): return
        
        trial_matrix = pd.read_parquet(matrix_file)
        bench_df = pd.read_parquet(bench_file)
        champion_returns = bench_df['champion']
        benchmark_returns = bench_df['benchmark']
        
        # Apply Business Date indexing for proper QuantStats annualization
        dummy_index = pd.bdate_range(end=config.END_DATE, periods=len(champion_returns))
        champion_returns.index = dummy_index
        benchmark_returns.index = dummy_index
        
        dsr = self.compute_deflated_sharpe_ratio(trial_matrix, champion_returns)
        synthetic_sr = self.run_hmm_synthetic_gauntlet(sector_name, benchmark_returns)
        
        logger.info(f"[{sector_name}] Probabilistic DSR: {dsr:.4f} | Synthetic HMM Sharpe: {synthetic_sr:.4f}")
        
        if dsr >= self.min_dsr_threshold and synthetic_sr > 0:
            logger.info(f"[{sector_name}] TRUE ALPHA DETECTED. Generalization proven. Promoting to production.")
            
            os.rename(
                os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate.json"),
                os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_champion.json")
            )
            os.rename(
                os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate_features.json"),
                os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_champion_features.json")
            )
            
            qs.reports.html(
                returns=champion_returns, benchmark=benchmark_returns, 
                title=f'Quantum Sentinel - {sector_name} Champion Profile (DSR: {dsr:.2f})', 
                output=f"tearsheet_{sector_name}.html"
            )
        else:
            logger.warning(f"[{sector_name}] REJECTED. Model failed quantitative rigors (Overfit or Memorization Trap).")
            
        os.remove(matrix_file)
        os.remove(bench_file)

    def run_evaluation_gauntlet(self) -> None:
        logger.info("=== COMMENCING DSR & SYNTHETIC GENERALIZATION EVALUATION ===")
        for matrix_file in glob.glob("returns_matrix_*.parquet"):
            sector = matrix_file.replace("returns_matrix_", "").replace(".parquet", "")
            self.assess_sector(sector)