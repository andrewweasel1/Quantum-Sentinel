import os
import glob
import itertools
import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import jsharpe
import quantstats as qs

import config

# ==============================================================================
# 0. CENTRALIZED LOGGING CONFIGURATION
# ==============================================================================
logger = logging.getLogger(__name__)

class QuantitativeEvaluator:
    def __init__(self) -> None:
        """
        Initializes the Evaluation suite. It relies on the Parquet matrices
        and benchmark files exported by tournament.py.
        """
        self.confidence_level = 0.95
        self.pbo_partitions = 8

    def compute_trial_p_values(self, returns_matrix: pd.DataFrame) -> np.ndarray:
        """
        Calculates the p-value (1 - PSR) for every trial in the returns matrix
        to feed into the jsharpe False Discovery Rate (FDR) control.
        """
        p_values = []
        for col in returns_matrix.columns:
            returns = returns_matrix[col].values
            if np.std(returns) == 0:
                p_values.append(1.0)
                continue

            # Calculate Probabilistic Sharpe Ratio (PSR) for the trial
            try:
                psr = jsharpe.probabilistic_sharpe_ratio(returns)
            except Exception:
                sr = np.mean(returns) / np.std(returns)
                skewness = stats.skew(returns)
                kurt = stats.kurtosis(returns)
                n = len(returns)
                
                stat = sr * np.sqrt(n - 1)
                denom = np.sqrt(1 - skewness * sr + ((kurt - 1) / 4) * sr**2)
                psr = stats.norm.cdf(stat / denom)

            p_values.append(1.0 - float(psr))
            
        return np.array(p_values)

    def calculate_native_pbo(self, returns_matrix: pd.DataFrame) -> float:
        """
        Natively calculates the Probability of Backtest Overfitting (PBO) by 
        implementing the Bailey et al. (2015) combinatorial framework.
        """
        mat = returns_matrix.values
        T, N = mat.shape
        
        if T < self.pbo_partitions or N < 2:
            logger.warning("Not enough data or trials to compute PBO meaningfully.")
            return 0.0
            
        # 1. Split timeline into equal, contiguous partitions
        partitions = np.array_split(mat, self.pbo_partitions, axis=0)
        partition_indices = list(range(self.pbo_partitions))
        
        # 2. Form all Combinatorial In-Sample (IS) and Out-Of-Sample (OOS) paths
        is_combos = list(itertools.combinations(partition_indices, self.pbo_partitions // 2))
        
        logits = []
        for is_idx in is_combos:
            oos_idx = [i for i in partition_indices if i not in is_idx]
            
            is_mat = np.vstack([partitions[i] for i in is_idx])
            oos_mat = np.vstack([partitions[i] for i in oos_idx])
            
            # 3. Calculate Sharpe Ratios for both sets
            is_std = np.std(is_mat, axis=0)
            is_std[is_std == 0] = 1e-8  # Prevent division by zero
            is_sr = np.mean(is_mat, axis=0) / is_std
            
            oos_std = np.std(oos_mat, axis=0)
            oos_std[oos_std == 0] = 1e-8
            oos_sr = np.mean(oos_mat, axis=0) / oos_std
            
            # 4. Identify Best IS Strategy and find its OOS rank
            best_is_idx = np.argmax(is_sr)
            best_is_oos_sr = oos_sr[best_is_idx]
            
            # Rank is the fraction of configurations it beat out-of-sample
            rank = np.sum(oos_sr <= best_is_oos_sr) / N
            rank = np.clip(rank, 1e-5, 1.0 - 1e-5)
            
            # 5. Logit transformation
            logit = np.log(rank / (1.0 - rank))
            logits.append(logit)
            
        # 6. PBO is the fraction of configurations where OOS performance fell below the median
        pbo_value = np.sum(np.array(logits) < 0) / len(logits)
        return float(pbo_value)

    def assess_sector(self, sector_name: str) -> bool:
        """
        Evaluates a candidate model for a specific sector using institutional metrics.
        """
        logger.info(f"\n{'='*60}\nEvaluating Candidate for Sector: {sector_name}\n{'='*60}")

        matrix_file = f"returns_matrix_{sector_name}.parquet"
        bench_file = f"benchmark_{sector_name}.parquet"

        if not os.path.exists(matrix_file) or not os.path.exists(bench_file):
            logger.warning(f"Missing tournament data for {sector_name}. Skipping...")
            return False

        returns_matrix = pd.read_parquet(matrix_file)
        bench_df = pd.read_parquet(bench_file)
        
        champion_returns = bench_df['champion']
        benchmark_returns = bench_df['benchmark']

        # 1. Multiple Testing Bias (FDR)
        logger.info("1. Evaluating Multiple Testing Bias (FDR)...")
        p_values = self.compute_trial_p_values(returns_matrix)
        passes_fdr = jsharpe.control_for_FDR(p_values, alpha=1.0 - self.confidence_level)
        
        best_trial_idx = np.argmin(p_values)
        if not passes_fdr[best_trial_idx]:
            logger.warning(f"[{sector_name}] REJECTED: Failed False Discovery Rate (FDR) control.")
            return False
        logger.info(f"[{sector_name}] PASSED FDR Control.")

        # 2. Probability of Backtest Overfitting (PBO)
        logger.info("2. Evaluating Probability of Backtest Overfitting (PBO)...")
        try:
            overfit_prob = self.calculate_native_pbo(returns_matrix)
            logger.info(f"[{sector_name}] PBO: {overfit_prob * 100:.2f}%")
            
            if overfit_prob > 0.50:
                logger.warning(f"[{sector_name}] REJECTED: PBO exceeds 50% threshold.")
                return False
        except Exception as e:
            logger.error(f"[{sector_name}] PBO calculation failed: {e}")
            return False

        # 3. Minimum Track Record Length (MinTRL)
        logger.info("3. Evaluating Minimum Track Record Length (MinTRL)...")
        champ_sr = champion_returns.mean() / champion_returns.std()
        champ_skew = stats.skew(champion_returns)
        champ_kurt = stats.kurtosis(champion_returns)

        try:
            min_trl = jsharpe.minimum_track_record_length(
                sharpe_ratio=champ_sr,
                skewness=champ_skew,
                kurtosis=champ_kurt,
                confidence_level=self.confidence_level
            )
            logger.info(f"[{sector_name}] MinTRL Required: {min_trl:.2f} | Actual: {len(champion_returns)} observations.")
            
            if len(champion_returns) < min_trl:
                logger.warning(f"[{sector_name}] REJECTED: Track record length too short to prove significance.")
                return False
        except Exception as e:
            logger.error(f"[{sector_name}] MinTRL calculation failed: {e}")
            return False

        # 4. Generate Institutional Tearsheet & Promote
        logger.info(f"[{sector_name}] TRUE ALPHA DETECTED. Generating tearsheet and promoting to production.")
        try:
            qs.reports.html(
                returns=champion_returns, 
                benchmark=benchmark_returns, 
                title=f'Quantum Sentinel - {sector_name} Champion Profile', 
                output=f"tearsheet_{sector_name}.html",
                download_filename=f"tearsheet_{sector_name}.html"
            )
        except Exception as e:
            logger.error(f"[{sector_name}] Failed to generate QuantStats tearsheet: {e}")

        # Promote candidate model to champion
        candidate_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate.json")
        champion_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_champion.json")
        
        candidate_feat_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate_features.json")
        champion_feat_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_champion_features.json")
        
        if os.path.exists(candidate_path):
            os.replace(candidate_path, champion_path)
        if os.path.exists(candidate_feat_path):
            os.replace(candidate_feat_path, champion_feat_path)

        return True

    def run_evaluation_gauntlet(self) -> None:
        """
        Main execution pipeline. Scans for all sector matrices and evaluates them.
        """
        logger.info("=== COMMENCING POST-TOURNAMENT EVALUATION ===")
        
        matrix_files = glob.glob("returns_matrix_*.parquet")

        if not matrix_files:
            logger.error("No tournament returns matrices found. Ensure tournament.py completed successfully.")
            return

        approved_sectors = []
        for file in matrix_files:
            sector_name = file.replace("returns_matrix_", "").replace(".parquet", "")
            is_approved = self.assess_sector(sector_name)
            
            if is_approved:
                approved_sectors.append(sector_name)

        logger.info(f"\n=== EVALUATION CONCLUDED. {len(approved_sectors)} SECTORS APPROVED FOR PRODUCTION. ===")

if __name__ == "__main__":
    evaluator = QuantitativeEvaluator()
    evaluator.run_evaluation_gauntlet()
Removing the Package
You can now safely remove pypbo and its limiting statsmodels 0.8.0 requirement from your requirements.txt:
# Institutional Quantitative Metrics
jsharpe
quantstats
statsmodels>=0.13.0